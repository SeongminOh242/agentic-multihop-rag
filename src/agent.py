from __future__ import annotations

import json
import re
from typing import Any

from src.retriever import BM25Retriever
from src.reranker import CrossEncoderReranker

AGENT_PROMPT = """You are a research agent answering complex questions by searching documents iteratively.

Accumulated context so far:
{context}

Original question: {question}
Current hop: {hop}/{max_hops}

Based on the context, decide:
1. If you have enough information, set "sufficient": true and provide "final_answer" as a SHORT phrase (1-5 words, no explanation).
2. If you need more information, set "sufficient": false and provide a specific "sub_query" to search next.

Respond ONLY with valid JSON:
{{"sub_query": "...", "sufficient": false}}
OR
{{"sub_query": "", "sufficient": true, "final_answer": "short answer here"}}
"""

_MAX_CONTEXT_CHARS = 3000  # cap accumulated context sent to LLM to avoid hitting token limits


class AgenticRAG:
    def __init__(
        self,
        corpus: list[str],
        top_k: int = 5,
        max_hops: int = 4,
        min_hops: int = 2,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        llm_provider: str = "hf",
    ):
        self.retriever = BM25Retriever(corpus)
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k
        self.max_hops = max_hops
        self.min_hops = min_hops
        self.model_name = model_name
        self.llm_provider = llm_provider

    def _call_llm(self, prompt: str) -> str:
        from src.llm import get_llm

        return get_llm(
            model_name=self.model_name,
            provider=self.llm_provider,
        ).generate(prompt, max_new_tokens=512)

    def answer(self, question: str) -> dict[str, Any]:
        accumulated_context: list[str] = []
        all_retrieved_docs: list[dict] = []
        per_hop_docs: list[dict] = []
        hop_traces: list[dict[str, Any]] = []
        sub_queries: list[str] = [question]
        current_query = question

        for hop in range(1, self.max_hops + 1):
            raw_docs = self.retriever.retrieve(current_query, top_k=self.top_k * 2)
            reranked_docs = self.reranker.rerank(current_query, raw_docs, top_k=self.top_k)
            all_retrieved_docs.extend(reranked_docs)
            per_hop_docs.append({"query": current_query, "docs": reranked_docs})

            new_context = "\n".join(d["text"] for d in reranked_docs)
            accumulated_context.append(f"[Hop {hop} — Query: {current_query}]\n{new_context}")

            context_str = "\n\n".join(accumulated_context)
            if len(context_str) > _MAX_CONTEXT_CHARS:
                context_str = context_str[-_MAX_CONTEXT_CHARS:]
            prompt = AGENT_PROMPT.format(
                context=context_str,
                question=question,
                hop=hop,
                max_hops=self.max_hops,
            )
            raw_response = self._call_llm(prompt)
            trace_entry: dict[str, Any] = {
                "hop": hop,
                "query": current_query,
                "top_doc_titles": [
                    self._infer_doc_title(doc) for doc in reranked_docs[: self.top_k]
                ],
                "raw_response": raw_response,
            }

            try:
                decision = self._parse_json(raw_response)
                trace_entry["decision"] = decision
            except (json.JSONDecodeError, ValueError) as exc:
                trace_entry["parse_error"] = repr(exc)
                if hop >= self.min_hops:
                    decision = {"sufficient": True, "final_answer": ""}
                else:
                    decision = {"sufficient": False, "sub_query": question}
                trace_entry["decision"] = decision

            next_query = decision.get("sub_query") or question
            sufficient = decision.get("sufficient", False) or (
                hop >= self.min_hops and next_query.strip() == current_query.strip()
            )
            trace_entry["next_query"] = next_query
            trace_entry["sufficient"] = bool(sufficient)
            hop_traces.append(trace_entry)

            if sufficient and hop >= self.min_hops:
                final_answer = decision.get("final_answer", "")
                if not final_answer:
                    ctx = "\n\n".join(accumulated_context)
                    if len(ctx) > _MAX_CONTEXT_CHARS:
                        ctx = ctx[-_MAX_CONTEXT_CHARS:]
                    final_answer = self._call_llm(
                        f"Answer in a short phrase (1-5 words). No explanation.\n\n"
                        f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
                    )
                hop_traces[-1]["final_answer"] = final_answer
                return {
                    "answer": final_answer,
                    "num_hops": hop,
                    "retrieved_docs": all_retrieved_docs,
                    "per_hop_docs": per_hop_docs,
                    "sub_queries": sub_queries,
                    "hop_traces": hop_traces,
                }

            sub_queries.append(next_query)
            current_query = next_query

        # Max hops reached — force a final answer
        context_str = "\n\n".join(accumulated_context)
        if len(context_str) > _MAX_CONTEXT_CHARS:
            context_str = context_str[-_MAX_CONTEXT_CHARS:]
        fallback_prompt = (
            "Answer in a short phrase (1-5 words). No explanation.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
        )
        final_answer = self._call_llm(fallback_prompt)
        if hop_traces:
            hop_traces[-1]["final_answer"] = final_answer
        return {
            "answer": final_answer,
            "num_hops": self.max_hops,
            "retrieved_docs": all_retrieved_docs,
            "per_hop_docs": per_hop_docs,
            "sub_queries": sub_queries,
            "hop_traces": hop_traces,
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from LLM response, stripping markdown code fences if present."""
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)

    @staticmethod
    def _infer_doc_title(doc: dict[str, Any]) -> str:
        title = doc.get("title")
        if title:
            return str(title)
        text = str(doc.get("text", ""))
        if "." in text:
            return text.split(".", 1)[0].strip()
        return text[:80].strip()

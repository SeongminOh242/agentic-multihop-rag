from __future__ import annotations

import json
import os
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
1. If you have enough information, set "sufficient": true and provide "final_answer".
2. If you need more information, set "sufficient": false and provide a specific "sub_query" to search next.

Respond ONLY with valid JSON:
{{"sub_query": "...", "sufficient": false}}
OR
{{"sub_query": "", "sufficient": true, "final_answer": "..."}}
"""


class AgenticRAG:
    def __init__(self, corpus: list[str], top_k: int = 5, max_hops: int = 4, min_hops: int = 2):
        self.retriever = BM25Retriever(corpus)
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k
        self.max_hops = max_hops
        self.min_hops = min_hops
        self.client = self._build_client()

    def _call_llm(self, prompt: str) -> str:
        if self.client is not None:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        from src.llm import get_llm
        return get_llm().generate(prompt, max_new_tokens=256)

    def answer(self, question: str) -> dict[str, Any]:
        accumulated_context: list[str] = []
        all_retrieved_docs: list[dict] = []
        per_hop_docs: list[dict] = []
        sub_queries: list[str] = [question]
        current_query = question

        for hop in range(1, self.max_hops + 1):
            raw_docs = self.retriever.retrieve(current_query, top_k=self.top_k * 2)
            reranked_docs = self.reranker.rerank(current_query, raw_docs, top_k=self.top_k)
            all_retrieved_docs.extend(reranked_docs)
            per_hop_docs.append({"query": current_query, "docs": reranked_docs})

            new_context = "\n".join(d["text"] for d in reranked_docs)
            accumulated_context.append(f"[Hop {hop} — Query: {current_query}]\n{new_context}")

            prompt = AGENT_PROMPT.format(
                context="\n\n".join(accumulated_context),
                question=question,
                hop=hop,
                max_hops=self.max_hops,
            )
            raw_response = self._call_llm(prompt)

            try:
                decision = self._parse_json(raw_response)
            except (json.JSONDecodeError, ValueError):
                if hop >= self.min_hops:
                    decision = {"sufficient": True, "final_answer": ""}
                else:
                    decision = {"sufficient": False, "sub_query": question}

            next_query = decision.get("sub_query") or question
            sufficient = decision.get("sufficient", False) or (
                hop >= self.min_hops and next_query.strip() == current_query.strip()
            )

            if sufficient and hop >= self.min_hops:
                final_answer = decision.get("final_answer", "")
                if not final_answer:
                    context_str = "\n\n".join(accumulated_context)
                    final_answer = self._call_llm(
                        f"Answer concisely based on the context below.\n\n"
                        f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
                    )
                return {
                    "answer": final_answer,
                    "num_hops": hop,
                    "retrieved_docs": all_retrieved_docs,
                    "per_hop_docs": per_hop_docs,
                    "sub_queries": sub_queries,
                }

            sub_queries.append(next_query)
            current_query = next_query

        # Max hops reached — force a final answer
        context_str = "\n\n".join(accumulated_context)
        fallback_prompt = (
            "Based on this context, answer the question as best you can.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
        )
        final_answer = self._call_llm(fallback_prompt)
        return {
            "answer": final_answer,
            "num_hops": self.max_hops,
            "retrieved_docs": all_retrieved_docs,
            "per_hop_docs": per_hop_docs,
            "sub_queries": sub_queries,
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
    def _build_client():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        return OpenAI(api_key=api_key)

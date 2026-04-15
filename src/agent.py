from __future__ import annotations

import json
import os
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
    def __init__(self, corpus: list[str], top_k: int = 5, max_hops: int = 4):
        self.retriever = BM25Retriever(corpus)
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k
        self.max_hops = max_hops
        self.client = self._build_client()

    def _call_llm(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY first.")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

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
                decision = json.loads(raw_response)
            except json.JSONDecodeError:
                decision = {"sufficient": False, "sub_query": question}

            if decision.get("sufficient", False):
                return {
                    "answer": decision.get("final_answer", ""),
                    "num_hops": hop,
                    "retrieved_docs": all_retrieved_docs,
                    "per_hop_docs": per_hop_docs,
                    "sub_queries": sub_queries,
                }

            next_query = decision.get("sub_query") or question
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
    def _build_client():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        return OpenAI(api_key=api_key)

from __future__ import annotations

import os
from typing import Any

from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever


class StaticRAG:
    """Static single-hop RAG baseline for Week 1 experiments."""

    def __init__(self, corpus: list[str], top_k: int = 5, model_name: str = "gpt-4o-mini"):
        self.corpus = corpus
        self.retriever = BM25Retriever(corpus)
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k
        self.model_name = model_name
        self.client = self._build_client()

    def answer(self, question: str) -> dict[str, Any]:
        raw_docs = self.retriever.retrieve(question, top_k=self.top_k * 2)
        reranked_docs = self.reranker.rerank(question, raw_docs, top_k=self.top_k)
        context = "\n\n".join(document["text"] for document in reranked_docs)
        prompt = (
            "Answer the following question using only the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            answer = self._call_llm(prompt)
        except RuntimeError:
            answer = reranked_docs[0]["text"] if reranked_docs else ""

        return {
            "answer": answer,
            "retrieved_docs": reranked_docs,
            "num_hops": 1,
            "context": context,
        }

    def _call_llm(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY first.")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

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


BaselineRAG = StaticRAG

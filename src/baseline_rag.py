from __future__ import annotations

import traceback
from typing import Any

from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever


class StaticRAG:
    """Static single-hop RAG baseline for Week 1 experiments."""

    def __init__(
        self,
        corpus: list[str],
        top_k: int = 5,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        llm_provider: str = "hf",
    ):
        self.corpus = corpus
        self.retriever = BM25Retriever(corpus)
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k
        self.model_name = model_name
        self.llm_provider = llm_provider

    def answer(self, question: str) -> dict[str, Any]:
        raw_docs = self.retriever.retrieve(question, top_k=self.top_k * 2)
        reranked_docs = self.reranker.rerank(question, raw_docs, top_k=self.top_k)
        context = "\n\n".join(document["text"] for document in reranked_docs)
        prompt = (
            "Answer the following question using only the context below. "
            "Give a short answer (1-5 words). No explanation.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            answer = self._call_llm(prompt)
        except Exception as exc:
            print(f"[baseline LLM error] {exc!r}")
            traceback.print_exc()
            answer = reranked_docs[0]["text"] if reranked_docs else ""

        return {
            "answer": answer,
            "retrieved_docs": reranked_docs,
            "num_hops": 1,
            "context": context,
        }

    def _call_llm(self, prompt: str) -> str:
        from src.llm import get_llm

        return get_llm(
            model_name=self.model_name,
            provider=self.llm_provider,
        ).generate(prompt)


BaselineRAG = StaticRAG

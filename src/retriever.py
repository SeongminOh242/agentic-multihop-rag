from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .types import CorpusDocument

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(slots=True)
class RetrievedDocument:
    """Common retrieval output contract for baseline and agent pipelines."""

    document: CorpusDocument
    score: float
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Interface contract for concrete retriever implementations."""

    def retrieve(
        self,
        query: str,
        corpus: Sequence[CorpusDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        ...


class BaseRetriever:
    """Placeholder base class for the Week 1 retrieval implementation."""

    def retrieve(
        self,
        query: str,
        corpus: Sequence[CorpusDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        raise NotImplementedError("Implement BM25 retrieval in src/retriever.py.")


class BM25Retriever:
    """BM25-based retriever over a list of plain-text documents."""

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self._tokenized_corpus = [self._tokenize(document) for document in corpus]
        self._bm25 = self._build_bm25()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not self.corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._score_documents(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [
            {
                "text": self.corpus[index],
                "score": float(scores[index]),
                "doc_id": index,
            }
            for index in top_indices
        ]

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return None
        return BM25Okapi(self._tokenized_corpus)

    def _score_documents(self, tokenized_query: list[str]) -> list[float]:
        if self._bm25 is not None:
            return [float(score) for score in self._bm25.get_scores(tokenized_query)]

        query_terms = set(tokenized_query)
        scores: list[float] = []
        for tokens in self._tokenized_corpus:
            overlap = sum(1 for token in tokens if token in query_terms)
            scores.append(float(overlap))
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

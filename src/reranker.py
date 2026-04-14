from __future__ import annotations

import os
import re
from typing import Any
from typing import Sequence, Protocol

from .retriever import RetrievedDocument

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class Reranker(Protocol):
    """Interface contract for concrete reranker implementations."""

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        ...


class BaseReranker:
    """Placeholder base class for the Week 1 reranker implementation."""

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        raise NotImplementedError("Implement reranking in src/reranker.py.")


class PassthroughReranker:
    """Stable default used until a learned reranker is available."""

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        del query

        if top_k is None:
            selected = list(documents)
        else:
            selected = list(documents[:top_k])

        reranked: list[RetrievedDocument] = []
        for index, item in enumerate(selected, start=1):
            reranked.append(
                RetrievedDocument(
                    document=item.document,
                    score=item.score,
                    rank=index,
                    metadata=dict(item.metadata),
                )
            )

        return reranked


class CrossEncoderReranker:
    """
    Cross-encoder reranker with a lexical fallback.

    The fallback keeps tests and offline development usable when the transformer
    model is unavailable locally or cannot be downloaded yet.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        load_model: bool | None = None,
    ):
        self.model_name = model_name
        self.load_model = (
            os.getenv("ENABLE_CROSS_ENCODER", "0") == "1"
            if load_model is None
            else load_model
        )
        self._model: Any | None = None
        self._model_load_failed = False

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not docs:
            return []

        reranked_docs = [dict(doc) for doc in docs]
        scores = self._predict_scores(query, reranked_docs)

        for doc, score in zip(reranked_docs, scores, strict=True):
            doc["rerank_score"] = float(score)

        reranked_docs.sort(key=lambda doc: doc["rerank_score"], reverse=True)
        return reranked_docs[:top_k]

    def _predict_scores(self, query: str, docs: list[dict]) -> list[float]:
        model = self._get_model()
        if model is not None:
            pairs = [(query, doc["text"]) for doc in docs]
            try:
                return [float(score) for score in model.predict(pairs)]
            except Exception:
                self._model = None
                self._model_load_failed = True

        return [self._lexical_score(query, doc["text"]) for doc in docs]

    def _get_model(self):
        if not self.load_model:
            return None
        if self._model is not None or self._model_load_failed:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            self._model_load_failed = True
            return None

        try:
            self._model = CrossEncoder(self.model_name)
        except Exception:
            self._model_load_failed = True
            self._model = None

        return self._model

    @staticmethod
    def _lexical_score(query: str, document_text: str) -> float:
        query_terms = set(CrossEncoderReranker._tokenize(query))
        document_terms = CrossEncoderReranker._tokenize(document_text)
        overlap = sum(1 for token in document_terms if token in query_terms)
        return float(overlap)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

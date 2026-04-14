from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SupportingFact:
    """Sentence-level evidence annotation from HotpotQA."""

    title: str
    sentence_id: int


@dataclass(slots=True)
class CorpusDocument:
    """Single retrievable document assembled from a HotpotQA context entry."""

    doc_id: str
    title: str
    sentences: list[str]
    text: str
    source_sample_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HotpotExample:
    """Normalized HotpotQA example used throughout the pipelines."""

    sample_id: str
    question: str
    answer: str
    question_type: str
    level: str
    supporting_facts: list[SupportingFact]
    context_documents: list[CorpusDocument]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "type": self.question_type,
            "level": self.level,
            "supporting_facts": [
                {"title": fact.title, "sent_id": fact.sentence_id}
                for fact in self.supporting_facts
            ],
            "context_documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "sentences": list(doc.sentences),
                    "text": doc.text,
                    "source_sample_id": doc.source_sample_id,
                    "metadata": dict(doc.metadata),
                }
                for doc in self.context_documents
            ],
        }

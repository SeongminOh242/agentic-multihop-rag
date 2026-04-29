from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .types import CorpusDocument, HotpotExample, SupportingFact


def _load_dataset(
    dataset_name: str,
    subset: str,
    split: str,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "datasets is required to load HotpotQA. Install requirements.txt first."
        ) from exc

    return load_dataset(dataset_name, subset, split=split)


def load_hotpotqa(
    split: str = "validation",
    max_samples: int = 500,
    dataset_name: str = "hotpotqa/hotpot_qa",
    subset: str = "distractor",
) -> list[dict[str, Any]]:
    """Load HotpotQA samples using the simple dict format from the implementation plan."""

    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    dataset = _load_dataset(dataset_name=dataset_name, subset=subset, split=split)
    samples: list[dict[str, Any]] = []

    for index, item in enumerate(dataset):
        if index >= max_samples:
            break
        samples.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"],
                "id": item["id"],
                "type": item.get("type", ""),
                "level": item.get("level", ""),
            }
        )

    return samples


def build_corpus(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    """Flatten all context passages into a single deduplicated list of strings."""

    corpus: list[str] = []
    for sample in samples:
        context = sample["context"]
        titles = context["title"]
        sentences_list = context["sentences"]
        for title, sentences in zip(titles, sentences_list, strict=True):
            passage = f"{title}. {' '.join(sentences)}"
            corpus.append(passage)

    return list(dict.fromkeys(corpus))


class HotpotQADataLoader:
    """Loader and normalizer for the HotpotQA Hugging Face dataset."""

    def __init__(
        self,
        dataset_name: str = "hotpotqa/hotpot_qa",
        subset: str = "distractor",
    ) -> None:
        self.dataset_name = dataset_name
        self.subset = subset

    def load(
        self,
        split: str = "validation",
        sample_size: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> list[HotpotExample]:
        """Load a split from Hugging Face and return normalized examples."""

        if sample_size is not None and sample_size <= 0:
            raise ValueError("sample_size must be a positive integer when provided.")

        dataset = _load_dataset(
            dataset_name=self.dataset_name,
            subset=self.subset,
            split=split,
        )
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        if sample_size is not None:
            dataset = dataset.select(range(min(sample_size, len(dataset))))

        return self.parse_records(dataset)

    @staticmethod
    def parse_records(records: Sequence[Mapping[str, Any]]) -> list[HotpotExample]:
        return [HotpotQADataLoader.parse_record(record) for record in records]

    @staticmethod
    def parse_record(record: Mapping[str, Any]) -> HotpotExample:
        sample_id = str(record["id"])
        context_documents = HotpotQADataLoader._parse_context(
            sample_id=sample_id,
            context=record.get("context", {}),
        )
        supporting_facts = HotpotQADataLoader._parse_supporting_facts(
            supporting_facts=record.get("supporting_facts", {}),
        )

        return HotpotExample(
            sample_id=sample_id,
            question=str(record["question"]),
            answer=str(record["answer"]),
            question_type=str(record["type"]),
            level=str(record["level"]),
            supporting_facts=supporting_facts,
            context_documents=context_documents,
        )

    @staticmethod
    def build_corpus(
        examples: Sequence[HotpotExample],
        deduplicate: bool = False,
    ) -> list[CorpusDocument]:
        corpus: list[CorpusDocument] = []
        seen_keys: set[tuple[str, str]] = set()

        for example in examples:
            for document in example.context_documents:
                key = (document.title, document.text)
                if deduplicate and key in seen_keys:
                    continue
                seen_keys.add(key)
                corpus.append(document)

        return corpus

    @staticmethod
    def _parse_context(
        sample_id: str,
        context: Mapping[str, Any],
    ) -> list[CorpusDocument]:
        titles = list(context.get("title", []))
        sentences_by_title = list(context.get("sentences", []))

        if len(titles) != len(sentences_by_title):
            raise ValueError("HotpotQA context titles and sentences must have matching lengths.")

        documents: list[CorpusDocument] = []
        for index, (title, sentences) in enumerate(zip(titles, sentences_by_title, strict=True)):
            normalized_sentences = [
                str(sentence).strip()
                for sentence in sentences
                if str(sentence).strip()
            ]
            documents.append(
                CorpusDocument(
                    doc_id=f"{sample_id}:{index}",
                    title=str(title),
                    sentences=normalized_sentences,
                    text=" ".join(normalized_sentences),
                    source_sample_id=sample_id,
                    metadata={"context_index": index},
                )
            )

        return documents

    @staticmethod
    def _parse_supporting_facts(
        supporting_facts: Mapping[str, Any],
    ) -> list[SupportingFact]:
        titles = list(supporting_facts.get("title", []))
        sentence_ids = list(supporting_facts.get("sent_id", []))

        if len(titles) != len(sentence_ids):
            raise ValueError(
                "HotpotQA supporting_facts titles and sent_id must have matching lengths."
            )

        return [
            SupportingFact(title=str(title), sentence_id=int(sentence_id))
            for title, sentence_id in zip(titles, sentence_ids, strict=True)
        ]

from __future__ import annotations

import src.data_loader as data_loader


class FakeDataset(list):
    def shuffle(self, seed: int):
        del seed
        return self

    def select(self, indices):
        return FakeDataset([self[index] for index in indices])


def _fake_records() -> FakeDataset:
    return FakeDataset(
        [
            {
                "question": "Which city is home to the Eiffel Tower?",
                "answer": "Paris",
                "supporting_facts": {"title": ["Eiffel Tower"], "sent_id": [0]},
                "context": {
                    "title": ["Eiffel Tower", "Paris"],
                    "sentences": [
                        ["The Eiffel Tower is located in Paris, France."],
                        ["Paris is the capital of France."],
                    ],
                },
                "id": "sample-001",
                "type": "bridge",
                "level": "easy",
            },
            {
                "question": "What country contains Paris?",
                "answer": "France",
                "supporting_facts": {"title": ["Paris"], "sent_id": [0]},
                "context": {
                    "title": ["Paris"],
                    "sentences": [["Paris is in France."]],
                },
                "id": "sample-002",
                "type": "bridge",
                "level": "easy",
            },
        ]
    )


def test_load_hotpotqa_returns_list(monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "_load_dataset", lambda **kwargs: _fake_records())

    samples = data_loader.load_hotpotqa(split="validation", max_samples=2)

    assert isinstance(samples, list)
    assert len(samples) == 2


def test_sample_has_required_fields(monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "_load_dataset", lambda **kwargs: _fake_records())

    sample = data_loader.load_hotpotqa(split="validation", max_samples=1)[0]

    assert "question" in sample
    assert "answer" in sample
    assert "supporting_facts" in sample
    assert "context" in sample


def test_build_corpus_returns_list_of_strings(monkeypatch) -> None:
    monkeypatch.setattr(data_loader, "_load_dataset", lambda **kwargs: _fake_records())

    samples = data_loader.load_hotpotqa(split="validation", max_samples=2)
    corpus = data_loader.build_corpus(samples)

    assert isinstance(corpus, list)
    assert all(isinstance(document, str) for document in corpus)
    assert corpus[0].startswith("Eiffel Tower.")

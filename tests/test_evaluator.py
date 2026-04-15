from __future__ import annotations

from src.evaluator import (
    Evaluator,
    exact_match_score,
    f1_score,
    normalize_answer,
)
from src.types import HotpotExample, SupportingFact


def test_normalize_answer() -> None:
    assert normalize_answer("The United States") == "united states"
    assert normalize_answer("A cat.") == "cat"
    assert normalize_answer("An Apple,   Inc") == "apple inc"
    assert normalize_answer("   extra    spaces   ") == "extra spaces"
    assert normalize_answer("") == ""


def test_exact_match_score() -> None:
    assert exact_match_score("Paris, France", "Paris France") is True
    assert exact_match_score("a cat", "cat") is True
    assert exact_match_score("dog", "cat") is False


def test_f1_score() -> None:
    assert f1_score("Paris, France", "Paris France") == 1.0
    assert f1_score("Paris", "Paris France") == 0.6666666666666666  # 2 * (1/1 * 1/2) / (1/1 + 1/2) = 2/3
    assert f1_score("London", "Paris France") == 0.0
    # Both empty strings
    assert f1_score("", "") == 1.0
    # One empty string
    assert f1_score("word", "") == 0.0
    assert f1_score("", "word") == 0.0


def _fake_hotpot_example() -> HotpotExample:
    return HotpotExample(
        sample_id="test-1",
        question="Where is the Eiffel Tower?",
        answer="Paris, France",
        question_type="bridge",
        level="easy",
        supporting_facts=[
            SupportingFact(title="Eiffel Tower", sentence_id=0),
            SupportingFact(title="Paris", sentence_id=2),
        ],
        context_documents=[],
    )


def test_evaluator_perfect_match() -> None:
    evaluator = Evaluator()
    ground_truth = _fake_hotpot_example()

    response = {
        "answer": "Paris, France",
        "retrieved_docs": [
            {"title": "Eiffel Tower"},
            {"title": "Paris"},
            {"title": "London"},
        ],
    }

    metrics = evaluator.evaluate(response, ground_truth)
    assert metrics["exact_match"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["retrieval_recall"] == 1.0


def test_evaluator_missing_keys_safe() -> None:
    """Must safely handle cases where the agent crashed and returned empty payload."""
    evaluator = Evaluator()
    ground_truth = _fake_hotpot_example()

    response = {}  # Empty dict!

    metrics = evaluator.evaluate(response, ground_truth)
    assert metrics["exact_match"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["retrieval_recall"] == 0.0


def test_evaluator_partial_retrieval() -> None:
    """If top_k cuts off correctly identifying docs, recall should show."""
    evaluator = Evaluator()
    ground_truth = _fake_hotpot_example()

    response = {
        "answer": "Paris",
        "retrieved_docs": [
            {"title": "Eiffel Tower"},
            # Paris is missing because top_k boundary clipped it maybe
        ],
    }

    metrics = evaluator.evaluate(response, ground_truth)
    assert metrics["retrieval_recall"] == 0.5  # 1 out of 2 expected titles
    # f1 for "Paris" vs "Paris France" == 2/3 ~ 0.666
    assert round(metrics["f1"], 2) == 0.67


def test_evaluator_baseline_dict_shape() -> None:
    """Handles baseline output which doesn't specify 'title' natively in dict but has it in text."""
    evaluator = Evaluator()
    ground_truth = _fake_hotpot_example()

    response = {
        "answer": "Paris",
        "retrieved_docs": [
            {"text": "Eiffel Tower. The Eiffel tower is tall.", "score": 9.2, "doc_id": 1},
        ],
    }

    metrics = evaluator.evaluate(response, ground_truth)
    assert metrics["retrieval_recall"] == 0.5  # Should parse "Eiffel Tower" from text

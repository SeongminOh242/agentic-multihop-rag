from __future__ import annotations

from src.evaluator import (
    Evaluator,
    evaluate_batch,
    exact_match_score,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    normalize_answer,
)
from src.types import HotpotExample, SupportingFact


def test_normalize_answer() -> None:
    assert normalize_answer("The United States") == "united states"
    assert normalize_answer("A cat.") == "cat"
    assert normalize_answer("An Apple,   Inc") == "apple inc"
    assert normalize_answer("   extra    spaces   ") == "extra spaces"
    assert normalize_answer("") == ""


def test_exact_match_true() -> None:
    assert exact_match_score("Paris", "Paris") is True


def test_exact_match_case_insensitive() -> None:
    assert exact_match_score("paris", "Paris") is True


def test_exact_match_false() -> None:
    assert exact_match_score("London", "Paris") is False


def test_f1_score_partial_overlap() -> None:
    score = f1_score("Paris France", "Paris is in France")
    assert 0.0 < score < 1.0


def test_f1_score_exact() -> None:
    assert f1_score("Paris", "Paris") == 1.0


def test_f1_score_no_overlap() -> None:
    assert f1_score("London", "Tokyo") == 0.0


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


def test_mrr_first_relevant() -> None:
    # Relevant doc is at position 0 → MRR = 1.0
    retrieved = [{"doc_id": 0}, {"doc_id": 1}, {"doc_id": 2}]
    relevant = {0}
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0


def test_mrr_second_relevant() -> None:
    retrieved = [{"doc_id": 1}, {"doc_id": 0}, {"doc_id": 2}]
    relevant = {0}
    assert abs(mean_reciprocal_rank(retrieved, relevant) - 0.5) < 1e-6


def test_ndcg_at_k_perfect() -> None:
    retrieved = [{"doc_id": 0}, {"doc_id": 1}]
    relevant = {0, 1}
    score = ndcg_at_k(retrieved, relevant, k=2)
    assert abs(score - 1.0) < 1e-6


def test_ndcg_at_k_none_relevant() -> None:
    retrieved = [{"doc_id": 0}, {"doc_id": 1}]
    relevant = {99}
    assert ndcg_at_k(retrieved, relevant, k=2) == 0.0


def test_evaluate_batch() -> None:
    results = [
        {
            "answer": "Paris",
            "num_hops": 2,
            "retrieved_docs": [{"doc_id": 10}, {"doc_id": 1}],
            "per_hop_docs": [
                {"query": "tower", "docs": [{"doc_id": 5}, {"doc_id": 8}]},
                {"query": "city", "docs": [{"doc_id": 10}, {"doc_id": 1}]},
            ],
        }
    ]
    ground_truths = [
        {
            "answer": "Paris France",
            "supporting_doc_ids": {10, 11},
        }
    ]

    metrics = evaluate_batch(results, ground_truths)
    assert "EM" in metrics
    assert "F1" in metrics
    assert metrics["MRR"] == 1.0  # doc_id 10 is rank 1
    assert metrics["avg_hops"] == 2.0
    assert "per_hop_MRR" in metrics

from __future__ import annotations

from src.analysis import (
    build_trace_record,
    first_hop_has_all_supporting_docs,
    requires_multiple_documents,
    summarize_question_slices,
)


def _fake_sample() -> dict:
    return {
        "id": "sample-1",
        "question": "Which state is the birthplace city of Einstein located in?",
        "answer": "Baden-Württemberg",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {
            "title": ["Albert Einstein", "Ulm"],
            "sent_id": [0, 0],
        },
    }


def test_requires_multiple_documents_true() -> None:
    assert requires_multiple_documents(_fake_sample()) is True


def test_build_trace_record_contains_intermediate_agent_data() -> None:
    sample = _fake_sample()
    result = {
        "answer": "Baden-Württemberg",
        "num_hops": 2,
        "sub_queries": ["Einstein birthplace", "Ulm state"],
        "retrieved_docs": [
            {"doc_id": 10, "title": "Albert Einstein", "text": "Albert Einstein. Born in Ulm."},
            {"doc_id": 11, "title": "Ulm", "text": "Ulm. Ulm is in Baden-Württemberg."},
        ],
        "per_hop_docs": [
            {"query": "Einstein birthplace", "docs": [{"doc_id": 10, "title": "Albert Einstein", "text": "Albert Einstein. Born in Ulm."}]},
            {"query": "Ulm state", "docs": [{"doc_id": 11, "title": "Ulm", "text": "Ulm. Ulm is in Baden-Württemberg."}]},
        ],
        "hop_traces": [
            {"hop": 1, "query": "Einstein birthplace", "decision": {"sufficient": False}},
            {"hop": 2, "query": "Ulm state", "decision": {"sufficient": True, "final_answer": "Baden-Württemberg"}},
        ],
    }

    trace = build_trace_record(sample, result, relevant_ids={10, 11}, system_name="agentic")

    assert trace["requires_multiple_documents"] is True
    assert trace["num_hops"] == 2
    assert trace["retrieval_hop_need"] == "needs_followup_hops"
    assert len(trace["per_hop"]) == 2
    assert len(trace["hop_traces"]) == 2
    assert trace["per_hop"][0]["query"] == "Einstein birthplace"


def test_summarize_question_slices_groups_by_support_doc_need() -> None:
    samples = [
        _fake_sample(),
        {
            "id": "sample-2",
            "question": "Where is the Eiffel Tower?",
            "answer": "Paris",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["Eiffel Tower"], "sent_id": [0]},
        },
    ]
    results = [
        {"answer": "Baden-Württemberg", "retrieved_docs": [{"doc_id": 10}, {"doc_id": 11}], "num_hops": 2, "per_hop_docs": []},
        {"answer": "Paris", "retrieved_docs": [{"doc_id": 20}], "num_hops": 1, "per_hop_docs": []},
    ]
    ground_truths = [
        {"answer": "Baden-Württemberg", "supporting_doc_ids": {10, 11}},
        {"answer": "Paris", "supporting_doc_ids": {20}},
    ]

    summary = summarize_question_slices(results, ground_truths, samples)

    assert "support_docs_needed" in summary
    assert "multiple_support_docs" in summary["support_docs_needed"]
    assert "single_support_doc" in summary["support_docs_needed"]


def test_first_hop_has_all_supporting_docs() -> None:
    result = {
        "retrieved_docs": [{"doc_id": 10}, {"doc_id": 11}],
        "per_hop_docs": [
            {"query": "first", "docs": [{"doc_id": 10}, {"doc_id": 11}]},
            {"query": "second", "docs": [{"doc_id": 12}]},
        ],
    }

    assert first_hop_has_all_supporting_docs(result, {10, 11}) is True
    assert first_hop_has_all_supporting_docs(result, {10, 12}) is False


def test_summarize_question_slices_groups_by_retrieval_hop_need() -> None:
    samples = [_fake_sample(), _fake_sample()]
    results = [
        {
            "answer": "Baden-Württemberg",
            "retrieved_docs": [{"doc_id": 10}, {"doc_id": 11}],
            "num_hops": 1,
            "per_hop_docs": [{"query": "q", "docs": [{"doc_id": 10}, {"doc_id": 11}]}],
        },
        {
            "answer": "Baden-Württemberg",
            "retrieved_docs": [{"doc_id": 10}, {"doc_id": 11}],
            "num_hops": 2,
            "per_hop_docs": [
                {"query": "q", "docs": [{"doc_id": 10}]},
                {"query": "q2", "docs": [{"doc_id": 11}]},
            ],
        },
    ]
    ground_truths = [
        {"answer": "Baden-Württemberg", "supporting_doc_ids": {10, 11}},
        {"answer": "Baden-Württemberg", "supporting_doc_ids": {10, 11}},
    ]

    summary = summarize_question_slices(results, ground_truths, samples)

    assert "retrieval_hop_need" in summary
    assert summary["retrieval_hop_need"]["single_hop_sufficient"]["count"] == 1.0
    assert summary["retrieval_hop_need"]["needs_followup_hops"]["count"] == 1.0

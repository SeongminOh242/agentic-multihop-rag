from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from .evaluator import evaluate_batch, exact_match_score, f1_score, mean_reciprocal_rank, ndcg_at_k


def requires_multiple_documents(sample: dict[str, Any]) -> bool:
    titles = sample.get("supporting_facts", {}).get("title", [])
    return len(set(titles)) > 1


def infer_title_from_doc(doc: Any) -> str:
    if isinstance(doc, dict):
        title = doc.get("title")
        if title:
            return str(title)
        text = str(doc.get("text", ""))
    else:
        title = getattr(doc, "title", None)
        if title:
            return str(title)
        text = str(getattr(doc, "text", ""))

    if "." in text:
        return text.split(".", 1)[0].strip()
    return text[:80].strip()


def serialize_docs(docs: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for doc in docs[:top_k]:
        serialized.append(
            {
                "doc_id": doc.get("doc_id"),
                "title": infer_title_from_doc(doc),
                "score": doc.get("score"),
                "rerank_score": doc.get("rerank_score"),
                "text_preview": str(doc.get("text", ""))[:220],
            }
        )
    return serialized


def first_hop_has_all_supporting_docs(
    result: dict[str, Any],
    relevant_ids: set[int],
) -> bool:
    """Whether a single retrieval pass already found every gold support document."""

    if not relevant_ids:
        return False

    per_hop_docs = result.get("per_hop_docs", [])
    if per_hop_docs:
        first_hop_docs = per_hop_docs[0].get("docs", [])
    else:
        first_hop_docs = result.get("retrieved_docs", [])

    first_hop_ids = {
        doc.get("doc_id")
        for doc in first_hop_docs
        if isinstance(doc, dict) and doc.get("doc_id") is not None
    }
    return relevant_ids.issubset(first_hop_ids)


def build_trace_record(
    sample: dict[str, Any],
    result: dict[str, Any],
    relevant_ids: set[int],
    system_name: str,
) -> dict[str, Any]:
    answer = result.get("answer") or ""
    gold_answer = sample.get("answer", "")
    supporting_titles = list(dict.fromkeys(sample.get("supporting_facts", {}).get("title", [])))
    retrieved_docs = result.get("retrieved_docs", [])
    first_hop_sufficient = first_hop_has_all_supporting_docs(result, relevant_ids)

    return {
        "system": system_name,
        "sample_id": sample.get("id"),
        "question": sample.get("question", ""),
        "gold_answer": gold_answer,
        "predicted_answer": answer,
        "exact_match": float(exact_match_score(answer, gold_answer)),
        "f1": f1_score(answer, gold_answer),
        "num_hops": result.get("num_hops", 1),
        "question_type": sample.get("type", ""),
        "level": sample.get("level", ""),
        "requires_multiple_documents": requires_multiple_documents(sample),
        "first_hop_sufficient": first_hop_sufficient,
        "retrieval_hop_need": (
            "single_hop_sufficient"
            if first_hop_sufficient
            else "needs_followup_hops"
        ),
        "supporting_titles": supporting_titles,
        "sub_queries": list(result.get("sub_queries", [])),
        "hop_traces": list(result.get("hop_traces", [])),
        "fallback_error": result.get("fallback_error"),
        "final_mrr": mean_reciprocal_rank(retrieved_docs, relevant_ids),
        "final_ndcg@10": ndcg_at_k(retrieved_docs, relevant_ids, k=10),
        "retrieved_docs": serialize_docs(retrieved_docs),
        "per_hop": [
            {
                "query": hop.get("query", ""),
                "docs": serialize_docs(hop.get("docs", [])),
            }
            for hop in result.get("per_hop_docs", [])
        ],
    }


def summarize_by_group(
    results: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    group_fn: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, float]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped_indices[group_fn(sample)].append(index)

    summary: dict[str, dict[str, float]] = {}
    for group_name, indices in grouped_indices.items():
        grouped_results = [results[index] for index in indices]
        grouped_truths = [ground_truths[index] for index in indices]
        metrics = evaluate_batch(grouped_results, grouped_truths)
        metrics["count"] = float(len(indices))
        summary[group_name] = metrics

    return summary


def summarize_by_result_group(
    results: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    group_fn: Callable[[dict[str, Any], dict[str, Any]], str],
) -> dict[str, dict[str, float]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, (result, ground_truth) in enumerate(zip(results, ground_truths, strict=True)):
        grouped_indices[group_fn(result, ground_truth)].append(index)

    summary: dict[str, dict[str, float]] = {}
    for group_name, indices in grouped_indices.items():
        grouped_results = [results[index] for index in indices]
        grouped_truths = [ground_truths[index] for index in indices]
        metrics = evaluate_batch(grouped_results, grouped_truths)
        metrics["count"] = float(len(indices))
        summary[group_name] = metrics

    return summary


def summarize_question_slices(
    results: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "support_docs_needed": summarize_by_group(
            results,
            ground_truths,
            samples,
            lambda sample: (
                "multiple_support_docs"
                if requires_multiple_documents(sample)
                else "single_support_doc"
            ),
        ),
        "retrieval_hop_need": summarize_by_result_group(
            results,
            ground_truths,
            lambda result, ground_truth: (
                "single_hop_sufficient"
                if first_hop_has_all_supporting_docs(
                    result,
                    ground_truth.get("supporting_doc_ids", set()),
                )
                else "needs_followup_hops"
            ),
        ),
        "question_type": summarize_by_group(
            results,
            ground_truths,
            samples,
            lambda sample: str(sample.get("type", "") or "unknown"),
        ),
        "difficulty_level": summarize_by_group(
            results,
            ground_truths,
            samples,
            lambda sample: str(sample.get("level", "") or "unknown"),
        ),
    }


def select_case_studies(
    traces: list[dict[str, Any]],
    limit_per_group: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "successful_multi_doc": [],
        "failed_multi_doc": [],
        "successful_single_doc": [],
        "failed_single_doc": [],
    }

    for trace in traces:
        if trace["requires_multiple_documents"]:
            key = "successful_multi_doc" if trace["exact_match"] else "failed_multi_doc"
        else:
            key = "successful_single_doc" if trace["exact_match"] else "failed_single_doc"

        if len(groups[key]) >= limit_per_group:
            continue

        groups[key].append(
            {
                "sample_id": trace["sample_id"],
                "question": trace["question"],
                "gold_answer": trace["gold_answer"],
                "predicted_answer": trace["predicted_answer"],
                "num_hops": trace["num_hops"],
                "supporting_titles": trace["supporting_titles"],
                "retrieval_hop_need": trace["retrieval_hop_need"],
                "sub_queries": trace["sub_queries"],
                "hop_traces": trace["hop_traces"],
                "per_hop": trace["per_hop"],
            }
        )

    return groups

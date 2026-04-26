from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional outside the project env
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False

from src.analysis import build_trace_record, select_case_studies, summarize_question_slices
from src.agent import AgenticRAG
from src.baseline_rag import StaticRAG
from src.data_loader import build_corpus, ensure_hf_hub_env, load_hotpotqa
from src.evaluator import evaluate_batch

load_dotenv()
ensure_hf_hub_env()


def build_ground_truths(samples: list[dict[str, Any]], corpus: list[str]) -> list[dict[str, Any]]:
    """Map supporting-fact titles onto corpus doc_ids for retrieval metrics."""

    text_to_doc_id = {document_text: index for index, document_text in enumerate(corpus)}
    ground_truths: list[dict[str, Any]] = []

    for sample in samples:
        supporting_titles = set(sample["supporting_facts"]["title"])
        context = sample["context"]

        supporting_doc_ids: set[int] = set()
        for title, sentences in zip(context["title"], context["sentences"], strict=True):
            if title not in supporting_titles:
                continue
            passage = f"{title}. {' '.join(sentences)}"
            doc_id = text_to_doc_id.get(passage)
            if doc_id is not None:
                supporting_doc_ids.add(doc_id)

        ground_truths.append(
            {
                "answer": sample["answer"],
                "supporting_doc_ids": supporting_doc_ids,
            }
        )

    return ground_truths


def _fallback_docs(pipeline: Any, question: str) -> list[dict[str, Any]]:
    raw_docs = pipeline.retriever.retrieve(question, top_k=pipeline.top_k * 2)
    return pipeline.reranker.rerank(question, raw_docs, top_k=pipeline.top_k)


def _is_resource_exhausted_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.lower()
    return "resource_exhausted" in normalized or "prepayment credits are depleted" in normalized


def _baseline_answer_with_fallback(baseline: StaticRAG, question: str) -> dict[str, Any]:
    try:
        return baseline.answer(question)
    except Exception as exc:
        print(f"[baseline fallback] {exc!r}")
        traceback.print_exc()
        reranked_docs = _fallback_docs(baseline, question)
        context = "\n\n".join(document["text"] for document in reranked_docs)
        return {
            "answer": reranked_docs[0]["text"] if reranked_docs else "",
            "retrieved_docs": reranked_docs,
            "num_hops": 1,
            "context": context,
            "fallback_error": repr(exc),
        }


def _agent_answer_with_fallback(agent: AgenticRAG, question: str) -> dict[str, Any]:
    try:
        return agent.answer(question)
    except Exception as exc:
        print(f"[agentic fallback] {exc!r}")
        if not _is_resource_exhausted_error(repr(exc)):
            traceback.print_exc()
        reranked_docs = _fallback_docs(agent, question)
        return {
            "answer": reranked_docs[0]["text"] if reranked_docs else "",
            "num_hops": 1,
            "retrieved_docs": reranked_docs,
            "per_hop_docs": [{"query": question, "docs": reranked_docs}],
            "sub_queries": [question],
            "hop_traces": [],
            "fallback_error": repr(exc),
        }


def _run_pipeline(
    pipeline: Any,
    questions: list[str],
    answer_fn: Any,
) -> list[dict[str, Any]]:
    return [answer_fn(pipeline, question) for question in questions]


def _build_trace_bundle(
    system_name: str,
    samples: list[dict[str, Any]],
    results: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, Any]:
    traces = [
        build_trace_record(
            sample=sample,
            result=result,
            relevant_ids=ground_truth.get("supporting_doc_ids", set()),
            system_name=system_name,
        )
        for sample, result, ground_truth in zip(samples, results, ground_truths, strict=True)
    ]

    return {
        "aggregate": evaluate_batch(results, ground_truths),
        "breakdowns": summarize_question_slices(results, ground_truths, samples),
        "case_studies": select_case_studies(traces),
        "traces": traces,
    }


def run(
    num_samples: int = 50,
    baseline_provider: str = "hf",
    baseline_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    agent_provider: str = "hf",
    agent_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    frontier_provider: str | None = None,
    frontier_model_name: str = "gemini-2.0-flash",
    frontier_num_samples: int | None = None,
) -> dict[str, Any]:
    ensure_hf_hub_env()
    samples = load_hotpotqa(split="validation", max_samples=num_samples)
    corpus = build_corpus(samples)
    ground_truths = build_ground_truths(samples, corpus)
    questions = [sample["question"] for sample in samples]

    print(f"Running baseline RAG on {num_samples} samples...")
    baseline = StaticRAG(
        corpus,
        top_k=5,
        model_name=baseline_model_name,
        llm_provider=baseline_provider,
    )
    baseline_results = _run_pipeline(baseline, questions, _baseline_answer_with_fallback)
    baseline_bundle = _build_trace_bundle("baseline", samples, baseline_results, ground_truths)

    print(f"Running Agentic RAG on {num_samples} samples...")
    agent = AgenticRAG(
        corpus,
        top_k=5,
        max_hops=4,
        min_hops=2,
        model_name=agent_model_name,
        llm_provider=agent_provider,
    )
    agent_results = _run_pipeline(agent, questions, _agent_answer_with_fallback)
    agent_bundle = _build_trace_bundle("agentic", samples, agent_results, ground_truths)

    report = {
        "metadata": {
            "num_samples": num_samples,
            "baseline_provider": baseline_provider,
            "baseline_model_name": baseline_model_name,
            "agent_provider": agent_provider,
            "agent_model_name": agent_model_name,
        },
        "baseline": baseline_bundle["aggregate"],
        "agentic": agent_bundle["aggregate"],
    }

    results_path = Path("results/experiment_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")

    detailed_report = {
        "metadata": report["metadata"],
        "baseline": baseline_bundle,
        "agentic": agent_bundle,
    }

    if frontier_provider is not None:
        resolved_frontier_num_samples = frontier_num_samples or num_samples
        frontier_samples_subset = samples[:resolved_frontier_num_samples]
        frontier_ground_truths_subset = ground_truths[:resolved_frontier_num_samples]
        frontier_questions = questions[:resolved_frontier_num_samples]

        print(f"Running frontier Agentic RAG on {resolved_frontier_num_samples} samples...")
        frontier_agent = AgenticRAG(
            corpus,
            top_k=5,
            max_hops=4,
            min_hops=2,
            model_name=frontier_model_name,
            llm_provider=frontier_provider,
        )
        report["metadata"]["frontier_provider"] = frontier_provider
        report["metadata"]["frontier_model_name"] = frontier_model_name
        report["metadata"]["frontier_num_samples"] = resolved_frontier_num_samples

        frontier_results: list[dict[str, Any]] = []
        if frontier_questions:
            first_frontier_result = _agent_answer_with_fallback(frontier_agent, frontier_questions[0])
            if _is_resource_exhausted_error(first_frontier_result.get("fallback_error")):
                print(
                    "[frontier skipped] Gemini returned RESOURCE_EXHAUSTED. "
                    "Add credits or rerun with frontier_provider=None."
                )
                report["metadata"]["frontier_status"] = "skipped_resource_exhausted"
            else:
                frontier_results.append(first_frontier_result)
                for frontier_question in frontier_questions[1:]:
                    frontier_result = _agent_answer_with_fallback(
                        frontier_agent,
                        frontier_question,
                    )
                    if _is_resource_exhausted_error(frontier_result.get("fallback_error")):
                        print(
                            "[frontier stopped] Gemini credits were exhausted mid-run. "
                            "Reporting the completed frontier samples only."
                        )
                        report["metadata"]["frontier_status"] = "partial_resource_exhausted"
                        break
                    frontier_results.append(frontier_result)

        if frontier_results:
            frontier_bundle = _build_trace_bundle(
                "frontier_agentic",
                frontier_samples_subset[: len(frontier_results)],
                frontier_results,
                frontier_ground_truths_subset[: len(frontier_results)],
            )
            detailed_report["frontier_agentic"] = frontier_bundle
            report["frontier_agentic"] = frontier_bundle["aggregate"]
            report["metadata"]["frontier_completed_samples"] = len(frontier_results)

        results_path.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")

    details_path = Path("results/experiment_details.json")
    details_path.write_text(f"{json.dumps(detailed_report, indent=2)}\n", encoding="utf-8")

    print("\n=== RESULTS ===")
    header = f"{'Metric':<15} {'Baseline':>10} {'Agentic':>10}"
    if "frontier_agentic" in report:
        header += f" {'Frontier':>10}"
    print(header)
    print("-" * len(header))
    for metric in ["EM", "F1", "MRR", "NDCG@10", "avg_hops"]:
        row = (
            f"{metric:<15}"
            f" {baseline_bundle['aggregate'][metric]:>10.4f}"
            f" {agent_bundle['aggregate'][metric]:>10.4f}"
        )
        if "frontier_agentic" in report:
            row += f" {report['frontier_agentic'][metric]:>10.4f}"
        print(row)

    return report


if __name__ == "__main__":
    run(num_samples=50)

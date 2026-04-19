from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent import AgenticRAG
from src.baseline_rag import StaticRAG
from src.data_loader import build_corpus, load_hotpotqa
from src.evaluator import evaluate_batch

load_dotenv()


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


def _baseline_answer_with_fallback(baseline: StaticRAG, question: str) -> dict[str, Any]:
    try:
        return baseline.answer(question)
    except Exception:
        reranked_docs = _fallback_docs(baseline, question)
        context = "\n\n".join(document["text"] for document in reranked_docs)
        return {
            "answer": reranked_docs[0]["text"] if reranked_docs else "",
            "retrieved_docs": reranked_docs,
            "num_hops": 1,
            "context": context,
        }


def _agent_answer_with_fallback(agent: AgenticRAG, question: str) -> dict[str, Any]:
    try:
        return agent.answer(question)
    except Exception:
        reranked_docs = _fallback_docs(agent, question)
        return {
            "answer": reranked_docs[0]["text"] if reranked_docs else "",
            "num_hops": 1,
            "retrieved_docs": reranked_docs,
            "per_hop_docs": [{"query": question, "docs": reranked_docs}],
            "sub_queries": [question],
        }


def run(num_samples: int = 50) -> dict[str, dict[str, float]]:
    samples = load_hotpotqa(split="validation", max_samples=num_samples)
    corpus = build_corpus(samples)
    ground_truths = build_ground_truths(samples, corpus)
    questions = [sample["question"] for sample in samples]

    print(f"Running baseline RAG on {num_samples} samples...")
    baseline = StaticRAG(corpus, top_k=5)
    baseline_results = [_baseline_answer_with_fallback(baseline, question) for question in questions]
    baseline_metrics = evaluate_batch(baseline_results, ground_truths)

    print(f"Running Agentic RAG on {num_samples} samples...")
    agent = AgenticRAG(corpus, top_k=5, max_hops=4, min_hops=2)
    agent_results = [_agent_answer_with_fallback(agent, question) for question in questions]
    agent_metrics = evaluate_batch(agent_results, ground_truths)

    report = {
        "baseline": baseline_metrics,
        "agentic": agent_metrics,
    }

    results_path = Path("results/experiment_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")

    print("\n=== RESULTS ===")
    print(f"{'Metric':<15} {'Baseline':>10} {'Agentic':>10}")
    print("-" * 37)
    for metric in ["EM", "F1", "MRR", "NDCG@10", "avg_hops"]:
        print(f"{metric:<15} {baseline_metrics[metric]:>10.4f} {agent_metrics[metric]:>10.4f}")

    return report


if __name__ == "__main__":
    run(num_samples=50)

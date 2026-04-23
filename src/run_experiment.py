from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent import AgenticRAG
from src.baseline_rag import StaticRAG
from src.data_loader import build_corpus, load_hotpotqa
from src.evaluator import evaluate_batch, exact_match_score, f1_score

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


def _supporting_titles(sample: dict[str, Any]) -> set[str]:
    titles = sample.get("supporting_facts", {}).get("title", []) or []
    return {str(t).strip() for t in titles if str(t).strip()}


def _gold_needs_multihop(sample: dict[str, Any]) -> bool:
    """
    Heuristic: if the gold supporting facts touch 2+ distinct titles,
    the question requires multi-document evidence (multi-hop).
    """
    return len(_supporting_titles(sample)) >= 2


def _metrics_for_answer(pred: str, gold: str) -> dict[str, float]:
    return {
        "em": float(exact_match_score(pred or "", gold or "")),
        "f1": float(f1_score(pred or "", gold or "")),
    }


def _doc_titles_from_retrieved(retrieved_docs: list[dict[str, Any]]) -> list[str]:
    titles: list[str] = []
    for doc in retrieved_docs or []:
        if not isinstance(doc, dict):
            continue
        title = doc.get("title")
        if not title and "text" in doc and isinstance(doc["text"], str) and "." in doc["text"]:
            title = doc["text"].split(".", 1)[0].strip()
        if title:
            titles.append(str(title))
    return titles


def _summarize_trace(result: dict[str, Any]) -> dict[str, Any]:
    per_hop = []
    for hop in result.get("per_hop_docs", []) or []:
        query = hop.get("query", "")
        docs = hop.get("docs", []) or []
        per_hop.append(
            {
                "query": query,
                "top_titles": _doc_titles_from_retrieved(docs)[:5],
            }
        )
    return {
        "num_hops": result.get("num_hops", 1),
        "sub_queries": list(result.get("sub_queries", []) or []),
        "per_hop": per_hop,
        "hop_decisions": list(result.get("hop_decisions", []) or []),
    }


def _group_metrics(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, float]:
    subset = [r for r in rows if r.get(key) == value]
    if not subset:
        return {"count": 0, "EM": 0.0, "F1": 0.0, "avg_hops": 0.0}
    em = sum(r["agent_metrics"]["em"] for r in subset) / len(subset)
    f1 = sum(r["agent_metrics"]["f1"] for r in subset) / len(subset)
    avg_hops = sum(float(r.get("agent_trace", {}).get("num_hops", 1)) for r in subset) / len(subset)
    return {"count": len(subset), "EM": em, "F1": f1, "avg_hops": avg_hops}


def _write_multihop_report(rows: list[dict[str, Any]], out_path: Path) -> None:
    single = _group_metrics(rows, "gold_multihop", False)
    multi = _group_metrics(rows, "gold_multihop", True)

    def pick_examples(multihop: bool, n: int = 3) -> list[dict[str, Any]]:
        subset = [r for r in rows if r.get("gold_multihop") == multihop]
        # prioritize cases where agent F1 is high (show best traces)
        subset.sort(key=lambda r: (r["agent_metrics"]["f1"], r["agent_metrics"]["em"]), reverse=True)
        return subset[:n]

    def fmt_example(r: dict[str, Any]) -> str:
        trace = r.get("agent_trace", {}) or {}
        lines = []
        lines.append(f"- **id**: `{r.get('id','')}`")
        lines.append(f"  - **question**: {r.get('question','')}")
        lines.append(f"  - **gold_answer**: {r.get('gold_answer','')}")
        lines.append(f"  - **agent_answer**: {r.get('agent_answer','')}")
        lines.append(f"  - **gold_supporting_titles**: {', '.join(r.get('gold_supporting_titles', []) or [])}")
        lines.append(f"  - **agent_hops**: {trace.get('num_hops', 1)}")
        for hop in trace.get("per_hop", [])[:4]:
            q = hop.get("query", "")
            tops = hop.get("top_titles", []) or []
            lines.append(f"    - **hop_query**: {q}")
            lines.append(f"      - **top_titles**: {', '.join(tops)}")
        return "\n".join(lines)

    examples_single = pick_examples(False)
    examples_multi = pick_examples(True)

    md = []
    md.append("# Multi-hop vs Single-hop Comparison\n")
    md.append("This report splits validation questions by whether the **gold supporting facts** span multiple Wikipedia titles.\n")
    md.append("## Subset metrics (agent only)\n")
    md.append("| Subset | Count | EM | F1 | Avg hops used |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| gold single-hop (1 title) | {single['count']} | {single['EM']:.3f} | {single['F1']:.3f} | {single['avg_hops']:.2f} |")
    md.append(f"| gold multi-hop (2+ titles) | {multi['count']} | {multi['EM']:.3f} | {multi['F1']:.3f} | {multi['avg_hops']:.2f} |")
    md.append("")
    md.append("## Example traces (gold single-hop)\n")
    md.append("\n".join(fmt_example(r) for r in examples_single) or "_No examples._")
    md.append("")
    md.append("## Example traces (gold multi-hop)\n")
    md.append("\n".join(fmt_example(r) for r in examples_multi) or "_No examples._")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


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

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregate metrics
    (results_dir / "experiment_results.json").write_text(
        f"{json.dumps(report, indent=2)}\n", encoding="utf-8"
    )

    # Save per-example traces for white-box analysis
    rows: list[dict[str, Any]] = []
    for sample, gt, baseline_out, agent_out in zip(
        samples, ground_truths, baseline_results, agent_results, strict=True
    ):
        gold_titles = sorted(_supporting_titles(sample))
        row = {
            "id": sample.get("id", ""),
            "question": sample.get("question", ""),
            "type": sample.get("type", ""),
            "level": sample.get("level", ""),
            "gold_answer": gt.get("answer", ""),
            "gold_supporting_titles": gold_titles,
            "gold_multihop": _gold_needs_multihop(sample),
            "baseline_answer": baseline_out.get("answer", ""),
            "agent_answer": agent_out.get("answer", ""),
            "baseline_metrics": _metrics_for_answer(baseline_out.get("answer", ""), gt.get("answer", "")),
            "agent_metrics": _metrics_for_answer(agent_out.get("answer", ""), gt.get("answer", "")),
            "agent_trace": _summarize_trace(agent_out),
        }
        rows.append(row)

    (results_dir / "per_example_traces.json").write_text(
        f"{json.dumps({'created_at': datetime.now(timezone.utc).isoformat(), 'rows': rows}, indent=2)}\n",
        encoding="utf-8",
    )

    _write_multihop_report(rows, results_dir / "multihop_vs_singlehop.md")

    print("\n=== RESULTS ===")
    print(f"{'Metric':<15} {'Baseline':>10} {'Agentic':>10}")
    print("-" * 37)
    for metric in ["EM", "F1", "MRR", "NDCG@10", "avg_hops"]:
        print(f"{metric:<15} {baseline_metrics[metric]:>10.4f} {agent_metrics[metric]:>10.4f}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline vs agentic RAG on HotpotQA.")
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    run(num_samples=args.num_samples)

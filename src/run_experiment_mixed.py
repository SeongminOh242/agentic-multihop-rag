from __future__ import annotations

"""
Mixed single-hop vs multi-hop experiment.

HotpotQA (distractor) serves as the multi-hop slice.
SQuAD serves as the single-hop slice (questions answerable from one passage).

Outputs:
  - results/per_example_traces_mixed.json
  - results/multihop_vs_singlehop_mixed.md
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent import AgenticRAG
from src.baseline_rag import StaticRAG
from src.data_loader import build_corpus, build_corpus_squad, load_hotpotqa, load_squad
from src.evaluator import exact_match_score, f1_score
from src.run_experiment import _agent_answer_with_fallback, _baseline_answer_with_fallback, build_ground_truths


def _metrics_for_answer(pred: str, gold: str) -> dict[str, float]:
    return {"em": float(exact_match_score(pred or "", gold or "")), "f1": float(f1_score(pred or "", gold or ""))}


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
        per_hop.append({"query": query, "top_titles": _doc_titles_from_retrieved(docs)[:5]})
    return {
        "num_hops": result.get("num_hops", 1),
        "sub_queries": list(result.get("sub_queries", []) or []),
        "per_hop": per_hop,
        "hop_decisions": list(result.get("hop_decisions", []) or []),
    }


def _group(rows: list[dict[str, Any]], multihop: bool) -> dict[str, float]:
    subset = [r for r in rows if bool(r.get("gold_multihop")) == multihop]
    if not subset:
        return {"count": 0, "EM": 0.0, "F1": 0.0, "avg_hops": 0.0}
    em = sum(r["agent_metrics"]["em"] for r in subset) / len(subset)
    f1 = sum(r["agent_metrics"]["f1"] for r in subset) / len(subset)
    avg_hops = sum(float(r["agent_trace"].get("num_hops", 1)) for r in subset) / len(subset)
    return {"count": len(subset), "EM": em, "F1": f1, "avg_hops": avg_hops}


def _write_report(rows: list[dict[str, Any]], out_path: Path) -> None:
    single = _group(rows, multihop=False)
    multi = _group(rows, multihop=True)

    md: list[str] = []
    md.append("# Multi-hop vs Single-hop Comparison (Mixed)\n")
    md.append(
        "This report compares **single-hop** questions (SQuAD) against **multi-hop** questions (HotpotQA distractor).\n"
    )
    md.append("## Subset metrics (agent only)\n")
    md.append("| Subset | Count | EM | F1 | Avg hops used |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(
        f"| gold single-hop (SQuAD) | {single['count']} | {single['EM']:.3f} | {single['F1']:.3f} | {single['avg_hops']:.2f} |"
    )
    md.append(
        f"| gold multi-hop (HotpotQA) | {multi['count']} | {multi['EM']:.3f} | {multi['F1']:.3f} | {multi['avg_hops']:.2f} |"
    )
    md.append("")
    out_path.write_text("\n".join(md), encoding="utf-8")


def _run_squad(num_samples: int) -> list[dict[str, Any]]:
    samples = load_squad(split="validation", max_samples=num_samples)
    corpus = build_corpus_squad(samples)
    text_to_doc_id = {text: i for i, text in enumerate(corpus)}

    questions = [s["question"] for s in samples]
    gold_answers = [s["answer"] for s in samples]

    baseline = StaticRAG(corpus, top_k=5)
    agent = AgenticRAG(corpus, top_k=5, max_hops=4, min_hops=1)

    baseline_results = [_baseline_answer_with_fallback(baseline, q) for q in questions]
    agent_results = [_agent_answer_with_fallback(agent, q) for q in questions]

    rows: list[dict[str, Any]] = []
    for sample, q, gold, base_out, agent_out in zip(samples, questions, gold_answers, baseline_results, agent_results, strict=True):
        doc_id = text_to_doc_id.get(sample["context_text"])
        rows.append(
            {
                "dataset": "squad",
                "id": sample.get("id", ""),
                "question": q,
                "type": "singlehop",
                "level": "",
                "gold_answer": gold,
                "gold_supporting_titles": [],
                "gold_multihop": False,
                "baseline_answer": base_out.get("answer", ""),
                "agent_answer": agent_out.get("answer", ""),
                "baseline_metrics": _metrics_for_answer(base_out.get("answer", ""), gold),
                "agent_metrics": _metrics_for_answer(agent_out.get("answer", ""), gold),
                "supporting_doc_ids": [doc_id] if doc_id is not None else [],
                "agent_trace": _summarize_trace(agent_out),
            }
        )
    return rows


def _run_hotpot(num_samples: int) -> list[dict[str, Any]]:
    samples = load_hotpotqa(split="validation", max_samples=num_samples)
    corpus = build_corpus(samples)
    gts = build_ground_truths(samples, corpus)
    questions = [s["question"] for s in samples]

    baseline = StaticRAG(corpus, top_k=5)
    agent = AgenticRAG(corpus, top_k=5, max_hops=4, min_hops=2)

    baseline_results = [_baseline_answer_with_fallback(baseline, q) for q in questions]
    agent_results = [_agent_answer_with_fallback(agent, q) for q in questions]

    def supporting_titles(sample: dict[str, Any]) -> list[str]:
        titles = sample.get("supporting_facts", {}).get("title", []) or []
        return sorted({str(t).strip() for t in titles if str(t).strip()})

    rows: list[dict[str, Any]] = []
    for sample, gt, q, base_out, agent_out in zip(samples, gts, questions, baseline_results, agent_results, strict=True):
        gold = gt.get("answer", "")
        gold_titles = supporting_titles(sample)
        rows.append(
            {
                "dataset": "hotpotqa",
                "id": sample.get("id", ""),
                "question": q,
                "type": sample.get("type", ""),
                "level": sample.get("level", ""),
                "gold_answer": gold,
                "gold_supporting_titles": gold_titles,
                "gold_multihop": True,
                "baseline_answer": base_out.get("answer", ""),
                "agent_answer": agent_out.get("answer", ""),
                "baseline_metrics": _metrics_for_answer(base_out.get("answer", ""), gold),
                "agent_metrics": _metrics_for_answer(agent_out.get("answer", ""), gold),
                "supporting_doc_ids": sorted(list(gt.get("supporting_doc_ids", set()) or [])),
                "agent_trace": _summarize_trace(agent_out),
            }
        )
    return rows


def run(num_hotpot: int = 50, num_squad: int = 50) -> dict[str, Any]:
    load_dotenv()
    rows = _run_squad(num_squad) + _run_hotpot(num_hotpot)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "per_example_traces_mixed.json").write_text(
        json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(rows, results_dir / "multihop_vs_singlehop_mixed.md")
    return {"count": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a mixed single-hop vs multi-hop comparison.")
    parser.add_argument("--num-hotpot", type=int, default=50)
    parser.add_argument("--num-squad", type=int, default=50)
    args = parser.parse_args()
    run(num_hotpot=args.num_hotpot, num_squad=args.num_squad)


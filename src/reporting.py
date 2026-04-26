from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_detailed_results(path: str | Path = "results/experiment_details.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_support_doc_breakdown(
    details: dict[str, Any],
    system_name: str = "agentic",
) -> None:
    system = details[system_name]
    breakdown = system["breakdowns"]["support_docs_needed"]

    print(f"\n=== {system_name.upper()} SUPPORT-DOC BREAKDOWN ===")
    print(f"{'Group':<24} {'Count':>7} {'EM':>8} {'F1':>8} {'MRR':>8} {'NDCG@10':>10} {'Hops':>8}")
    print("-" * 75)
    for group_name, metrics in breakdown.items():
        print(
            f"{group_name:<24}"
            f" {int(metrics['count']):>7}"
            f" {metrics['EM']:>8.4f}"
            f" {metrics['F1']:>8.4f}"
            f" {metrics['MRR']:>8.4f}"
            f" {metrics['NDCG@10']:>10.4f}"
            f" {metrics['avg_hops']:>8.2f}"
        )


def print_support_doc_comparison(details: dict[str, Any]) -> None:
    """Compare systems on questions needing one vs multiple supporting docs."""

    systems = [name for name in ("baseline", "agentic", "frontier_agentic") if name in details]
    groups = ("single_support_doc", "multiple_support_docs")

    print("\n=== SUPPORT-DOC NEED COMPARISON ===")
    print(
        f"{'Group':<24} {'System':<18} {'Count':>7} {'EM':>8} "
        f"{'F1':>8} {'MRR':>8} {'NDCG@10':>10} {'Hops':>8}"
    )
    print("-" * 92)

    for group_name in groups:
        for system_name in systems:
            breakdown = details[system_name]["breakdowns"]["support_docs_needed"]
            metrics = breakdown.get(group_name)
            if metrics is None:
                continue
            print(
                f"{group_name:<24}"
                f" {system_name:<18}"
                f" {int(metrics['count']):>7}"
                f" {metrics['EM']:>8.4f}"
                f" {metrics['F1']:>8.4f}"
                f" {metrics['MRR']:>8.4f}"
                f" {metrics['NDCG@10']:>10.4f}"
                f" {metrics['avg_hops']:>8.2f}"
            )


def print_retrieval_hop_need_comparison(details: dict[str, Any]) -> None:
    """Compare examples where first-hop retrieval was sufficient vs not."""

    systems = [name for name in ("baseline", "agentic", "frontier_agentic") if name in details]
    groups = ("single_hop_sufficient", "needs_followup_hops")

    print("\n=== RETRIEVAL HOP-NEED COMPARISON ===")
    print(
        f"{'Group':<24} {'System':<18} {'Count':>7} {'EM':>8} "
        f"{'F1':>8} {'MRR':>8} {'NDCG@10':>10} {'Hops':>8}"
    )
    print("-" * 92)

    for group_name in groups:
        for system_name in systems:
            breakdown = details[system_name]["breakdowns"]["retrieval_hop_need"]
            metrics = breakdown.get(group_name)
            if metrics is None:
                continue
            print(
                f"{group_name:<24}"
                f" {system_name:<18}"
                f" {int(metrics['count']):>7}"
                f" {metrics['EM']:>8.4f}"
                f" {metrics['F1']:>8.4f}"
                f" {metrics['MRR']:>8.4f}"
                f" {metrics['NDCG@10']:>10.4f}"
                f" {metrics['avg_hops']:>8.2f}"
            )


def print_case_studies(
    details: dict[str, Any],
    system_name: str = "agentic",
    group_name: str = "successful_multi_doc",
) -> None:
    print(f"\n=== {system_name.upper()} {group_name.upper()} ===")
    studies = details[system_name]["case_studies"].get(group_name, [])
    if not studies:
        print("No case studies available.")
        return

    for index, study in enumerate(studies, start=1):
        print(f"\nCase {index}: {study['question']}")
        print(f"Gold: {study['gold_answer']}")
        print(f"Pred: {study['predicted_answer']}")
        print(f"Hops: {study['num_hops']}")
        print(f"Supporting titles: {', '.join(study['supporting_titles'])}")
        print(f"Sub-queries: {study['sub_queries']}")
        for trace in study.get("hop_traces", []):
            decision = trace.get("decision", {})
            print(
                f"  Decision hop {trace.get('hop')}: "
                f"sufficient={trace.get('sufficient')} "
                f"next={trace.get('next_query')!r} "
                f"final={decision.get('final_answer')!r}"
            )
        for hop in study["per_hop"]:
            doc_titles = [doc["title"] for doc in hop["docs"]]
            print(f"  Hop query: {hop['query']}")
            print(f"  Docs: {doc_titles}")


def print_intermediate_trace(
    details: dict[str, Any],
    system_name: str = "agentic",
    trace_index: int = 0,
) -> None:
    """Print the white-box agent loop for one sample."""

    trace = details[system_name]["traces"][trace_index]
    print(f"\n=== {system_name.upper()} TRACE {trace_index} ===")
    print("Question:", trace["question"])
    print("Gold:", trace["gold_answer"])
    print("Pred:", trace["predicted_answer"])
    print("Requires multiple docs:", trace["requires_multiple_documents"])
    print("Num hops:", trace["num_hops"])
    print("Sub-queries:", trace["sub_queries"])

    for hop, decision_trace in zip(trace["per_hop"], trace.get("hop_traces", [])):
        print(f"\nHop {decision_trace.get('hop')} query: {hop['query']}")
        print("Top docs:", [doc["title"] for doc in hop["docs"]])
        print("Decision:", decision_trace.get("decision"))
        print("Sufficient:", decision_trace.get("sufficient"))
        if decision_trace.get("parse_error"):
            print("Parse error:", decision_trace["parse_error"])


def print_full_trace_with_text(
    trace: dict[str, Any],
    max_text_chars: int = 300,
) -> None:
    """White-box dump of one agent trace including retrieved text excerpts at each hop."""
    print(f"\nQuestion    : {trace['question']}")
    print(f"Gold        : {trace['gold_answer']}")
    print(f"Pred        : {trace['predicted_answer']}")
    print(f"Correct     : {'yes' if trace['exact_match'] else 'no'}")
    print(f"Hops        : {trace['num_hops']}")
    print(f"Multi-doc   : {trace['requires_multiple_documents']}")
    print(f"Sub-queries : {trace['sub_queries']}")

    hop_traces = trace.get("hop_traces", [])
    for hop_idx, hop in enumerate(trace.get("per_hop", []), start=1):
        decision = hop_traces[hop_idx - 1] if hop_idx - 1 < len(hop_traces) else {}
        print(f"\n  --- Hop {hop_idx}: {hop['query']!r} ---")
        if decision:
            sufficient = decision.get("sufficient", "?")
            next_q = decision.get("next_query", "")
            print(f"  Decision  : sufficient={sufficient}, next_query={next_q!r}")
        for doc in hop.get("docs", []):
            text = (doc.get("text_preview") or "")[:max_text_chars]
            print(f"  [{doc['title']}] {text}...")


def print_frontier_ceiling_summary(details: dict[str, Any]) -> None:
    """Compare baseline, agentic, and frontier aggregate metrics in one table."""
    system_labels = {
        "baseline": "Baseline RAG (LLaMA-3.1-8B)",
        "agentic": "Agentic RAG (LLaMA-3.1-8B)",
        "frontier_agentic": "Frontier Agentic (Gemini-2.5-Flash)",
    }

    print("\n=== FRONTIER MODEL CEILING ===")
    print(
        f"{'System':<38} {'EM':>8} {'F1':>8} {'MRR':>8} {'NDCG@10':>10} {'Avg Hops':>10}"
    )
    print("-" * 86)

    rows: dict[str, dict[str, float]] = {}
    for key, label in system_labels.items():
        if key not in details:
            continue
        agg = details[key]["aggregate"]
        rows[key] = agg
        print(
            f"{label:<38}"
            f" {agg['EM']:>8.4f}"
            f" {agg['F1']:>8.4f}"
            f" {agg['MRR']:>8.4f}"
            f" {agg['NDCG@10']:>10.4f}"
            f" {agg['avg_hops']:>10.2f}"
        )

    if "frontier_agentic" in rows and "agentic" in rows:
        print()
        for metric in ["EM", "F1", "MRR", "NDCG@10"]:
            delta = rows["frontier_agentic"][metric] - rows["agentic"][metric]
            sign = "+" if delta >= 0 else ""
            print(f"  Frontier vs Agentic  {metric:<8}: {sign}{delta:.4f}")
        print()
        if rows["frontier_agentic"]["EM"] <= rows["agentic"]["EM"]:
            print(
                "  Observation: Frontier (Gemini-2.5-Flash) does NOT improve EM over the smaller\n"
                "  open-source agentic model.  The retrieval loop architecture — not raw model\n"
                "  capacity — is the binding constraint on exact-match accuracy."
            )
        else:
            delta_em = rows["frontier_agentic"]["EM"] - rows["agentic"]["EM"]
            print(f"  Observation: Frontier improves EM by {delta_em:+.4f} over agentic.")

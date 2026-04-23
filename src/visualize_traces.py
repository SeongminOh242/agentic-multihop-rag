"""
Generate figures from per-example trace JSON (e.g. per_example_traces_mixed.json).

Writes PNGs under ``visuals/`` by default:
  - subset_agent_metrics.png   — EM / F1 / avg hops by gold single-hop vs multi-hop
  - hop_count_distribution.png — histogram of agent num_hops per subset
  - f1_vs_num_hops.png         — scatter of agent F1 vs num_hops
  - example_hop_pipeline.png   — one annotated multi-hop example (if available)

Run:
  python -m src.visualize_traces --input results/per_example_traces_mixed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON object with 'rows' list, got keys: {list(data.keys())}")
    return rows


def subset_by_multihop(rows: list[dict[str, Any]], multihop: bool) -> list[dict[str, Any]]:
    return [r for r in rows if bool(r.get("gold_multihop")) == multihop]


def aggregate_subset_metrics(subset: list[dict[str, Any]]) -> dict[str, float]:
    if not subset:
        return {"count": 0.0, "em": 0.0, "f1": 0.0, "avg_hops": 0.0}
    n = len(subset)
    em = sum(float(r.get("agent_metrics", {}).get("em", 0.0)) for r in subset) / n
    f1 = sum(float(r.get("agent_metrics", {}).get("f1", 0.0)) for r in subset) / n
    avg_hops = sum(float((r.get("agent_trace") or {}).get("num_hops", 1)) for r in subset) / n
    return {"count": float(n), "em": em, "f1": f1, "avg_hops": avg_hops}


def plot_subset_agent_metrics(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    single_label: str = "Gold single-hop",
    multi_label: str = "Gold multi-hop",
) -> None:
    single = subset_by_multihop(rows, False)
    multi = subset_by_multihop(rows, True)
    s_m = aggregate_subset_metrics(single)
    m_m = aggregate_subset_metrics(multi)

    labels = [single_label, multi_label]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width, [s_m["em"], m_m["em"]], width, label="EM", color="#4C72B0")
    ax1.bar(x, [s_m["f1"], m_m["f1"]], width, label="F1", color="#55A868")
    ax1.set_ylabel("Score (0–1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.bar(
        x + width,
        [s_m["avg_hops"], m_m["avg_hops"]],
        width,
        label="Avg hops",
        color="#C44E52",
        alpha=0.85,
    )
    ax2.set_ylabel("Avg hops used")
    max_h = max(s_m["avg_hops"], m_m["avg_hops"], 1.0)
    ax2.set_ylim(0, max(2.5, max_h * 1.25))

    counts = f"n={int(s_m['count'])}, n={int(m_m['count'])}"
    ax1.set_title(f"Agent metrics by subset ({counts})")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hop_count_distribution(rows: list[dict[str, Any]], out_path: Path) -> None:
    single = subset_by_multihop(rows, False)
    multi = subset_by_multihop(rows, True)

    def hops_list(subset: list[dict[str, Any]]) -> list[int]:
        out: list[int] = []
        for r in subset:
            n = int((r.get("agent_trace") or {}).get("num_hops", 1))
            out.append(max(1, n))
        return out

    hs = hops_list(single)
    hm = hops_list(multi)
    all_hops = sorted(set(hs + hm) or {1})
    bins = [b - 0.5 for b in all_hops] + [all_hops[-1] + 0.5]

    fig, ax = plt.subplots(figsize=(8, 5))
    if hs:
        ax.hist(hs, bins=bins, alpha=0.65, label=f"Single-hop (n={len(hs)})", color="#4C72B0", density=False)
    if hm:
        ax.hist(hm, bins=bins, alpha=0.65, label=f"Multi-hop (n={len(hm)})", color="#C44E52", density=False)
    ax.set_xlabel("Agent num_hops")
    ax.set_ylabel("Count")
    ax.set_xticks(all_hops)
    ax.legend()
    ax.set_title("Distribution of retrieval hops by gold subset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_f1_vs_num_hops(rows: list[dict[str, Any]], out_path: Path) -> None:
    xs: list[int] = []
    ys: list[float] = []
    colors: list[float] = []
    for r in rows:
        nh = int((r.get("agent_trace") or {}).get("num_hops", 1))
        f1 = float(r.get("agent_metrics", {}).get("f1", 0.0))
        xs.append(max(1, nh))
        ys.append(f1)
        colors.append(1.0 if r.get("gold_multihop") else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(xs, ys, c=colors, cmap="coolwarm", alpha=0.7, edgecolors="k", linewidths=0.3, s=40)
    ax.set_xlabel("num_hops")
    ax.set_ylabel("Agent F1")
    ax.set_xticks(sorted(set(xs)))
    cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["single-hop", "multi-hop"])
    ax.set_title("Agent F1 vs hops (color = gold subset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _truncate(s: str, max_len: int) -> str:
    s = (s or "").replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def pick_multihop_example_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        r
        for r in rows
        if r.get("gold_multihop")
        and int((r.get("agent_trace") or {}).get("num_hops", 1)) >= 2
    ]
    if not candidates:
        candidates = [r for r in rows if r.get("gold_multihop")]
    return candidates[0] if candidates else None


def plot_example_hop_pipeline(row: dict[str, Any], out_path: Path) -> None:
    trace = row.get("agent_trace") or {}
    per_hop = trace.get("per_hop") or []
    decisions = trace.get("hop_decisions") or []
    q = row.get("question", "")
    gold = row.get("gold_answer", "")
    pred = row.get("agent_answer", "")

    fig, ax = plt.subplots(figsize=(10, max(4, 2.2 * max(len(per_hop), 1))))
    ax.axis("off")
    y = 0.98
    line_h = 0.085
    ax.text(0.02, y, "Question: " + _truncate(str(q), 200), fontsize=10, transform=ax.transAxes, va="top")
    y -= line_h
    ax.text(0.02, y, "Gold: " + _truncate(str(gold), 120), fontsize=9, color="#333", transform=ax.transAxes, va="top")
    y -= line_h * 0.9
    ax.text(0.02, y, "Agent: " + _truncate(str(pred), 120), fontsize=9, color="#333", transform=ax.transAxes, va="top")
    y -= line_h * 1.2

    for i, hop in enumerate(per_hop):
        hop_n = i + 1
        ax.text(
            0.02,
            y,
            f"Hop {hop_n} — query: {_truncate(str(hop.get('query', '')), 160)}",
            fontsize=9,
            weight="bold",
            transform=ax.transAxes,
            va="top",
        )
        y -= line_h * 0.85
        titles = hop.get("top_titles") or []
        top_s = "; ".join(_truncate(str(t), 60) for t in titles[:4])
        ax.text(0.04, y, "Top titles: " + _truncate(top_s, 280), fontsize=8, transform=ax.transAxes, va="top")
        y -= line_h * 1.1
        dec = next((d for d in decisions if int(d.get("hop", 0)) == hop_n), None)
        if dec:
            raw = str(dec.get("raw_llm_response", ""))
            ax.text(0.04, y, "Decision: " + _truncate(raw, 320), fontsize=7, family="monospace", transform=ax.transAxes, va="top")
            y -= line_h * 1.3
        y -= line_h * 0.3

    ax.set_title("Example agent trace (one multi-hop row)", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_visuals(
    input_json: Path,
    out_dir: Path,
    *,
    skip_example_if_none: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_trace_rows(input_json)
    written: list[Path] = []

    p1 = out_dir / "subset_agent_metrics.png"
    plot_subset_agent_metrics(rows, p1)
    written.append(p1)

    p2 = out_dir / "hop_count_distribution.png"
    plot_hop_count_distribution(rows, p2)
    written.append(p2)

    p3 = out_dir / "f1_vs_num_hops.png"
    plot_f1_vs_num_hops(rows, p3)
    written.append(p3)

    ex = pick_multihop_example_row(rows)
    p4 = out_dir / "example_hop_pipeline.png"
    if ex is not None:
        plot_example_hop_pipeline(ex, p4)
        written.append(p4)
    elif not skip_example_if_none:
        # Still write a minimal note figure if no multi-hop rows
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No gold multi-hop rows; skipped example_hop_pipeline.png", ha="center", va="center")
        ax.axis("off")
        fig.savefig(p4, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p4)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visuals from per-example trace JSON.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("results/per_example_traces_mixed.json"),
        help="Path to per_example_traces*.json",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("visuals"),
        help="Output directory for PNGs",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    paths = generate_visuals(args.input, args.out)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()

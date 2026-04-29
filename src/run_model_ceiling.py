from __future__ import annotations

"""
Frontier model ceiling experiment.

Goal: keep retrieval/agent logic fixed, vary only the LLM model name,
and compare answer metrics. This helps demonstrate the "ceiling" effect:
stronger frontier LLMs often lift EM/F1 even when retrieval is unchanged.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent import AgenticRAG
from src.baseline_rag import StaticRAG
from src.data_loader import build_corpus, load_hotpotqa
from src.evaluator import evaluate_batch
from src.run_experiment import build_ground_truths, _agent_answer_with_fallback, _baseline_answer_with_fallback


def run(num_samples: int, models: list[str]) -> dict[str, Any]:
    load_dotenv()

    samples = load_hotpotqa(split="validation", max_samples=num_samples)
    corpus = build_corpus(samples)
    ground_truths = build_ground_truths(samples, corpus)
    questions = [sample["question"] for sample in samples]

    out: dict[str, Any] = {"num_samples": num_samples, "models": {}}

    for model_name in models:
        baseline = StaticRAG(corpus, top_k=5, model_name=model_name)
        agent = AgenticRAG(corpus, top_k=5, max_hops=4, min_hops=2, model_name=model_name)

        baseline_results = [_baseline_answer_with_fallback(baseline, q) for q in questions]
        agent_results = [_agent_answer_with_fallback(agent, q) for q in questions]

        out["models"][model_name] = {
            "baseline": evaluate_batch(baseline_results, ground_truths),
            "agentic": evaluate_batch(agent_results, ground_truths),
        }

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/model_ceiling.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare models (frontier ceiling) for RAG pipelines.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gpt-4o"],
        help="OpenAI model names (or OpenAI-compatible names).",
    )
    args = parser.parse_args()

    run(num_samples=args.num_samples, models=list(args.models))


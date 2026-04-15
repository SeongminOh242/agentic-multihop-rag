from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any

from .types import HotpotExample


def normalize_answer(s: str) -> str:
    """Standard HotpotQA normalization for evaluation."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if the normalized prediction exactly matches ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score over normalized strings."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        # If either is empty, F1 is 1 if both are empty, else 0
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# Aliases to match the original 04-08 spec verbatim
_normalize = normalize_answer
exact_match = exact_match_score
f1_score_tokens = f1_score


class Evaluator:
    """Evaluates agent execution traces against HotpotQA ground truth examples."""

    def evaluate(
        self, agent_response: dict[str, Any], ground_truth: HotpotExample
    ) -> dict[str, float]:
        """
        Evaluate single hop or multihop RAG agent output.
        Safely handles malformed agent payloads to prevent crashes.
        """
        pred_answer = agent_response.get("answer") or ""

        em = float(exact_match_score(pred_answer, ground_truth.answer))
        f1 = f1_score(pred_answer, ground_truth.answer)

        # Retrieval evaluation
        retrieved_docs = agent_response.get("retrieved_docs") or []
        retrieved_titles = set()

        for doc in retrieved_docs:
            if isinstance(doc, dict):
                title = doc.get("title")
                # Fallback heuristics for basic text-only dictionaries
                if not title and "text" in doc:
                    text_val = doc["text"]
                    # If it's formatted like baseline: "Title. sentence..."
                    if "." in text_val:
                        inferred_title = text_val.split(".", 1)[0].strip()
                        retrieved_titles.add(inferred_title)
                elif title:
                    retrieved_titles.add(title)
            elif hasattr(doc, "title"):
                retrieved_titles.add(getattr(doc, "title"))

        expected_titles = {fact.title for fact in ground_truth.supporting_facts}

        if not expected_titles:
            retrieval_recall = 1.0  # Safe fallback if example has 0 expected facts
        else:
            intersection = expected_titles.intersection(retrieved_titles)
            retrieval_recall = float(len(intersection)) / len(expected_titles)

        return {
            "exact_match": em,
            "f1": f1,
            "retrieval_recall": retrieval_recall,
        }


def mean_reciprocal_rank(retrieved: list[dict], relevant_ids: set) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc.get("doc_id") in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[dict], relevant_ids: set, k: int = 10) -> float:
    def dcg(hits):
        return sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))

    hits = [1.0 if doc.get("doc_id") in relevant_ids else 0.0 for doc in retrieved[:k]]
    ideal = sorted(hits, reverse=True)
    actual_dcg = dcg(hits)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_batch(results: list[dict], ground_truths: list[dict]) -> dict:
    """
    results: list of {"answer": str, "retrieved_docs": [...], "num_hops": int,
                       "per_hop_docs": [{"query": str, "docs": [...]}]}
    ground_truths: list of {"answer": str, "supporting_doc_ids": set}

    Per-hop MRR/NDCG measure intermediate search quality as required by the spec.
    Final MRR/NDCG measure overall retrieval quality across all hops combined.
    """
    em_scores, f1_scores, mrr_scores, ndcg_scores, hops = [], [], [], [], []
    per_hop_mrr_scores, per_hop_ndcg_scores = [], []

    for result, gt in zip(results, ground_truths):
        ans = result.get("answer") or ""
        expected_ans = gt.get("answer") or ""
        em_scores.append(float(exact_match_score(ans, expected_ans)))
        f1_scores.append(f1_score(ans, expected_ans))
        
        relevant = gt.get("supporting_doc_ids", set())

        # Final (aggregated) retrieval metrics
        ret_docs = result.get("retrieved_docs", [])
        mrr_scores.append(mean_reciprocal_rank(ret_docs, relevant))
        ndcg_scores.append(ndcg_at_k(ret_docs, relevant, k=10))
        hops.append(result.get("num_hops", 1))

        # Per-hop retrieval metrics
        hop_mrrs, hop_ndcgs = [], []
        for hop_result in result.get("per_hop_docs", []):
            docs = hop_result.get("docs", [])
            hop_mrrs.append(mean_reciprocal_rank(docs, relevant))
            hop_ndcgs.append(ndcg_at_k(docs, relevant, k=10))
        
        if hop_mrrs:
            per_hop_mrr_scores.append(sum(hop_mrrs) / len(hop_mrrs))
            per_hop_ndcg_scores.append(sum(hop_ndcgs) / len(hop_ndcgs))

    out = {
        "EM": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "F1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "MRR": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "avg_hops": sum(hops) / len(hops) if hops else 0.0,
    }
    
    if per_hop_mrr_scores:
        out["per_hop_MRR"] = sum(per_hop_mrr_scores) / len(per_hop_mrr_scores)
        out["per_hop_NDCG@10"] = sum(per_hop_ndcg_scores) / len(per_hop_ndcg_scores)
        
    return out

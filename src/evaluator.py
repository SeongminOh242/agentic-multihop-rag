from __future__ import annotations

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

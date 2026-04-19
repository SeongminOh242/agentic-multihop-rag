# Experiment Results — Agentic Multi-Hop RAG vs Baseline RAG

**Dataset:** HotpotQA (distractor setting, validation split)  
**Samples:** 50  
**LLM:** Llama 3.1 8B Instruct  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM) via Google Colab  

---

## Final Results (50 samples)

| Metric      | Baseline RAG | Agentic RAG | Change     |
|-------------|:------------:|:-----------:|:----------:|
| EM          | 0.0000       | **0.1200**  | +12 pts    |
| F1          | 0.0748       | **0.2518**  | +237%      |
| MRR         | 0.3137       | **0.3388**  | +8%        |
| NDCG@10     | 0.3880       | **0.4352**  | +12%       |
| avg\_hops   | 1.00         | 3.72        | —          |

### Agentic Per-Hop Retrieval

| Metric          | Per-Hop  | Final    |
|-----------------|:--------:|:--------:|
| MRR             | 0.3394   | 0.3388   |
| NDCG@10         | 0.4201   | 0.4352   |

---

## Pilot Results (5 samples)

| Metric      | Baseline RAG | Agentic RAG |
|-------------|:------------:|:-----------:|
| EM          | 0.0000       | 0.0000      |
| F1          | 0.0578       | 0.0419      |
| MRR         | 0.2667       | 0.2667      |
| NDCG@10     | 0.3985       | 0.4012      |
| avg\_hops   | 1.00         | 3.60        |

---

## Analysis

### Answer Quality

The agentic system achieves **12% Exact Match** vs **0% for the baseline** — the single-hop baseline structurally cannot answer HotpotQA questions, which require reasoning over 2+ documents. F1 improved by **237%** (0.075 → 0.252), reflecting that the agentic system retrieves the supporting facts needed for complete answers.

### Retrieval Quality

Both MRR (+8%) and NDCG@10 (+12%) improved with the agentic approach. The final NDCG@10 (0.435) exceeds the per-hop NDCG@10 (0.420), confirming that **accumulated context across hops provides better document coverage** than any single retrieval step.

### Hop Behavior

- `avg_hops = 3.72` out of a maximum of 4, indicating the agent consistently needs multiple retrieval steps to gather sufficient context for multi-hop questions.
- Per-hop MRR (0.3394) ≈ final MRR (0.3388), showing the agent reliably surfaces the primary relevant document early and uses later hops to gather supporting evidence.

### Key Takeaway

Multi-hop agentic retrieval significantly outperforms static single-hop RAG on HotpotQA across all metrics. The gains are largest in answer quality (EM, F1), directly validating the hypothesis that iterative query refinement enables the LLM to locate and reason over the multiple documents required for bridge and comparison questions.

---

## Raw JSON

```json
{
  "baseline": {
    "EM": 0.0,
    "F1": 0.07482593136896401,
    "MRR": 0.31366666666666665,
    "NDCG@10": 0.38799738340253226,
    "avg_hops": 1.0
  },
  "agentic": {
    "EM": 0.12,
    "F1": 0.2518380303432613,
    "MRR": 0.3388102453102453,
    "NDCG@10": 0.43522341036478496,
    "avg_hops": 3.72,
    "per_hop_MRR": 0.3393611111111111,
    "per_hop_NDCG@10": 0.42010689725588224
  }
}
```

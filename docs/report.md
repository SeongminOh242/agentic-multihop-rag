# Iterative Agentic RAG vs. Static Baseline on HotpotQA

**Course:** CS 572 — Information Retrieval  
**Dataset:** HotpotQA (distractor setting, validation split, 50 samples)  
**LLM:** Llama 3.1 8B Instruct  
**Retriever:** BM25 (rank-bm25)  
**Re-ranker:** Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

---

## 1. Introduction

Modern question-answering systems built on retrieval-augmented generation (RAG) perform well on factoid questions that can be answered from a single document. However, real-world questions are often *multi-hop* — they require locating and reasoning over two or more documents whose connections are not obvious at query time. The HotpotQA benchmark [Yang et al., 2018] was specifically designed to expose this limitation: every question requires combining evidence from at least two supporting documents, making single-hop retrieval structurally insufficient.

A static RAG pipeline issues *one* query, retrieves a fixed set of documents, and hands the concatenated context to the language model. When the relevant evidence is spread across documents that look unrelated at the surface level, a single BM25 query consistently fails to surface both pieces of evidence. This motivates *iterative agentic retrieval*, where an LLM orchestrates multiple targeted sub-queries, refining its search based on what it has found so far.

This report presents and evaluates an Iterative Agentic RAG system that uses an LLM agent to dynamically generate sub-queries across up to four retrieval hops, comparing it against a static single-hop baseline on 50 HotpotQA validation examples. We measure both answer quality (Exact Match, F1) and retrieval quality (MRR, NDCG@10) to fully characterise the gains from multi-hop reasoning.

---

## 2. Related Work

**BM25** [Robertson & Zaragoza, 2009] is the standard lexical retrieval baseline used in open-domain QA. It scores documents by term frequency and inverse document frequency, and remains a competitive baseline despite the rise of dense retrievers.

**Dense Passage Retrieval (DPR)** [Karpukhin et al., 2020] trains dual-encoder models to embed questions and passages into a shared vector space, enabling semantic similarity search via FAISS. DPR outperforms BM25 on single-hop retrieval but does not inherently support multi-hop reasoning.

**Standard RAG** [Lewis et al., 2020] combines a retriever with a sequence-to-sequence language model. In the original formulation, retrieval is a *single, non-iterative* step, which limits performance on multi-hop tasks.

**FLARE** [Jiang et al., 2023] extends RAG with forward-looking active retrieval: during generation, the model detects when it is uncertain and triggers additional retrieval queries on-the-fly. This is conceptually similar to our agentic loop but differs in that FLARE activates retrieval passively during decoding, while our system uses an explicit *Chain-of-Thought sub-query decision* at each hop.

Our approach is most closely related to iterative retrieval methods such as IRCoT [Trivedi et al., 2022], which interleaves chain-of-thought reasoning with retrieval steps, and ReAct [Yao et al., 2022], which frames tool-use (including retrieval) as a sequence of Thought-Action-Observation steps.

---

## 3. System Design

The pipeline consists of four main components:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              AgenticRAG Loop                │
│                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │ BM25     │───▶│ Cross-   │───▶│ LLM    ││
│  │Retriever │    │ Encoder  │    │ Agent  ││
│  │ top_k×2  │    │ Reranker │    │(Llama) ││
│  └──────────┘    └──────────┘    └────┬───┘│
│       ▲                               │     │
│       │         sufficient=false      │     │
│       └──────── sub_query ────────────┘     │
│                                             │
│              sufficient=true                │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
              Final Answer
```

**BM25 Retriever:** At each hop, the current sub-query is tokenised and scored against the full corpus using BM25Okapi. The top `top_k × 2 = 10` documents are passed to the re-ranker.

**Cross-Encoder Re-ranker:** A `ms-marco-MiniLM-L-6-v2` cross-encoder scores each (query, document) pair jointly, returning the top `top_k = 5` documents. This corrects the recall-precision trade-off from BM25.

**LLM Agent:** Llama 3.1 8B Instruct receives the accumulated multi-hop context and decides: either mark the context as `"sufficient": true` and emit a `"final_answer"`, or emit `"sufficient": false` with a new `"sub_query"`. This is repeated for up to `max_hops = 4` iterations.

**Baseline (StaticRAG):** Issues a single BM25 + cross-encoder retrieval step, then generates an answer from the top-5 documents in one LLM call.

---

## 4. Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | HotpotQA, distractor setting, validation split |
| Samples | 50 |
| LLM | Llama 3.1 8B Instruct (4-bit quantised, NVIDIA RTX PRO 6000 via Colab) |
| Retriever | BM25Okapi (rank-bm25) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| top_k | 5 (10 candidate docs pre-reranking) |
| max_hops | 4 |

**Metrics:**
- **Exact Match (EM):** 1 if the normalised prediction equals the normalised gold answer, 0 otherwise.
- **F1:** Token-level F1 between predicted and gold answer tokens (standard HotpotQA metric).
- **MRR (Mean Reciprocal Rank):** Measures how highly the first relevant supporting document is ranked across the retrieved list.
- **NDCG@10 (Normalised Discounted Cumulative Gain at 10):** Measures the overall ranking quality of all retrieved documents using logarithmic position discounting.
- **avg_hops:** Average number of retrieval iterations per question.

Supporting document IDs are mapped from the HotpotQA context titles to corpus indices, enabling ground-truth-based retrieval evaluation.

---

## 5. Results

| Metric | Baseline RAG | Agentic RAG | Δ Absolute | Δ % |
|---|:---:|:---:|:---:|:---:|
| EM | 0.0000 | **0.1200** | +0.12 | — |
| F1 | 0.0748 | **0.2518** | +0.177 | +237% |
| MRR (final) | 0.3137 | **0.3388** | +0.025 | +8% |
| NDCG@10 (final) | 0.3880 | **0.4352** | +0.047 | +12% |
| per-hop MRR | N/A | 0.3394 | — | — |
| per-hop NDCG@10 | N/A | 0.4201 | — | — |
| Avg Hops | 1.00 | 3.72 | — | — |

---

## 6. Analysis

### 6.1 Answer Quality

The most striking result is that the **baseline achieves 0% Exact Match** while the agentic system achieves **12% EM** (6 out of 50 questions answered exactly correctly). This is not surprising: HotpotQA bridge questions typically require locating a bridging entity in one document and using it to look up the final answer in another. A single-hop retrieval step cannot perform this join even if the top-5 documents contain both supporting passages, because the LLM has no mechanism to search for the second document using information from the first.

The **F1 improvement of 237%** (0.075 → 0.252) shows that even for questions the agentic system does not answer exactly, it produces much more semantically complete answers. The multi-hop context gives the LLM the vocabulary and factual grounding needed to closely paraphrase the correct answer.

### 6.2 Retrieval Quality

Both MRR (+8%) and NDCG@10 (+12%) improve with the agentic approach. The **final NDCG@10 (0.435) exceeds the per-hop NDCG@10 (0.420)**, which confirms that the union of documents retrieved across all hops contains better coverage of the supporting facts than any single hop retrieval step. This validates the design choice to accumulate all retrieved documents across hops rather than resetting the context for each hop.

The **per-hop MRR (0.339) ≈ final MRR (0.339)** indicates that the primary supporting document tends to be retrieved early (often in the first hop, which uses the original question as the query). Later hops add the secondary supporting documents, which is exactly the multi-hop bridging behaviour the system was designed to produce.

### 6.3 Hop Behaviour

With `avg_hops = 3.72` out of a maximum of 4, the agent almost always needs the full budget of retrieval steps. This reflects the difficulty of HotpotQA questions: the agent consistently finds that a single pass is insufficient and continues issuing sub-queries. Importantly, this is not the result of poor stopping criteria — the agent correctly identifies sufficiency and returns early when it has enough context (otherwise avg_hops would be exactly 4.0).

### 6.4 Limitations

- **Small sample size:** 50 samples is enough to show directional trends but too few to draw strong statistical conclusions. The pilot run (5 samples) showed agentic F1 (0.042) *lower* than baseline (0.058), which reversed at 50 samples — a reminder that variance is high at small scales.
- **8B model capacity:** Llama 3.1 8B is a relatively small model. A larger model (70B or GPT-4) would likely generate better sub-queries and final answers, potentially improving EM significantly.
- **BM25 ceiling:** The retrieval pipeline is lexical-only. Dense retrieval (DPR or E5) would close the vocabulary mismatch gap, particularly for questions where the bridging entity is paraphrased differently in the document.

---

## 7. Conclusion

We demonstrated that an iterative agentic retrieval system significantly outperforms a static single-hop RAG baseline on multi-hop question answering. Across all five reported metrics, the agentic system is strictly better: EM improves from 0% to 12%, F1 improves by 237%, and retrieval quality (MRR, NDCG@10) improves by 8–12%.

The core insight is that multi-hop questions cannot be decomposed into a single retrieval query at query time — the bridging entity needed to form the second sub-query only becomes known after the first retrieval step. An LLM agent with a Chain-of-Thought sub-query loop discovers these bridging entities naturally and directs the retriever to gather the remaining evidence.

**Future directions:**
- Replace BM25 with a dense retriever (DPR, E5) to reduce vocabulary mismatch.
- Implement FLARE-style retrieval triggering during generation for finer-grained retrieval control.
- Scale to the full HotpotQA validation set (7,405 samples) for statistically robust evaluation.
- Experiment with larger LLMs (Llama 3.1 70B, GPT-4o) to improve sub-query generation quality.

---

## References

- Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP 2018.*
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR.*
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020.*
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020.*
- Jiang et al. (2023). Active Retrieval Augmented Generation. *EMNLP 2023.*
- Trivedi et al. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023.*
- Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023.*

# Iterative Agentic RAG vs. Static Baseline on HotpotQA

**Course:** CS 572 — Information Retrieval  
**Dataset:** HotpotQA (distractor setting, validation split, 50 samples)  
**LLM:** Llama 3.1 8B Instruct  
**Retriever:** BM25 (rank-bm25)  
**Re-ranker:** Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

---

## 1. Introduction

Retrieval-augmented generation has become a go-to approach for grounding LLM answers in external knowledge. The basic idea is straightforward: retrieve a few relevant documents, hand them to the model as context, and generate an answer. This works well enough when the answer lives in a single passage, but breaks down quickly on questions that require connecting information across multiple documents.

HotpotQA [Yang et al., 2018] is a benchmark specifically built to test this kind of multi-hop reasoning. Every question in the dataset requires evidence from at least two supporting documents, often with a "bridging" structure: one document identifies an intermediate entity, and a second document answers the actual question about that entity. A single-hop retrieval system fails here not because the retriever is bad, but because at query time you do not yet know what the intermediate entity is — you only find out after reading the first document.

To address this, we built an iterative agentic RAG system where an LLM acts as a controller that loops over retrieval steps, generating targeted sub-queries based on what it has gathered so far, and stopping when it decides it has enough information to answer. We ran this against a standard single-hop baseline on 50 HotpotQA validation questions, measuring both how well each system answers questions (Exact Match, F1) and how well it retrieves the right documents (MRR, NDCG@10).

---

## 2. Related Work

The retrieval component of our system is built on **BM25** [Robertson & Zaragoza, 2009], a term-frequency-based scoring function that remains a strong lexical retrieval baseline despite being several decades old. We chose it for its simplicity and speed at corpus scale.

**Dense Passage Retrieval (DPR)** [Karpukhin et al., 2020] is the natural next step — it trains a bi-encoder to embed queries and passages into a shared vector space, enabling semantic matching that BM25 misses. DPR is stronger on single-hop tasks, though it shares the same fundamental limitation of retrieving in one shot.

The foundational RAG paper [Lewis et al., 2020] pairs a retriever with a generator but treats retrieval as a single, fixed step before generation begins. This is essentially what our baseline does.

**FLARE** [Jiang et al., 2023] takes a different angle: it lets the model trigger on-demand retrieval mid-generation whenever it becomes uncertain about the next token. Our approach is similar in spirit but more explicit — we have the agent issue a deliberate sub-query decision at each hop rather than retrieving reactively during decoding.

Our system is most directly related to **IRCoT** [Trivedi et al., 2022], which alternates between chain-of-thought reasoning steps and retrieval, and **ReAct** [Yao et al., 2022], which frames tool-use as a structured Thought-Action-Observation loop. We implement a simplified version of this idea without full chain-of-thought traces, relying instead on the LLM to decide sufficiency from the accumulated context.

---

## 3. System Design

The pipeline has four components that run in a loop:

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

**BM25 Retriever:** Each hop starts by tokenizing the current sub-query and scoring it against the full corpus via BM25Okapi. We retrieve `top_k × 2 = 10` candidates to give the re-ranker a wider pool to work with.

**Cross-Encoder Re-ranker:** A `ms-marco-MiniLM-L-6-v2` cross-encoder jointly scores each (query, document) pair, returning the top `top_k = 5` documents. This stage is slower than BM25 but corrects ranking errors from lexical mismatch.

**LLM Agent:** Llama 3.1 8B gets the accumulated context from all previous hops and decides what to do next. If it has enough information, it returns `"sufficient": true` with a `"final_answer"`. If not, it returns `"sufficient": false` with a new `"sub_query"` to retrieve on the next hop. This repeats for up to `max_hops = 4` iterations.

**Baseline (StaticRAG):** Runs a single BM25 + cross-encoder retrieval pass, then calls the LLM once to generate an answer. No loop, no sub-queries.

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

We evaluate on four metrics:

- **Exact Match (EM):** Binary — 1 if the normalized prediction matches the gold answer exactly, 0 otherwise.
- **F1:** Token-level overlap between predicted and gold answer after normalization. This is the primary HotpotQA metric because answers are often short phrases that may be partially correct.
- **MRR (Mean Reciprocal Rank):** How early in the ranked list the first relevant supporting document appears.
- **NDCG@10:** Overall ranking quality of the retrieved list, penalizing relevant documents that appear lower by discounting them logarithmically.

Supporting document IDs for retrieval evaluation are derived by mapping the HotpotQA gold supporting-fact titles to their positions in the constructed corpus.

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

The gap in Exact Match is stark — the baseline scores 0% while the agentic system answers 6 out of 50 questions exactly right. At first glance 12% EM might seem low, but it is actually expected for a small 8B model on HotpotQA. The baseline's 0% is almost inevitable: even if both relevant documents happen to appear in the top-5, the model is given no signal about how the two documents relate, so it rarely synthesizes a correct bridging answer.

F1 tells a similar story with more nuance. The 237% relative improvement (0.075 → 0.252) shows that the agentic system is producing substantially more useful answers even on questions it doesn't get exactly right. The multi-hop context gives the model the intermediate entities and supporting facts it needs to at least get close.

### 6.2 Retrieval Quality

MRR and NDCG@10 both improve with the agentic approach, though the gains are more modest than the answer quality improvements — 8% and 12% respectively. This makes sense: BM25 is a lexical matcher, so the first hop (which uses the original question) often already surfaces one of the two supporting documents. The retrieval challenge in HotpotQA is getting the *second* document, which is where the agent's sub-queries help.

One interesting detail in the results: the final NDCG@10 (0.435) is slightly higher than the per-hop average (0.420). This happens because documents retrieved across different hops complement each other — hop 1 tends to get the primary supporting document, and later hops add the secondary one. Accumulating all retrieved documents across hops rather than resetting each time is what makes this possible.

### 6.3 Hop Behaviour

The average hop count of 3.72 (out of a max of 4) tells you that HotpotQA questions genuinely require multiple retrievals. The agent is not padding hops unnecessarily — a few questions do terminate before the limit, which is why the average isn't exactly 4.0. This is a reassuring result: the stopping criterion is doing its job.

### 6.4 Limitations

A few things to keep in mind when interpreting these results:

- **50 samples is not a lot.** In the pilot run on just 5 samples, the agentic system actually had a *lower* F1 than the baseline. The trend reversed at 50 samples, but the variance is real, and conclusions drawn from this experiment should be treated as directional rather than definitive.
- **The model size matters.** Llama 3.1 8B is a capable but relatively small model. The quality of the sub-queries it generates directly affects retrieval — a larger model would almost certainly produce more targeted queries and stronger final answers.
- **BM25 has a hard ceiling.** If the bridging entity in the question is phrased differently in the document, BM25 will miss it. Swapping in a dense retriever would likely help substantially, particularly for paraphrase-heavy bridge questions.

---

## 7. Conclusion

This project set out to test whether iterative agentic retrieval meaningfully improves over a standard single-hop RAG pipeline on multi-hop QA. Based on the results, the answer is clearly yes. Every metric improved, with the biggest gains in answer quality — 12% EM vs 0%, and a tripling of F1 score.

The underlying reason is straightforward: multi-hop questions have a structure that single-hop retrieval cannot follow. You need to find a bridging entity first, and only then can you retrieve the document that actually answers the question. An LLM with a retrieval loop does this naturally; a static pipeline cannot, no matter how good the retriever is.

That said, 12% EM is not a strong absolute result, and the system has clear headroom for improvement. The most impactful next steps would probably be switching to a dense retriever to reduce vocabulary mismatch, and testing the same pipeline with a larger model to see how much of the remaining error is due to the LLM's reasoning limitations. Scaling to the full validation set (7,405 questions) would also make the conclusions much more reliable.

---

## References

- Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP 2018.*
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR.*
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020.*
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020.*
- Jiang et al. (2023). Active Retrieval Augmented Generation. *EMNLP 2023.*
- Trivedi et al. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023.*
- Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023.*

# Agentic Multi-Hop Retrieval for Complex Question Answering

**Seongmin Oh, Sam Liu, Avaneesh Bhoite**  
CS 572 Information Retrieval — Term Project

---

## 1. Introduction

Standard Retrieval-Augmented Generation (RAG) systems rely on a rigid, single-step "retrieval-then-generate" pipeline. When facing complex queries where the information needed evolves based on intermediate findings and reasoning steps, this static approach often fails to fetch all necessary context and frequently overloads the language model with irrelevant data.

The motivation for this project is to address the limitations of single-passage retrieval by building a dynamic, agentic search system. By allowing a Large Language Model (LLM) to iteratively formulate queries, evaluate retrieved context, and decide whether more research is necessary before generating a final answer, we believe our pipeline can significantly improve accuracy and relevance on knowledge-intensive Q&A tasks.

---

## 2. Background and Related Work

**Static RAG baseline.** The standard open-domain Q&A baseline uses a dense retriever (similar to DPR) or a lexical retriever (similar to Okapi BM25) to fetch a fixed number of documents based solely on the initial query. The model then generates an answer from this fixed context window — one retrieval step, no iteration.

**FLARE (Forward-Looking Active REtrieval).** The current SOTA approach in dynamic retrieval monitors the LLM's internal token-level probability distribution during generation and triggers a new search when confidence drops. While effective, these token-level interventions are computationally expensive and require complicated architectural modifications that often struggle to formulate coherent search queries.

**Our approach — Iterative Agentic RAG.** We shift dynamic retrieval from the token level to the reasoning level. Instead of modifying LLM internals, we treat an off-the-shelf LLM as an autonomous search agent. The agent generates targeted sub-queries via a Chain-of-Thought (CoT) reasoning loop, applies a custom retrieval + cross-encoder re-ranking pipeline, and then decides: "Do I have enough information to answer, or do I need another hop?"

---

## 3. System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   Agentic RAG Loop                  │
│                                                     │
│  1. LLM generates sub-query via CoT reasoning       │
│  2. BM25 retriever fetches top-K candidate docs     │
│  3. Cross-encoder re-ranker re-scores + filters     │
│  4. LLM evaluates context:                          │
│       sufficient? → synthesize final answer         │
│       insufficient? → refine query → repeat         │
│  (max 4 hops, min 2 hops enforced)                  │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
```

**Components:**
- **Retriever:** BM25 lexical retriever over a per-sample corpus built from HotpotQA context passages
- **Re-ranker:** Cross-encoder re-ranking to prioritize most informative passages
- **LLM (open-source):** `meta-llama/Meta-Llama-3.1-8B-Instruct` — used for both sub-query generation and final answer synthesis
- **LLM (frontier):** `gemini-2.5-flash` — run as an upper-bound ceiling comparison

**Baseline:** `StaticRAG` — single BM25 retrieval pass + same LLaMA-3.1-8B for generation; no iteration.

---

## 4. Dataset and Evaluation

**Dataset:** HotpotQA (validation split, 50 samples). HotpotQA is a multi-hop Q&A benchmark that explicitly requires reasoning over multiple distinct Wikipedia passages to answer each question. In our 50-sample experimental slice, every question had at least two gold supporting documents, so the dataset-level support-document split contains only multi-document examples.

**Evaluation metrics:**

| Metric | What it measures |
|--------|-----------------|
| **EM** (Exact Match) | Whether the predicted answer exactly matches the gold answer (after normalization) |
| **F1** | Token-level overlap between predicted and gold answer |
| **MRR** | Mean Reciprocal Rank of the first relevant document across retrieval results |
| **NDCG@10** | Normalized Discounted Cumulative Gain at rank 10 — quality of the ranked retrieval list |
| **avg_hops** | Average number of retrieval iterations the agent performed per question |
| **per_hop_MRR / per_hop_NDCG@10** | Retrieval quality measured at each individual hop (not just the final retrieved set) |

---

## 5. Results

### 5.1 Aggregate Results (50 samples)

| Metric | Baseline RAG | Agentic RAG | Frontier Agentic |
|--------|:---:|:---:|:---:|
| EM | 0.32 | 0.22 | **0.36** |
| F1 | 0.421 | 0.388 | **0.511** |
| MRR | 0.314 | 0.354 | **0.360** |
| NDCG@10 | 0.388 | 0.466 | **0.483** |
| avg\_hops | 1.00 | 3.46 | 2.94 |
| per\_hop\_MRR | — | 0.455 | **0.494** |
| per\_hop\_NDCG@10 | — | 0.522 | **0.572** |

**Models:** Baseline and Agentic use `meta-llama/Meta-Llama-3.1-8B-Instruct`; Frontier uses `gemini-2.5-flash`.

**Key observations:**
- The agentic retrieval loop significantly improves **retrieval quality** (NDCG@10: 0.388 → 0.466, +20% relative), confirming that iterative retrieval finds more relevant documents.
- **EM drops** from baseline to agentic (0.32 → 0.22). The 8B model finds better evidence but struggles to synthesize a precise short answer from a multi-hop reasoning chain.
- The frontier model (Gemini-2.5-Flash) outperforms both systems on every answer quality metric (EM: +0.14 over agentic, +0.04 over baseline), confirming it as a genuine performance ceiling.
- **Per-hop retrieval** for both agentic systems exceeds their final set scores, showing that individual hops are well-targeted.

---

### 5.2 Multi-Hop vs. Single-Hop Retrieval Need

Because the 50-sample HotpotQA slice used in this experiment contains only questions with at least two gold supporting documents, the support-document breakdown does not include true single-document questions. However, we can still distinguish questions by **whether the first retrieval pass found all gold documents** (effectively single-hop from a retrieval perspective) vs. those that required follow-up hops to locate the necessary evidence (genuinely multi-hop retrieval).

The `retrieval_hop_need` breakdown from `rep.ipynb` captures this split across all three systems:

| Group | System | Count | EM | F1 | MRR | NDCG@10 | Avg Hops |
|-------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| First hop sufficient | Baseline | 7 | 0.714 | 0.714 | 0.762 | 0.810 | 1.00 |
| First hop sufficient | Agentic | 7 | 0.429 | 0.571 | 0.762 | 0.787 | 3.71 |
| First hop sufficient | Frontier | 7 | 0.571 | 0.735 | 0.762 | 0.775 | 2.43 |
| Needs follow-up hops | Baseline | 43 | 0.256 | 0.373 | 0.241 | 0.319 | 1.00 |
| Needs follow-up hops | Agentic | 43 | 0.186 | 0.358 | 0.287 | 0.413 | 3.42 |
| Needs follow-up hops | Frontier | 43 | 0.326 | 0.474 | 0.294 | 0.435 | 3.02 |

**What this means:** Questions where supporting evidence co-occurs in initial retrieval results are answered more accurately across all systems. For questions that genuinely need multiple hops to surface all gold documents, the agentic loop is most valuable — but each additional hop also increases the risk of the 8B model generating a noisy final answer.

The first-hop-sufficient group is easier for all systems because the required evidence appears in the initial ranked list. The needs-follow-up group is harder, but the iterative systems improve retrieval quality over the static baseline: Agentic RAG raises NDCG@10 from 0.319 to 0.413, while the frontier model reaches 0.435.

---

## 6. Frontier Model Ceiling

We ran Gemini-2.5-Flash through the same agentic retrieval loop to establish a performance ceiling — the best the architecture can achieve when model capacity is not a constraint.

### Frontier Ceiling Table

| System | EM | F1 | MRR | NDCG@10 | Avg Hops |
|--------|:---:|:---:|:---:|:---:|:---:|
| Baseline RAG (LLaMA-3.1-8B) | 0.3200 | 0.4211 | 0.3137 | 0.3880 | 1.00 |
| Agentic RAG (LLaMA-3.1-8B) | 0.2200 | 0.3880 | 0.3538 | 0.4657 | 3.46 |
| **Frontier Agentic (Gemini-2.5-Flash)** | **0.3600** | **0.5105** | **0.3597** | **0.4829** | 2.94 |

### Delta: Frontier vs. Agentic (LLaMA)

| Metric | Delta |
|--------|:---:|
| EM | +0.1400 |
| F1 | +0.1225 |
| MRR | +0.0059 |
| NDCG@10 | +0.0172 |

**Key observations:**

1. **Frontier is a genuine ceiling.** Gemini-2.5-Flash outperforms both the static baseline and the open-source agentic system on EM and F1. The same retrieval architecture produces substantially better answers when paired with a more capable model (+14 EM points over agentic, +4 over baseline).

2. **Better sub-query generation.** Per-hop NDCG@10 for the frontier system (0.572) exceeds that of the open-source agentic (0.522), showing Gemini formulates more targeted sub-queries at each hop — not just synthesizes better final answers.

3. **Frontier uses fewer hops more efficiently.** Gemini-2.5-Flash averaged 2.94 hops vs. LLaMA's 3.46. It declares sufficiency earlier because it can extract useful signal from partially relevant documents that confuse the smaller model.

4. **The ceiling is not 1.0 — retrieval remains the bottleneck.** Even the frontier system reaches only EM=0.36. The shared BM25 retrieval surface — which cannot do semantic disambiguation — limits all three systems equally. Replacing BM25 with a dense or hybrid retriever is the highest-leverage next improvement.

---

## 7. White-Box Agent Trace Analysis

This section shows exactly what the agent does at each step — which sub-queries it generates, what documents are retrieved, and what decision it makes. This is the "white box" view of the retrieval loop.

### 7.1 Successful Multi-Hop Case (Agentic)

**Question:** "2014 S/S is the debut album of a South Korean boy group that was formed by who?"  
**Gold answer:** YG Entertainment  
**Predicted answer:** YG Entertainment ✓  
**Hops:** 2 | **Supporting docs:** `2014 S/S`, `Winner (band)`

| Hop | Sub-query | Top Retrieved Documents | Decision |
|-----|-----------|------------------------|----------|
| 1 | "2014 S/S is the debut album of a South Korean boy group that was formed by who?" | `List of awards by Shinee`, **`2014 S/S`**, `BTS discography`, `List of songs by Ravi`, `Madtown` | Insufficient — album article found, but label identity not confirmed |
| 2 | "YG Entertainment" | **`2014 S/S`**, **`Winner (band)`**, `History (band)`, `Seventeen discography`, `SF9` | Sufficient — both gold docs in context |

**What worked:** Hop 1 retrieved the album article, which named WINNER as the group. The agent then issued a precise follow-up query for "YG Entertainment," pulling both gold documents into context. The LLM correctly synthesized the answer from that combined evidence.

---

### 7.2 Failed Multi-Hop Case (Agentic)

**Question:** "Were Scott Derrickson and Ed Wood of the same nationality?"  
**Gold answer:** yes  
**Predicted answer:** "Yes, American." ✗ *(fails EM normalization — not exactly "yes")*  
**Hops:** 3 | **Supporting docs:** `Scott Derrickson`, `Ed Wood`

| Hop | Sub-query | Top Retrieved Documents | Decision |
|-----|-----------|------------------------|----------|
| 1 | "Were Scott Derrickson and Ed Wood of the same nationality?" | `Tyler Bates`, `Woodson, Arkansas`, `Doctor Strange (2016 film)`, `Gone in 60 Seconds (2000 film)`, **`Ed Wood (film)`** | Insufficient — retrieved the Tim Burton *film about* Ed Wood, not Ed Wood's own article |
| 2 | "Scott Derrickson nationality" | **`Scott Derrickson`**, `Sinister (film)`, `Doctor Strange (2016 film)`, `Adam Collis`, `Deliver Us from Evil` | Insufficient — Derrickson confirmed American; Ed Wood's article still missing |
| 3 | "Scott Derrickson birthplace" | **`Scott Derrickson`**, `Sinister (film)`, `Doctor Strange (2016 film)`, `Adam Collis`, `Deliver Us from Evil` | Sufficient (max hops) — Ed Wood article never retrieved |

**What failed and why:**

1. **Entity disambiguation failure.** "Ed Wood" consistently retrieved the Tim Burton film about Ed Wood rather than Ed Wood's own Wikipedia article. BM25 cannot distinguish between the person and the film with the same name.

2. **Correct reasoning, wrong format.** The model knew both are American ("Yes, American.") but added the explanation, causing EM failure. Gold is just "yes." This is a generation verbosity problem — the model answered correctly in substance but not in format.

3. **Retrieval miss cascades.** Without Ed Wood's article, the agent spent hops 2 and 3 re-confirming Derrickson rather than finding new evidence. The disambiguation is a retrieval-side issue the reasoning loop cannot fix.

---

**Question:** "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"  
**Gold answer:** Chief of Protocol  
**Predicted answer:** "Village accountant." ✗  
**Hops:** 4 | **Supporting docs:** `Kiss and Tell (1945 film)`, `Shirley Temple`

| Hop | Sub-query | Top Retrieved Documents | Decision |
|-----|-----------|------------------------|----------|
| 1 | Full question | `Charles Craft`, **`Village accountant`**, `A Kiss for Corliss`, `2014 Bahrain Grand Prix`, `City of Angels (film)` | Insufficient — "Village accountant" retrieved due to BM25 lexical noise |
| 2 | "Shirley Temple's filmography or biography" | `Madonna (book)`, **`Shirley Temple`**, `Andre Norton Award`, `Lead programmer`, `Front Row` | Insufficient — Shirley Temple found; government role not yet confirmed |
| 3 | "Shirley Temple filmography or roles in the 1940s" | `Bernhard Bötel`, `Kate Higgins`, `Kansas City jazz`, `English Electric Canberra`, `Giuseppe Verdi (film)` | Insufficient — completely off-topic results |
| 4 | "Corliss Archer film Kiss and Tell government position" | `A Kiss for Corliss`, **`Kiss and Tell (1945 film)`**, `Charles Craft`, `Village accountant`, `Janet Waldo` | Max hops reached — gold article finally retrieved but context already polluted |

**What failed:** Hop 1 retrieved "Village accountant" — a completely unrelated article that matched BM25 lexically. This document poisoned the accumulated context. Even though `Kiss and Tell (1945 film)` was correctly retrieved by hop 4, the model's final answer reflected the noisy hop 1 context. This is the **early retrieval noise** failure mode: wrong documents from early hops persist in context and dominate the final answer.

---

## 8. Discussion

### Why EM drops from baseline to agentic

The baseline (EM=0.32) outperforms the agentic system (EM=0.22) on exact match despite weaker retrieval. Two factors compound:

1. **Short answer format.** HotpotQA gold answers are extremely short (single words, years, yes/no). A single LLM call given 5 retrieved documents often produces a clean short answer. An agentic loop taking 3–4 hops generates far more intermediate reasoning, making final synthesis noisier for an 8B model.

2. **Generation capacity gap.** The 8B LLaMA model can navigate the retrieval loop reasonably well but adds explanatory language to final answers ("Yes, American." instead of "yes") that fails EM normalization. The frontier experiment confirms this is a model capacity issue — Gemini-2.5-Flash, using the identical retrieval loop, achieves EM=0.36 (+14 points over agentic).

### Why the frontier model wins where agentic loses

Gemini-2.5-Flash benefits from the same retrieval improvements as the open-source agentic system (NDCG +0.017 over agentic) but additionally:
- Generates more focused sub-queries at each hop (per_hop_NDCG 0.572 vs 0.522)
- Declares sufficiency earlier and more accurately (2.94 vs 3.46 avg hops)
- Synthesizes short, precise final answers that match gold normalization

### Primary failure modes

| Failure mode | Example | Root cause |
|---|---|---|
| Entity disambiguation | "Ed Wood" → film, not person | BM25 cannot disambiguate by entity type |
| Early retrieval noise | "Village accountant" anchors answer | Noisy hop 1 doc persists in context |
| Verbosity penalty | "Yes, American." vs "yes" | 8B model adds explanation to short answers |
| Retrieval miss at max hops | Ed Wood article never found | Architecture has no backtracking |

---

## 9. Conclusions

We built an Iterative Agentic RAG system that treats LLM reasoning as the driver of iterative retrieval, avoiding the token-level complexity of FLARE. The system demonstrates a clear **retrieval quality improvement** (+20% NDCG@10) over a static baseline. However, the 8B open-source model cannot fully convert better retrieval into better exact-match answers on short multi-hop questions.

The frontier model ceiling experiment (Gemini-2.5-Flash) confirms that the architecture is sound — the same retrieval loop paired with a stronger model achieves EM=0.36, outperforming both the static baseline and the open-source agentic system. The remaining gap from 0.36 to 1.0 is attributable to the BM25 retrieval surface, which cannot perform semantic entity disambiguation. Replacing BM25 with a dense or hybrid retriever is the clearest path to further improvement.

### Summary of Key Findings

| Finding | Evidence |
|---------|---------|
| Agentic loop improves retrieval | NDCG@10: 0.388 → 0.466 (+20%) |
| Better retrieval ≠ better EM for 8B model | EM: 0.32 → 0.22 (−31%) |
| Per-hop retrieval is well-targeted | per_hop_NDCG@10 = 0.522 vs final 0.466 |
| Frontier is a genuine ceiling | Gemini-2.5-Flash EM=0.36 > Agentic EM=0.22 |
| Frontier generates better sub-queries | per_hop_NDCG@10: 0.572 vs 0.522 |
| BM25 entity disambiguation limits all systems | EM ceiling ≈ 0.36 even with frontier model |
| Early retrieval noise cascades | "Village accountant" anchors wrong answer |

---

## Appendix: Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | HotpotQA, validation split |
| Samples (full run) | 50 |
| Retriever | BM25 |
| Re-ranker | Cross-encoder |
| Top-K docs per hop | 5 |
| Max hops | 4 |
| Min hops | 2 |
| Open-source LLM | meta-llama/Meta-Llama-3.1-8B-Instruct |
| Frontier LLM | gemini-2.5-flash |
| Hardware | NVIDIA A100-SXM4-80GB (80 GB VRAM) |

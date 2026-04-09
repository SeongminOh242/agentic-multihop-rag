# Agentic Multi-Hop Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate an "Iterative Agentic RAG" system that uses an LLM agent to dynamically generate sub-queries, retrieve and re-rank documents across multiple hops, and synthesize answers for complex multi-hop questions.

**Architecture:** An LLM (e.g., GPT or open-source equivalent) orchestrated via LangChain acts as an autonomous search agent. For each user query, the agent uses Chain-of-Thought (CoT) reasoning to generate targeted sub-queries, retrieves candidate documents via BM25/dense retrieval, re-ranks them with a cross-encoder, evaluates sufficiency, and either synthesizes a final answer or triggers another hop.

**Tech Stack:** Python, LangChain, HotpotQA dataset, BM25 (rank_bm25), sentence-transformers (cross-encoder), OpenAI or HuggingFace LLM, FAISS or Elasticsearch (optional), scikit-learn, datasets (HuggingFace)

---

## File Structure

```
CS572/Final/
├── data/
│   ├── hotpotqa_sample.json         # Subset of HotpotQA for development
│   └── hotpotqa_corpus.json         # Full document corpus
├── src/
│   ├── data_loader.py               # Load and parse HotpotQA
│   ├── retriever.py                 # BM25 + dense retrieval
│   ├── reranker.py                  # Cross-encoder re-ranking
│   ├── baseline_rag.py              # Static single-hop RAG pipeline
│   ├── agent.py                     # Iterative agentic RAG loop
│   └── evaluator.py                 # NDCG@10, MRR, EM, F1, avg hops
├── notebooks/
│   └── analysis.ipynb               # Result visualization and analysis
├── tests/
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_baseline_rag.py
│   ├── test_agent.py
│   └── test_evaluator.py
├── results/
│   └── (JSON result dumps per run)
├── requirements.txt
└── README.md
```

---

## WEEK 1 — Environment, Data, and Baseline RAG

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `README.md`

- [ ] **Step 1: Initialize project structure**

```bash
cd /Users/Eddie/Documents/CS572/Final
mkdir -p src tests data results notebooks
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
langchain
langchain-community
langchain-openai
openai
sentence-transformers
rank-bm25
datasets
faiss-cpu
scikit-learn
numpy
pandas
tqdm
python-dotenv
pytest
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 4: Create .env for API keys**

```bash
touch .env
# Add: OPENAI_API_KEY=your_key_here
```

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt README.md .env.example
git commit -m "chore: project setup and dependencies"
```

---

### Task 2: Data Loader

**Files:**
- Create: `src/data_loader.py`
- Create: `tests/test_data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py
from src.data_loader import load_hotpotqa, build_corpus

def test_load_hotpotqa_returns_list():
    samples = load_hotpotqa(split="validation", max_samples=10)
    assert isinstance(samples, list)
    assert len(samples) == 10

def test_sample_has_required_fields():
    samples = load_hotpotqa(split="validation", max_samples=1)
    s = samples[0]
    assert "question" in s
    assert "answer" in s
    assert "supporting_facts" in s
    assert "context" in s

def test_build_corpus_returns_list_of_strings():
    samples = load_hotpotqa(split="validation", max_samples=5)
    corpus = build_corpus(samples)
    assert isinstance(corpus, list)
    assert all(isinstance(d, str) for d in corpus)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_data_loader.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement data_loader.py**

```python
# src/data_loader.py
from datasets import load_dataset

def load_hotpotqa(split="validation", max_samples=500):
    """Load HotpotQA samples from HuggingFace datasets."""
    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    samples = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        samples.append({
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": item["supporting_facts"],
            "context": item["context"],  # list of [title, sentences]
            "id": item["id"],
        })
    return samples

def build_corpus(samples):
    """Flatten all context passages into a single list of strings."""
    corpus = []
    for s in samples:
        titles = s["context"]["title"]
        sentences_list = s["context"]["sentences"]
        for title, sentences in zip(titles, sentences_list):
            passage = title + ". " + " ".join(sentences)
            corpus.append(passage)
    return list(set(corpus))  # deduplicate
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_data_loader.py -v
```
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: HotpotQA data loader and corpus builder"
```

---

### Task 3: BM25 Retriever

**Files:**
- Create: `src/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
from src.retriever import BM25Retriever

def test_bm25_returns_top_k_docs():
    corpus = [
        "The Eiffel Tower is in Paris France.",
        "Python is a programming language.",
        "Paris is the capital of France.",
        "Machine learning uses statistical methods.",
    ]
    retriever = BM25Retriever(corpus)
    results = retriever.retrieve("Eiffel Tower Paris", top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert "text" in results[0]
    assert "score" in results[0]

def test_bm25_most_relevant_doc_ranked_first():
    corpus = [
        "The Eiffel Tower is in Paris France.",
        "Python is a programming language.",
    ]
    retriever = BM25Retriever(corpus)
    results = retriever.retrieve("Eiffel Tower", top_k=2)
    assert "Eiffel" in results[0]["text"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retriever.py -v
```
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement BM25Retriever**

```python
# src/retriever.py
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"text": self.corpus[i], "score": float(scores[i]), "doc_id": i}
                for i in top_indices]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_retriever.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retriever.py tests/test_retriever.py
git commit -m "feat: BM25Okapi retriever with top-k document retrieval"
```

---

### Task 4: Cross-Encoder Re-Ranker

**Files:**
- Create: `src/reranker.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reranker.py
from src.reranker import CrossEncoderReranker

def test_reranker_returns_sorted_results():
    reranker = CrossEncoderReranker()
    query = "Where is the Eiffel Tower?"
    docs = [
        {"text": "Python is a programming language.", "score": 0.9, "doc_id": 0},
        {"text": "The Eiffel Tower is located in Paris, France.", "score": 0.5, "doc_id": 1},
    ]
    reranked = reranker.rerank(query, docs, top_k=2)
    assert len(reranked) == 2
    assert "Eiffel" in reranked[0]["text"]

def test_reranker_top_k_limits_results():
    reranker = CrossEncoderReranker()
    query = "capital of France"
    docs = [{"text": f"doc {i}", "score": float(i), "doc_id": i} for i in range(5)]
    reranked = reranker.rerank(query, docs, top_k=3)
    assert len(reranked) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_reranker.py -v
```
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement CrossEncoderReranker**

```python
# src/reranker.py
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return []
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        reranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_reranker.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/reranker.py tests/test_reranker.py
git commit -m "feat: cross-encoder re-ranker using sentence-transformers"
```

---

### Task 5: Static RAG Baseline

**Files:**
- Create: `src/baseline_rag.py`
- Create: `tests/test_baseline_rag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline_rag.py
from unittest.mock import MagicMock, patch
from src.baseline_rag import StaticRAG

def test_static_rag_returns_answer_and_docs():
    corpus = [
        "The Eiffel Tower is located in Paris, France.",
        "Paris is the capital of France.",
        "Python is a programming language.",
    ]
    rag = StaticRAG(corpus)
    with patch.object(rag, '_call_llm', return_value="Paris, France"):
        result = rag.answer("Where is the Eiffel Tower?")
    assert "answer" in result
    assert "retrieved_docs" in result
    assert isinstance(result["retrieved_docs"], list)

def test_static_rag_retrieved_docs_count():
    corpus = ["doc " + str(i) for i in range(20)]
    rag = StaticRAG(corpus, top_k=5)
    with patch.object(rag, '_call_llm', return_value="some answer"):
        result = rag.answer("test query")
    assert len(result["retrieved_docs"]) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_baseline_rag.py -v
```
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement StaticRAG**

```python
# src/baseline_rag.py
import os
from openai import OpenAI
from src.retriever import BM25Retriever
from src.reranker import CrossEncoderReranker

class StaticRAG:
    def __init__(self, corpus: list[str], top_k: int = 5):
        self.retriever = BM25Retriever(corpus)
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def answer(self, question: str) -> dict:
        raw_docs = self.retriever.retrieve(question, top_k=self.top_k * 2)
        reranked_docs = self.reranker.rerank(question, raw_docs, top_k=self.top_k)
        context = "\n\n".join(d["text"] for d in reranked_docs)
        prompt = (
            f"Answer the following question using only the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        answer = self._call_llm(prompt)
        return {
            "answer": answer,
            "retrieved_docs": reranked_docs,
            "num_hops": 1,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_baseline_rag.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/baseline_rag.py tests/test_baseline_rag.py
git commit -m "feat: static RAG baseline with BM25 + cross-encoder + LLM"
```

---

## WEEK 2 — Iterative Agentic RAG

### Task 6: Agentic RAG Loop

**Files:**
- Create: `src/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
from unittest.mock import MagicMock, patch
from src.agent import AgenticRAG

def test_agent_returns_answer_with_hop_count():
    corpus = [
        "Albert Einstein was born in Ulm, Germany.",
        "Ulm is a city in the state of Baden-Württemberg.",
        "Einstein developed the theory of relativity.",
    ]
    agent = AgenticRAG(corpus, max_hops=3)
    with patch.object(agent, '_call_llm', side_effect=[
        '{"sub_query": "Einstein birthplace", "sufficient": false}',
        '{"sub_query": "Ulm Germany state", "sufficient": false}',
        '{"sub_query": "", "sufficient": true, "final_answer": "Baden-Württemberg"}',
    ]):
        result = agent.answer("What state in Germany was Einstein born in?")
    assert "answer" in result
    assert "num_hops" in result
    assert result["num_hops"] <= 3

def test_agent_stops_at_max_hops():
    corpus = ["unrelated document " + str(i) for i in range(10)]
    agent = AgenticRAG(corpus, max_hops=2)
    with patch.object(agent, '_call_llm', return_value='{"sub_query": "still searching", "sufficient": false}'):
        result = agent.answer("impossible question")
    assert result["num_hops"] <= 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_agent.py -v
```
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement AgenticRAG**

```python
# src/agent.py
import os
import json
from openai import OpenAI
from src.retriever import BM25Retriever
from src.reranker import CrossEncoderReranker

AGENT_PROMPT = """You are a research agent answering complex questions by searching documents iteratively.

Accumulated context so far:
{context}

Original question: {question}
Current hop: {hop}/{max_hops}

Based on the context, decide:
1. If you have enough information, set "sufficient": true and provide "final_answer".
2. If you need more information, set "sufficient": false and provide a specific "sub_query" to search next.

Respond ONLY with valid JSON:
{{"sub_query": "...", "sufficient": false}}
OR
{{"sub_query": "", "sufficient": true, "final_answer": "..."}}
"""

class AgenticRAG:
    def __init__(self, corpus: list[str], top_k: int = 5, max_hops: int = 4):
        self.retriever = BM25Retriever(corpus)
        self.reranker = CrossEncoderReranker()
        self.top_k = top_k
        self.max_hops = max_hops
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def answer(self, question: str) -> dict:
        accumulated_context = []
        all_retrieved_docs = []
        hop = 0
        current_query = question

        for hop in range(1, self.max_hops + 1):
            raw_docs = self.retriever.retrieve(current_query, top_k=self.top_k * 2)
            reranked_docs = self.reranker.rerank(current_query, raw_docs, top_k=self.top_k)
            all_retrieved_docs.extend(reranked_docs)

            new_context = "\n".join(d["text"] for d in reranked_docs)
            accumulated_context.append(f"[Hop {hop} — Query: {current_query}]\n{new_context}")

            prompt = AGENT_PROMPT.format(
                context="\n\n".join(accumulated_context),
                question=question,
                hop=hop,
                max_hops=self.max_hops,
            )
            raw_response = self._call_llm(prompt)

            try:
                decision = json.loads(raw_response)
            except json.JSONDecodeError:
                decision = {"sufficient": False, "sub_query": question}

            if decision.get("sufficient", False):
                return {
                    "answer": decision.get("final_answer", ""),
                    "num_hops": hop,
                    "retrieved_docs": all_retrieved_docs,
                    "sub_queries": [question] + [decision.get("sub_query", "")],
                }
            current_query = decision.get("sub_query", question)

        # Max hops reached — force final answer
        context_str = "\n\n".join(accumulated_context)
        fallback_prompt = (
            f"Based on this context, answer the question as best you can.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
        )
        final_answer = self._call_llm(fallback_prompt)
        return {
            "answer": final_answer,
            "num_hops": self.max_hops,
            "retrieved_docs": all_retrieved_docs,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_agent.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: iterative agentic RAG with CoT hop loop and max_hops guard"
```

---

### Task 7: Evaluator (NDCG@10, MRR, EM, F1)

**Files:**
- Create: `src/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluator.py
from src.evaluator import (
    exact_match, f1_score_tokens,
    mean_reciprocal_rank, ndcg_at_k
)

def test_exact_match_true():
    assert exact_match("Paris", "Paris") is True

def test_exact_match_case_insensitive():
    assert exact_match("paris", "Paris") is True

def test_exact_match_false():
    assert exact_match("London", "Paris") is False

def test_f1_score_partial_overlap():
    score = f1_score_tokens("Paris France", "Paris is in France")
    assert 0.0 < score < 1.0

def test_f1_score_exact():
    assert f1_score_tokens("Paris", "Paris") == 1.0

def test_f1_score_no_overlap():
    assert f1_score_tokens("London", "Tokyo") == 0.0

def test_mrr_first_relevant():
    # Relevant doc is at position 0 → MRR = 1.0
    retrieved = [{"doc_id": 0}, {"doc_id": 1}, {"doc_id": 2}]
    relevant = {0}
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0

def test_mrr_second_relevant():
    retrieved = [{"doc_id": 1}, {"doc_id": 0}, {"doc_id": 2}]
    relevant = {0}
    assert abs(mean_reciprocal_rank(retrieved, relevant) - 0.5) < 1e-6

def test_ndcg_at_k_perfect():
    retrieved = [{"doc_id": 0}, {"doc_id": 1}]
    relevant = {0, 1}
    score = ndcg_at_k(retrieved, relevant, k=2)
    assert abs(score - 1.0) < 1e-6

def test_ndcg_at_k_none_relevant():
    retrieved = [{"doc_id": 0}, {"doc_id": 1}]
    relevant = {99}
    assert ndcg_at_k(retrieved, relevant, k=2) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_evaluator.py -v
```
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement evaluator.py**

```python
# src/evaluator.py
import math
import re
from collections import Counter

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize(prediction) == _normalize(ground_truth)

def f1_score_tokens(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def mean_reciprocal_rank(retrieved: list[dict], relevant_ids: set) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc["doc_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[dict], relevant_ids: set, k: int = 10) -> float:
    def dcg(hits):
        return sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))

    hits = [1.0 if doc["doc_id"] in relevant_ids else 0.0 for doc in retrieved[:k]]
    ideal = sorted(hits, reverse=True)
    actual_dcg = dcg(hits)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_batch(results: list[dict], ground_truths: list[dict]) -> dict:
    """
    results: list of {"answer": str, "retrieved_docs": [...], "num_hops": int}
    ground_truths: list of {"answer": str, "supporting_doc_ids": set}
    """
    em_scores, f1_scores, mrr_scores, ndcg_scores, hops = [], [], [], [], []
    for result, gt in zip(results, ground_truths):
        em_scores.append(float(exact_match(result["answer"], gt["answer"])))
        f1_scores.append(f1_score_tokens(result["answer"], gt["answer"]))
        relevant = gt.get("supporting_doc_ids", set())
        mrr_scores.append(mean_reciprocal_rank(result["retrieved_docs"], relevant))
        ndcg_scores.append(ndcg_at_k(result["retrieved_docs"], relevant, k=10))
        hops.append(result["num_hops"])
    return {
        "EM": sum(em_scores) / len(em_scores),
        "F1": sum(f1_scores) / len(f1_scores),
        "MRR": sum(mrr_scores) / len(mrr_scores),
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores),
        "avg_hops": sum(hops) / len(hops),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_evaluator.py -v
```
Expected: PASS (all 9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluation metrics — EM, F1, MRR, NDCG@10, avg hops"
```

---

## WEEK 3 — Experiments, Analysis, and Report

### Task 8: End-to-End Experiment Runner

**Files:**
- Create: `src/run_experiment.py`

- [ ] **Step 1: Implement experiment runner**

```python
# src/run_experiment.py
import json
import os
from dotenv import load_dotenv
from src.data_loader import load_hotpotqa, build_corpus
from src.baseline_rag import StaticRAG
from src.agent import AgenticRAG
from src.evaluator import evaluate_batch

load_dotenv()

def build_ground_truths(samples, corpus):
    """Map supporting fact titles to doc_ids in corpus for retrieval evaluation."""
    title_to_id = {}
    for idx, doc in enumerate(corpus):
        title = doc.split(".")[0]
        title_to_id[title] = idx
    ground_truths = []
    for s in samples:
        supporting_titles = set(s["supporting_facts"]["title"])
        supporting_ids = {title_to_id[t] for t in supporting_titles if t in title_to_id}
        ground_truths.append({
            "answer": s["answer"],
            "supporting_doc_ids": supporting_ids,
        })
    return ground_truths

def run(num_samples=50):
    samples = load_hotpotqa(split="validation", max_samples=num_samples)
    corpus = build_corpus(samples)
    ground_truths = build_ground_truths(samples, corpus)
    questions = [s["question"] for s in samples]

    print(f"Running baseline RAG on {num_samples} samples...")
    baseline = StaticRAG(corpus, top_k=5)
    baseline_results = [baseline.answer(q) for q in questions]
    baseline_metrics = evaluate_batch(baseline_results, ground_truths)

    print(f"Running Agentic RAG on {num_samples} samples...")
    agent = AgenticRAG(corpus, top_k=5, max_hops=4)
    agent_results = [agent.answer(q) for q in questions]
    agent_metrics = evaluate_batch(agent_results, ground_truths)

    report = {
        "baseline": baseline_metrics,
        "agentic": agent_metrics,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"{'Metric':<15} {'Baseline':>10} {'Agentic':>10}")
    print("-" * 37)
    for metric in ["EM", "F1", "MRR", "NDCG@10", "avg_hops"]:
        print(f"{metric:<15} {baseline_metrics[metric]:>10.4f} {agent_metrics[metric]:>10.4f}")

if __name__ == "__main__":
    run(num_samples=50)
```

- [ ] **Step 2: Run a small pilot (5 samples) to verify no crashes**

```bash
python -c "from src.run_experiment import run; run(num_samples=5)"
```
Expected: Prints results table with no errors, writes `results/experiment_results.json`

- [ ] **Step 3: Commit**

```bash
git add src/run_experiment.py
git commit -m "feat: end-to-end experiment runner comparing baseline vs agentic RAG"
```

---

### Task 9: Full Experiment Run + Results

- [ ] **Step 1: Run full experiment (50 samples)**

```bash
python src/run_experiment.py
```
Expected: Takes 10-20 minutes; writes results to `results/experiment_results.json`

- [ ] **Step 2: Check results are sane**

```bash
python -c "import json; d=json.load(open('results/experiment_results.json')); print(json.dumps(d, indent=2))"
```
Expected: Both baseline and agentic have EM, F1, MRR, NDCG@10 between 0 and 1; agentic avg_hops > 1.0

- [ ] **Step 3: Commit results**

```bash
git add results/experiment_results.json
git commit -m "results: experiment run on 50 HotpotQA validation samples"
```

---

### Task 10: Analysis Notebook

**Files:**
- Create: `notebooks/analysis.ipynb`

- [ ] **Step 1: Create notebook with result visualization**

Open `notebooks/analysis.ipynb` and add cells:

```python
# Cell 1: Load results
import json
import pandas as pd
import matplotlib.pyplot as plt

with open("../results/experiment_results.json") as f:
    results = json.load(f)

df = pd.DataFrame([results["baseline"], results["agentic"]], index=["Baseline", "Agentic"])
print(df)
```

```python
# Cell 2: Bar chart comparison
metrics = ["EM", "F1", "MRR", "NDCG@10"]
df[metrics].plot(kind="bar", figsize=(10, 5), title="Baseline vs Agentic RAG")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../results/metrics_comparison.png", dpi=150)
plt.show()
```

```python
# Cell 3: Hop distribution (if you logged per-sample hop counts)
print(f"Baseline avg hops: {results['baseline']['avg_hops']:.2f}")
print(f"Agentic  avg hops: {results['agentic']['avg_hops']:.2f}")
```

- [ ] **Step 2: Run all cells, verify no errors**

```bash
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/analysis.ipynb results/metrics_comparison.png
git commit -m "feat: analysis notebook with baseline vs agentic comparison charts"
```

---

### Task 11: Final Report Write-Up

- [ ] **Step 1: Outline the report sections**

The report should cover (in a document or PDF):
1. **Introduction** — problem motivation, why single-hop RAG fails on HotpotQA
2. **Related Work** — DPR, BM25, FLARE, standard RAG
3. **System Design** — architecture diagram (data flow: query → sub-queries → retrieval → rerank → hop decision → answer)
4. **Experimental Setup** — HotpotQA dataset, 50 samples, evaluation metrics
5. **Results** — Table with EM, F1, MRR, NDCG@10, avg_hops for both systems
6. **Analysis** — Where does agentic outperform? Where does it fail? Are more hops always better?
7. **Conclusion** — Summary and future directions (e.g., dense retriever, FLARE comparison)

- [ ] **Step 2: Write up results section using actual numbers from experiment_results.json**

Fill in the table:

| Metric   | Baseline RAG | Agentic RAG |
|----------|-------------|-------------|
| EM       | (from results) | (from results) |
| F1       | (from results) | (from results) |
| MRR      | (from results) | (from results) |
| NDCG@10  | (from results) | (from results) |
| Avg Hops | 1.00         | (from results) |

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "docs: final report and analysis complete"
```

---

## Self-Review Against Spec

| Spec Requirement | Covered In |
|---|---|
| Iterative agentic RAG with LLM orchestration | Task 6 (`agent.py`) |
| Sub-query generation via CoT loop | Task 6 (`AGENT_PROMPT`) |
| Custom retrieval pipeline | Task 3 (`retriever.py`) |
| Cross-encoder re-ranking | Task 4 (`reranker.py`) |
| HotpotQA dataset | Task 2 (`data_loader.py`) |
| NDCG@10 evaluation | Task 7 (`evaluator.py`) |
| MRR evaluation | Task 7 (`evaluator.py`) |
| Exact Match (EM) evaluation | Task 7 (`evaluator.py`) |
| F1-score evaluation | Task 7 (`evaluator.py`) |
| Average number of hops tracking | Tasks 6 + 7 |
| Comparison vs static RAG baseline | Task 5 + 8 |
| LangChain orchestration framework | *(Optional extension — see note below)* |

> **Note on LangChain:** The plan implements the agent loop manually for transparency and control. LangChain can be added as an orchestration wrapper in Task 6 if preferred, using `langchain.agents.AgentExecutor` and `langchain.tools.Tool` to wrap the retriever. This adds complexity without changing the core logic.

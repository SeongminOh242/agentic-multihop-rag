# agentic-multihop-rag

HotpotQA project comparing a static single-hop RAG baseline against an iterative agentic multi-hop RAG pipeline.

## What Changed

The experiment runner now supports:

- aggregate metrics for baseline and agentic systems
- white-box per-sample traces with hop-by-hop retrieved documents and sub-queries
- breakdowns for questions that need multiple supporting documents vs those that do not
- intermediate agent decisions for each hop
- optional frontier-model ceiling runs using Gemini

## Main Outputs

- `results/experiment_results.json`: compact aggregate metrics
- `results/experiment_details.json`: white-box traces, slice breakdowns, and case studies

## Colab Example

```python
from src.run_experiment import run

results = run(
    num_samples=50,
    baseline_provider="hf",
    baseline_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    agent_provider="hf",
    agent_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    frontier_provider=None,
)

print(results)
```

Set `frontier_provider="gemini"` only when your Gemini API project has credits. If Gemini returns `RESOURCE_EXHAUSTED`, the runner skips the frontier ceiling instead of reporting fallback retrieval as a model result.

## White-Box Analysis in Colab

```python
from src.reporting import (
    load_detailed_results,
    print_case_studies,
    print_intermediate_trace,
    print_support_doc_breakdown,
    print_support_doc_comparison,
)

details = load_detailed_results()
print_support_doc_comparison(details)
print_support_doc_breakdown(details, system_name="agentic")
print_case_studies(details, system_name="agentic", group_name="successful_multi_doc")
print_case_studies(details, system_name="agentic", group_name="failed_multi_doc")
print_intermediate_trace(details, system_name="agentic", trace_index=0)
```

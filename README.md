# agentic-multihop-rag

## What changed (to address feedback)

- **Multi-hop vs non-multi-hop comparison**: running `src/run_experiment.py` now writes:
  - `results/per_example_traces.json` (per-question traces + labels)
  - `results/multihop_vs_singlehop.md` (subset metrics + example traces)
- **Intermediate agent results**: agent responses now include `hop_decisions` and the experiment saves a compact per-hop summary (queries + top titles).
- **Frontier model ceiling**: `src/run_model_ceiling.py` runs the same pipelines across multiple model names and saves `results/model_ceiling.json`.

## Run

```bash
python -m src.run_experiment --num-samples 50
```

After running, open:
- `results/multihop_vs_singlehop.md`
- `results/per_example_traces.json`

## Frontier ceiling experiment

```bash
python -m src.run_model_ceiling --num-samples 50 --models gpt-4o-mini gpt-4o
```

This writes `results/model_ceiling.json`.

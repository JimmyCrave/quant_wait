#!/bin/sh
# Correction-duration study + audits. Run from quant_reasoning_lab/. Env: ../.venv (py3.12, torch 2.5.1, vLLM 0.7.0 wheel). Every GPU command uses `env -u LD_PRELOAD`.
cd "$(dirname "$0")/.."
P=../.venv/bin/python
# ---- phase 1: audits (CPU)
$P src/audit1_exhausted.py        # budget-exhausted but scorer-correct split            -> dur_study/audit/audit1_*.csv
$P src/audit2_paired.py           # group-mean difference vs paired difference           -> audit2_*.csv
$P src/audit3_markers.py          # 49/50 marker mapping + space-free variants           -> audit3_*.csv
$P src/audit4_budget.py           # B5 budget includes the BF16 window (raw files)       -> audit4_*.csv
$P src/make_human_pack.py         # 60 blind windows (status NOT COMPLETED)              -> dur_study/human_pack_60/
# ---- phase 2: anchors (hash-selected before any outcome), requests, generation
$P src/b7_pipeline.py build       # 32 anchors, 2256 runs (576 main, 144 long T=8192, 1536 consistency T=2048)
sh src/run_b7_stageA.sh           # stage A: no-switch runs + first-model windows (awq3 on GPUs 0-3, bf16 on GPUs 4-7)
$P src/b7_pipeline.py stageB      # stage-2 requests = prompt + prefix + window token ids (seed+100, max_tokens = T - window)
sh src/run_b7_stageB.sh
$P src/b7_pipeline.py merge       # -> dur_study/data/final_b7.jsonl (full token ids, window span, finish reasons, forward-cost counters)
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 $P src/b7_prepare.py cuda:0   # segmentation + repetition detector features
$P src/b7_outcomes.py             # -> dur_study/data/b7_outcomes.csv
$P src/b7_consistency.py          # audit item 5 (structural checks + distribution comparison)
$P src/b7_analyze.py              # -> dur_results.csv, dur_decomposition.csv, dur_cost.csv, figs/
$P src/b7_extra.py                # descriptive survival-matched / split / heterogeneity tables
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 $P src/seed_repro_check.py awq3   # why same-seed first tokens differ

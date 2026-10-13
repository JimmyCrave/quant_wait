#!/bin/sh
# Commands in the order they were run (stage 2). Env: ../.venv (py3.12, torch 2.5.1, vLLM 0.7.0 wheel, transformers 4.47.1). Every GPU command uses `env -u LD_PRELOAD` (cluster vGPU shim segfaults vLLM).
# Parallelism was done by hand over the 8 GPUs (shards); commands below show one shard / the full invocation.
cd "$(dirname "$0")"
# B0  extra seeds 43,44,45 for BF16 and AWQ3 (same 100 questions), token ids saved (inference.py diff: code_changes_stage2_inference.diff)
sh src/run_replicates.sh
# B1  segmentation + candidate detectors; Qwen2.5-14B labels (7 shards), phi-4 labels (8 shards), audit windows and annotation pack
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python src/build_steps.py data/steps_seed42.jsonl            # + data/steps_rep_bf16.jsonl 43,44,45 bf16 ; data/steps_rep_awq3.jsonl 43,44,45 awq3
../.venv/bin/python src/make_audit.py
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python src/label_llm.py --model models/Qwen2.5-14B-Instruct --steps data/steps_seed42.jsonl --shard 0 --nshards 2 --out data/labels_qwen14b_seed42_sh0.jsonl
sh src/run_phi_shard.sh 0 0       # shards 0..7 -> data/labels_phi4_all_sh*.jsonl
for cfg in "Qwen2.5-14B-Instruct 1 audit_qwen14b_ev" "Qwen2.5-14B-Instruct 0 audit_qwen14b_noev" "phi-4 1 audit_phi4_ev"; do set -- $cfg
  env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python src/label_llm.py --model models/$1 --steps data/steps_all.jsonl --windows data/audit_windows.json --evidence $2 --out data/$3.jsonl; done
../.venv/bin/python src/audit_agreement.py
# B2  
../.venv/bin/python src/analyze_b2.py --labels 'data/labels_qwen14b_*.jsonl' --labels2 'data/labels_phi4_all_sh*.jsonl' --out state_transition_results.csv --figdir figs --B 500
# B4  teacher-forced stats (pass 1: per-position scalars; pass 2: vocabulary-level mass change in pre-specified windows)
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python src/tf_stats.py --out_dir data/tf
../.venv/bin/python src/analyze_b4.py --labels 'data/labels_qwen14b_*.jsonl' --out results/distribution_results.csv --B 300
../.venv/bin/python src/b4_windows.py; env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python src/tf_direction.py --shard 0 --nshards 3; ../.venv/bin/python src/summarize_direction.py
# B3  prefix bank (fixed before continuation), continuations, analysis
../.venv/bin/python src/make_prefix_bank.py --labels 'data/labels_qwen14b_*.jsonl' --out prefix_bank.jsonl
sh src/run_b3.sh; cat data/b3/out_*.jsonl > data/b3/final_b3.jsonl
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python src/prepare_cont.py b3 data/b3 cuda:0; ../.venv/bin/python src/analyze_b3.py
# B5  anchors/branches (fixed before running), 2-stage BF16-window pipeline, analysis
../.venv/bin/python src/make_b5_requests.py 'data/labels_qwen14b_*.jsonl' data/b5
../.venv/bin/python src/b5_pipeline.py stage1 data/b5
sh src/run_b5_stage.sh awq3 data/b5/stage1_awq3.jsonl data/b5/s1_awq3 2,4,5,7;  sh src/run_b5_stage.sh bf16 data/b5/stage1_bf16.jsonl data/b5/s1_bf16 6
../.venv/bin/python src/b5_pipeline.py stage2 data/b5; sh src/run_b5_stage.sh awq3 data/b5/stage2_awq3.jsonl data/b5/s2_awq3 2,4,5,6,7
../.venv/bin/python src/b5_pipeline.py merge data/b5
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python src/prepare_cont.py b5 data/b5 cuda:0; ../.venv/bin/python src/analyze_b5.py

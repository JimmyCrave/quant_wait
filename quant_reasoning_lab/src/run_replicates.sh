#!/bin/sh
# Extra seeds (43,44,45) for BF16 and AWQ3 on the SAME 100 MATH-500 questions; identical settings to the seed-42 runs.
# Differences vs seed-42 runs: additionally stores generated_token_ids.
cd "$(dirname "$0")/../../Quantized-Reasoning-Models"
COMMON="--dataset MATH-500 --max_samples 100 --temperature 0.6 --top_p 0.95 --max_new_tokens 32768 --max_model_length 34816"
BF16=./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B
AWQ3=./outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1
LOG=../quant_reasoning_lab/logs
g=0
for seed in 43 44 45; do
  env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=$g ../.venv/bin/python inference.py --model $BF16 $COMMON --seed $seed --output_dir ./outputs/replicates/bf16_s$seed > $LOG/rep_bf16_s$seed.log 2>&1 &
  g=$((g+1))
  env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=$g ../.venv/bin/python inference.py --model $AWQ3 $COMMON --seed $seed --output_dir ./outputs/replicates/awq3_s$seed > $LOG/rep_awq3_s$seed.log 2>&1 &
  g=$((g+1))
done
wait

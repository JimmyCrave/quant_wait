#!/bin/sh
# Three groups, same 100 MATH-500 questions (lighteval shuffles docs with Random(42) then takes first 100), seed 42.
# max_model_length = 34816 >= max prompt (~830 tokens incl. template) + 32768 generation budget.
cd "$(dirname "$0")/../Quantized-Reasoning-Models"
COMMON="--dataset MATH-500 --max_samples 100 --seed 42 --temperature 0.6 --top_p 0.95 --max_new_tokens 32768 --max_model_length 34816"
BF16=./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B
AWQ3=./outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python inference.py --model $BF16 $COMMON --output_dir ./outputs/final/bf16 > ../artifacts/logs/final_bf16.log 2>&1 &
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python inference.py --model $AWQ3 $COMMON --output_dir ./outputs/final/awq3 > ../artifacts/logs/final_awq3.log 2>&1 &
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python inference.py --model $AWQ3 $COMMON --marker_lambda 1.0 --output_dir ./outputs/final/awq3_lambda1.0 > ../artifacts/logs/final_awq3_lambda1.0.log 2>&1 &
wait

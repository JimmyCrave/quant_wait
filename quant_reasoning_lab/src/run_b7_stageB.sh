#!/bin/sh
cd "$(dirname "$0")/.."
sh src/run_b5_stage.sh awq3 dur_study/data/B_awq3.jsonl dur_study/data/outB_awq3 0,1,2,3 &
sh src/run_b5_stage.sh bf16 dur_study/data/B_bf16.jsonl dur_study/data/outB_bf16 4,5,6,7 &
wait

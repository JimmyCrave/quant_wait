#!/bin/sh
# usage: run_phi_shard.sh <gpu> <shard> ; phi-4 labels ALL steps (8 shards) for the consensus (cons) label variant
cd "$(dirname "$0")/.."
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=$1 ../.venv/bin/python src/label_llm.py --model models/phi-4 --steps data/steps_all.jsonl --shard $2 --nshards 8 --out data/labels_phi4_all_sh$2.jsonl > logs/label_phi4_all_sh$2.log 2>&1

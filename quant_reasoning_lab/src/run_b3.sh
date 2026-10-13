#!/bin/sh
# same-prefix continuations: each model rebuilds its own KV (bf16 KV dtype both) from identical token prefixes
cd "$(dirname "$0")/.."
P=../.venv/bin/python
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=4 $P src/continue_gen.py --model bf16 --requests data/b3/requests.jsonl --shard 0 --nshards 2 --out data/b3/out_bf16_0.jsonl > logs/b3_bf16_0.log 2>&1 &
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=5 $P src/continue_gen.py --model bf16 --requests data/b3/requests.jsonl --shard 1 --nshards 2 --out data/b3/out_bf16_1.jsonl > logs/b3_bf16_1.log 2>&1 &
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=6 $P src/continue_gen.py --model awq3 --requests data/b3/requests.jsonl --shard 0 --nshards 2 --out data/b3/out_awq3_0.jsonl > logs/b3_awq3_0.log 2>&1 &
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=7 $P src/continue_gen.py --model awq3 --requests data/b3/requests.jsonl --shard 1 --nshards 2 --out data/b3/out_awq3_1.jsonl > logs/b3_awq3_1.log 2>&1 &
wait

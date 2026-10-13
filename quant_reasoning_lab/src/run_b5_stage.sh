#!/bin/sh
# usage: run_b5_stage.sh <model> <requests> <outprefix> <gpu,gpu,...>   (one process per gpu, shards over requests)
cd "$(dirname "$0")/.."
M=$1; R=$2; O=$3; G=$4; n=$(echo $G | tr ',' '\n' | wc -l); i=0
for g in $(echo $G | tr ',' ' '); do
  env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=$g ../.venv/bin/python src/continue_gen.py --model $M --requests $R --shard $i --nshards $n --out ${O}_$i.jsonl > logs/$(basename $O)_$i.log 2>&1 &
  i=$((i+1))
done
wait

"""Diagnostic: why do same-model same-seed requests diverge after ~2 tokens (B7 audit 5)? Same request (a) alone twice, (b) inside batches of different composition, (c) with different max_tokens."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, f"{os.path.dirname(__file__)}/../../Quantized-Reasoning-Models")
from common import *
from vllm import LLM, SamplingParams
model = sys.argv[1]; A = json.load(open(f"{LAB}/dur_study/data/anchors32.json")); P = [a["prompt_ids"] + a["prefix_ids"] for a in list(A.values())[:16]]
llm = LLM(BF16 if model == "bf16" else AWQ3, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=False)
def run(prompts, seeds, mt): return [list(o.outputs[0].token_ids) for o in llm.generate([{"prompt_token_ids": p} for p in prompts], [SamplingParams(temperature=0.6, top_p=0.95, seed=s, max_tokens=mt) for s in seeds], use_tqdm=False)]
def lcp(x, y): return next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
res = {}
a1 = [run([P[i]], [7], 128)[0] for i in range(4)]; a2 = [run([P[i]], [7], 128)[0] for i in range(4)]
res["alone_vs_alone_same_process_lcp"] = [lcp(x, y) for x, y in zip(a1, a2)]
b = run(P, [7] * 16, 128); res["alone_vs_batch16_lcp"] = [lcp(a1[i], b[i]) for i in range(4)]
b2 = run(P[::-1], [7] * 16, 128)[::-1]; res["batch16_vs_reversed_batch16_lcp"] = [lcp(b[i], b2[i]) for i in range(16)]
c = [run([P[i]], [7], 32)[0] for i in range(4)]; res["alone128_vs_alone32_lcp(max 32)"] = [lcp(a1[i][:32], c[i]) for i in range(4)]
d = run([P[0]] * 8, [7] * 8, 128); res["same prompt 8x identical seed in one batch: n distinct outputs"] = len({tuple(x) for x in d})
d = run([P[0]] * 8, list(range(8)), 128); res["same prompt 8 different seeds: n distinct"] = len({tuple(x) for x in d})
print("RESULT", model, json.dumps(res))

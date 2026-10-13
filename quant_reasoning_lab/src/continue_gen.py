"""Same-prefix continuation runner (B3/B5). Each request = exact prompt token ids (chat prompt + prefix tokens), a model, a seed, a budget.
Runs vLLM 0.7.0 (KV cache dtype bf16/auto for BOTH models; each model rebuilds its own KV from the identical token prefix).
Request jsonl fields: req_id, prompt_token_ids, seed, max_tokens, [lam] (marker penalty, quant only). Output jsonl adds token_ids, text, finish_reason, latency."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, f"{os.path.dirname(__file__)}/../../Quantized-Reasoning-Models")
from common import *

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["bf16", "awq3"]); ap.add_argument("--requests"); ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--max_len", type=int, default=40960)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from penalty.marker_penalty import MarkerPenaltyProcessor, load_marker_token_ids
    mk = load_marker_token_ids(f"{REPO}/penalty/qwen_marker_token_ids.json")
    reqs = [json.loads(l) for l in open(a.requests)]
    reqs = [r for r in reqs if r.get("model", a.model) == a.model]; reqs = [r for i, r in enumerate(reqs) if i % a.nshards == a.shard]
    llm = LLM(BF16 if a.model == "bf16" else AWQ3, dtype="bfloat16", max_model_len=a.max_len, gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=False)
    prompts, sps = [], []
    for r in reqs:
        lp = [MarkerPenaltyProcessor(mk, r["lam"])] if r.get("lam") not in (None, 0) else []
        sps.append(SamplingParams(temperature=0.6, top_p=0.95, seed=r["seed"], max_tokens=r["max_tokens"], logits_processors=lp))
        prompts.append({"prompt_token_ids": r["prompt_token_ids"]})
    t0 = time.time(); outs = llm.generate(prompts, sps); wall = time.time() - t0
    tok = tokenizer()
    with open(a.out, "w") as f:
        for r, o in zip(reqs, outs):
            c = o.outputs[0]; m = o.metrics
            f.write(json.dumps(dict(req_id=r["req_id"], model=a.model, seed=r["seed"], n_prompt=len(r["prompt_token_ids"]), token_ids=list(c.token_ids),
                                    text=c.text, finish_reason=c.finish_reason, n_gen=len(c.token_ids),
                                    latency_s=(m.finished_time - m.arrival_time) if m and m.finished_time else None)) + "\n")
    print("done", len(reqs), f"wall {wall:.1f}s")
if __name__ == "__main__": main()

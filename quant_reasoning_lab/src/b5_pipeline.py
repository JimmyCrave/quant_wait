"""B5 staging. BF16 correction window = BF16 samples W tokens from the shared prefix (identical to replacing quantized logits by BF16 logits for W steps
along the same sampled prefix); the quantized model then continues from prefix+window (it rebuilds KV by prefill here; a deployed system would instead run
a W-token catch-up). Costs are recorded per branch: BF16 prefill tokens = len(prompt+prefix), BF16 decode tokens = window tokens.
usage: b5_pipeline.py stage1|stage2|merge <dir>"""
import sys, os, json, glob, collections
sys.path.insert(0, os.path.dirname(__file__))
cmd, d = sys.argv[1], sys.argv[2]
B = [json.loads(l) for l in open(f"{d}/branches.jsonl")]; MAXTOK = 4096
def prompt(b): return b["prompt_ids"] + b["prefix_ids"]
if cmd == "stage1":
    with open(f"{d}/stage1_awq3.jsonl", "w") as fa, open(f"{d}/stage1_bf16.jsonl", "w") as fb:
        for b in B:
            if b["cond"] in ("C0", "C0r", "C1"):
                fa.write(json.dumps(dict(req_id=b["branch_id"], model="awq3", seed=b["seed"], max_tokens=MAXTOK, lam=(1.0 if b["cond"] == "C1" else None), prompt_token_ids=prompt(b))) + "\n")
            else:
                fb.write(json.dumps(dict(req_id=b["branch_id"], model="bf16", seed=b["seed"], max_tokens=b["W"], prompt_token_ids=prompt(b))) + "\n")
elif cmd == "stage2":
    bm = {b["branch_id"]: b for b in B}; n2 = 0
    with open(f"{d}/stage2_awq3.jsonl", "w") as f:
        for p in sorted(glob.glob(f"{d}/s1_bf16_*.jsonl")):
            for l in open(p):
                r = json.loads(l); b = bm[r["req_id"]]; w = r["token_ids"]
                if r["finish_reason"] == "stop" or len(w) < b["W"]: continue      # window already ended the generation
                f.write(json.dumps(dict(req_id=r["req_id"], model="awq3", seed=b["seed"] + 100, max_tokens=MAXTOK - len(w), lam=None, prompt_token_ids=prompt(b) + w)) + "\n"); n2 += 1
    print("stage2 requests", n2)
elif cmd == "merge":
    bm = {b["branch_id"]: b for b in B}; s1 = {}; s2 = {}; t = {}
    for p in glob.glob(f"{d}/s1_*.jsonl"):
        for l in open(p): r = json.loads(l); s1[r["req_id"]] = r
    for p in glob.glob(f"{d}/s2_awq3_*.jsonl"):
        for l in open(p): r = json.loads(l); s2[r["req_id"]] = r
    with open(f"{d}/final_b5.jsonl", "w") as f:
        for rid, r in s1.items():
            b = bm[rid]
            if rid in s2:   # window by BF16 then quant continuation
                w = r["token_ids"]; c = s2[rid]; out = dict(req_id=rid, model="awq3", seed=b["seed"], token_ids_all=w + c["token_ids"], finish_reason=c["finish_reason"], n_window=len(w),
                                                          bf16_prefill_tokens=len(prompt(b)), bf16_decode_tokens=len(w), lat_bf16=r["latency_s"], lat_quant=c["latency_s"], quant_prefill_tokens=len(prompt(b)) + len(w))
            elif b["cond"] in ("CW", "C5"):   # window ended generation
                w = r["token_ids"]; out = dict(req_id=rid, model="bf16->awq3", seed=b["seed"], token_ids_all=w, finish_reason=r["finish_reason"], n_window=len(w), bf16_prefill_tokens=len(prompt(b)),
                                              bf16_decode_tokens=len(w), lat_bf16=r["latency_s"], lat_quant=0.0, quant_prefill_tokens=0)
            else:
                out = dict(req_id=rid, model="awq3", seed=b["seed"], token_ids_all=r["token_ids"], finish_reason=r["finish_reason"], n_window=0, bf16_prefill_tokens=0, bf16_decode_tokens=0,
                           lat_bf16=0.0, lat_quant=r["latency_s"], quant_prefill_tokens=len(prompt(b)))
            f.write(json.dumps(out) + "\n")
    print("merged", len(s1), "of", len(B))

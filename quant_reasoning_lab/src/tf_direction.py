"""B4 pass 2: for pre-specified windows, accumulate sum over positions of (p_quant - p_bf16) per vocabulary item (raw and sampling distribution), per category.
Only the accumulated vocabulary vector per category is stored (never per-position logits)."""
import sys, os, json, argparse, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from tf_stats import samp_dist, prompt_ids
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); a = ap.parse_args()
from transformers import AutoModelForCausalLM
tok = tokenizer(); dev = "cuda"
mb = AutoModelForCausalLM.from_pretrained(BF16, torch_dtype=torch.bfloat16).to(dev).eval(); mq = AutoModelForCausalLM.from_pretrained(AWQ3, torch_dtype=torch.bfloat16).to(dev).eval()
W = json.load(open(f"{LAB}/data/b4_windows.json")); byt = collections.defaultdict(list)
for w in W: byt[w["traj_id"]].append(w)
trs = {t["traj_id"]: t for t in load_trajectories() if t["traj_id"] in byt}; ids = sorted(byt)[a.shard::a.nshards]
V = mb.lm_head.weight.shape[0]; acc = {}
def A(k):
    if k not in acc: acc[k] = dict(raw=torch.zeros(V, device=dev, dtype=torch.float64), samp=torch.zeros(V, device=dev, dtype=torch.float64), base_raw=torch.zeros(V, device=dev, dtype=torch.float64), n=0)
    return acc[k]
with torch.no_grad():
    for tid in ids:
        tr = trs[tid]; P = prompt_ids(tr["task_id"]); gen = tok(tr["text"], add_special_tokens=False).input_ids
        x = torch.tensor([P + gen], device=dev)
        hb = mb.model(input_ids=x).last_hidden_state[0]; hq = mq.model(input_ids=x).last_hidden_state[0]
        for w in byt[tid]:
            s, e = w["start"], min(w["end"], len(gen)); sl = slice(len(P) - 1 + s, len(P) - 1 + e)
            lb, lq = (hb[sl] @ mb.lm_head.weight.T).float(), (hq[sl] @ mq.lm_head.weight.T).float()
            pb, pq = torch.softmax(lb, -1), torch.softmax(lq, -1); sb, sq = samp_dist(lb), samp_dist(lq)
            k = A(f"{w['origin']}|{w['kind']}"); k["raw"] += (pq - pb).sum(0).double(); k["samp"] += (sq - sb).sum(0).double(); k["base_raw"] += pb.sum(0).double(); k["n"] += e - s
torch.save({k: dict(raw=v["raw"].cpu(), samp=v["samp"].cpu(), base_raw=v["base_raw"].cpu(), n=v["n"]) for k, v in acc.items()}, f"{LAB}/data/b4_direction_sh{a.shard}.pt")
print("done", len(ids))

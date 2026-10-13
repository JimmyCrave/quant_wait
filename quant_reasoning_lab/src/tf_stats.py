"""B4: teacher-forced BF16-vs-AWQ3 next-token distribution statistics along fixed trajectories (streamed, per position scalars;
full-vocabulary logits are never stored). Both models read the SAME token sequence (BF16 KV/activations; AWQ3 = fake-quant weights).
Raw = distribution from raw logits; samp = distribution actually sampled from (temperature 0.6 then top-p 0.95, vLLM order).
Usage: tf_stats.py --steps data/steps_X.jsonl --out_dir data/tf --shard i --nshards n [--windows file]"""
import sys, os, json, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
T_, TOPP = 0.6, 0.95
FEATS = ["H_b", "H_q", "KL_bq", "KL_qb", "lp_real_b", "lp_real_q", "mk_b", "mk_q", "end_b", "end_q", "eos_b", "eos_q",
         "mg_end_b", "mg_end_q", "mg_mk_b", "mg_mk_q", "mg_top_b", "mg_top_q", "sH_b", "sH_q", "sTV", "sKL_bq", "sOut_b", "smk_b", "smk_q", "send_b", "send_q"]

def entropy(lp): return -(lp.exp() * lp).sum(-1)
def samp_dist(logits):
    p = torch.softmax(logits / T_, -1); sp, si = p.sort(-1, descending=True); cs = sp.cumsum(-1)
    keep = (cs - sp) < TOPP; sp = sp * keep; sp = sp / sp.sum(-1, keepdim=True)
    return torch.zeros_like(p).scatter_(-1, si, sp)
def margin_of(logits, idx):
    v = logits[:, idx]; o = logits.clone(); o[:, idx] = -1e30; return v - o.max(-1).values
def mk_margin(logits, mk, nm_mask):
    return logits[:, mk].max(-1).values - logits.masked_fill(~nm_mask, -1e30).max(-1).values

@torch.no_grad()
def stats_chunk(hb, hq, Wb, Wq, real, mk, nm_mask, acc=None, cat_ids=None, ncat=0):
    lb, lq = (hb @ Wb.T).float(), (hq @ Wq.T).float()
    lpb, lpq = torch.log_softmax(lb, -1), torch.log_softmax(lq, -1); pb, pq = lpb.exp(), lpq.exp()
    ar = torch.arange(len(real), device=real.device)
    f = {}
    f["H_b"], f["H_q"] = entropy(lpb), entropy(lpq)
    f["KL_bq"] = (pb * (lpb - lpq)).sum(-1); f["KL_qb"] = (pq * (lpq - lpb)).sum(-1)
    f["lp_real_b"], f["lp_real_q"] = lpb[ar, real], lpq[ar, real]
    f["mk_b"], f["mk_q"] = pb[:, mk].sum(-1), pq[:, mk].sum(-1)
    f["end_b"], f["end_q"] = pb[:, THINK_END], pq[:, THINK_END]; f["eos_b"], f["eos_q"] = pb[:, EOS], pq[:, EOS]
    f["mg_end_b"], f["mg_end_q"] = margin_of(lb, THINK_END), margin_of(lq, THINK_END)
    f["mg_mk_b"], f["mg_mk_q"] = mk_margin(lb, mk, nm_mask), mk_margin(lq, mk, nm_mask)
    t2b, t2q = lb.topk(2, -1).values, lq.topk(2, -1).values; f["mg_top_b"], f["mg_top_q"] = t2b[:, 0] - t2b[:, 1], t2q[:, 0] - t2q[:, 1]
    sb, sq = samp_dist(lb), samp_dist(lq)
    f["sH_b"] = -(sb * torch.log(sb.clamp_min(1e-30))).sum(-1); f["sH_q"] = -(sq * torch.log(sq.clamp_min(1e-30))).sum(-1)
    f["sTV"] = 0.5 * (sb - sq).abs().sum(-1)
    f["sKL_bq"] = (sb * (torch.log(sb.clamp_min(1e-30)) - torch.log(sq.clamp_min(1e-6)))).sum(-1)     # q floored at 1e-6
    f["sOut_b"] = (sb * (sq == 0)).sum(-1)                                                              # b mass outside q's nucleus
    f["smk_b"], f["smk_q"] = sb[:, mk].sum(-1), sq[:, mk].sum(-1); f["send_b"], f["send_q"] = sb[:, THINK_END], sq[:, THINK_END]
    if acc is not None:   # accumulate sum of (p_q - p_b) per category (raw and samp) for directional analysis
        for c in range(ncat):
            m = cat_ids == c
            if m.any(): acc["raw"][c] += (pq[m] - pb[m]).sum(0); acc["samp"][c] += (sq[m] - sb[m]).sum(0); acc["n"][c] += int(m.sum())
    return torch.stack([f[k] for k in FEATS], -1)

def prompt_ids(task_id):
    tok = tokenizer(); q = problems()[task_id]
    s = f"<｜begin▁of▁sentence｜><｜User｜>{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
    return tok(s, add_special_tokens=False).input_ids

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out_dir"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="file with traj_ids (one per line); default all available trajectories")
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    from transformers import AutoModelForCausalLM
    tok = tokenizer(); mk_ids = sorted(set(json.load(open(f"{REPO}/penalty/qwen_marker_token_ids.json"))["token_ids"].values()))
    dev = "cuda"
    mb = AutoModelForCausalLM.from_pretrained(BF16, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    mq = AutoModelForCausalLM.from_pretrained(AWQ3, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    V = mb.lm_head.weight.shape[0]; mk = torch.tensor(mk_ids, device=dev); nm = torch.ones(V, dtype=torch.bool, device=dev); nm[mk] = False
    only = set(open(a.only).read().split()) if a.only else None
    trajs = [t for t in load_trajectories() if t["config"] in ("bf16", "awq3") and (only is None or t["traj_id"] in only)]
    trajs = trajs[a.shard::a.nshards][:a.limit]
    for tr in trajs:
        out = f"{a.out_dir}/{tr['traj_id'].replace('/', '_')}.npz"
        if os.path.exists(out): continue
        t0 = time.time(); P = prompt_ids(tr["task_id"]); gen = tok(tr["text"], add_special_tokens=False).input_ids; G = len(gen)
        x = torch.tensor([P + gen], device=dev); real_all = torch.tensor(gen, device=dev)
        with torch.no_grad():
            hb = mb.model(input_ids=x).last_hidden_state[0, len(P) - 1:len(P) - 1 + G]
            hq = mq.model(input_ids=x).last_hidden_state[0, len(P) - 1:len(P) - 1 + G]
            feats = [stats_chunk(hb[i:i + a.chunk], hq[i:i + a.chunk], mb.lm_head.weight, mq.lm_head.weight, real_all[i:i + a.chunk], mk, nm).half().cpu()
                     for i in range(0, G, a.chunk)]
        np.savez_compressed(out, feats=torch.cat(feats).numpy(), names=np.array(FEATS), ids=np.array(gen, dtype=np.int32))
        print(tr["traj_id"], G, f"{time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__": main()

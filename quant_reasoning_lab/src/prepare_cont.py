"""Turn continuation outputs (B3 / B5) into pseudo-trajectories (prefix text + continuation text), segment them and run the candidate detectors.
usage: prepare_cont.py <kind b3|b5> <dir> <gpu_dev>"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import build_steps
kind, d, dev = sys.argv[1], sys.argv[2], sys.argv[3]
tok = tokenizer(); trs = None
if kind == "b3":
    meta = {b["prefix_id"]: b for b in map(json.loads, open(f"{LAB}/prefix_bank.jsonl"))}
else:
    anchors = {a["anchor_id"]: a for a in map(json.loads, open(f"{d}/anchors.jsonl"))}
    branches = {b["branch_id"]: b for b in map(json.loads, open(f"{d}/branches.jsonl"))}
trajs = []
for p in sorted(glob.glob(f"{d}/final_*.jsonl")):
    for l in open(p):
        r = json.loads(l); rid = r["req_id"]
        if kind == "b3":
            b = meta[rid.split("|")[0]]; pre = b["prefix_ids"]; task = b["task_id"]; gold = b["gold"]; ex = dict(prefix_id=b["prefix_id"], cat=b["cat"], origin=b["origin"], model=r["model"], seed=r["seed"])
        else:
            b = branches[rid]; a = anchors[b["anchor_id"]]; pre = b["prefix_ids"]; task = b["task_id"]; gold = a["gold"]
            ex = dict(branch_id=rid, anchor_id=b["anchor_id"], family=b["family"], cond=b["cond"], W=b["W"], seed=b["seed"], origin=a["origin"], n_window=r.get("n_window", 0),
                      bf16_prefill_tokens=r.get("bf16_prefill_tokens", 0), bf16_decode_tokens=r.get("bf16_decode_tokens", 0))
        ids_full = r["token_ids_all"] if "token_ids_all" in r else r["token_ids"]
        pt = tok.decode(pre)
        variants = [("", ids_full[:4096], r["finish_reason"] if len(ids_full) <= 4096 else "length")]       # primary budget 4096
        if kind == "b3" and meta[rid.split("|")[0]]["heldout_sensitivity8192"] or (kind == "b5" and b.get("budget", 4096) > 4096):
            variants.append(("#8192", ids_full, r["finish_reason"]))                                        # sensitivity budget 8192
        for suf, ids, fin in variants:
            ct = tok.decode(ids); e2 = dict(ex, prefix_chars=len(pt), n_prefix_tokens=len(pre), n_cont_tokens=len(ids), cont_finish=fin, budget=8192 if suf else 4096)
            trajs.append(dict(traj_id=rid + suf, config=e2.get("model", e2.get("cond")), seed=e2["seed"], task_id=task, split=None, score=None, finish_reason=fin, n_gen=len(ids), gold=gold, text=pt + ct, extra=e2))
print("pseudo-trajectories:", len(trajs))
build_steps.run(trajs, f"{d}/cont_steps.jsonl", dev)

"""Per-run outcome table for B7 (token-id based endpoints + detector-based repetition flags)."""
import sys, os, json, bisect
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from outcomes import outcomes
from common import *
import re
tok = tokenizer(); BOXRE = re.compile(r"\\boxed\s*\{")

def closed_boxes(text):
    """[(start, end_exclusive, content)] of balanced \\boxed{...}"""
    out = []
    for m in BOXRE.finditer(text):
        i, d = m.end(), 1
        while i < len(text) and d:
            d += (text[i] == "{") - (text[i] == "}"); i += 1
        if d == 0: out.append((m.start(), i, text[m.end():i - 1]))
    return out

def answer_info(ids, gold):
    """think_end_pos / answer_complete_pos are 1-based counts of continuation tokens (inclusive of the token that closes </think> / the \\boxed{...})."""
    ids = list(ids); eos = ids.index(EOS) if EOS in ids else None
    body = ids[:eos] if eos is not None else ids
    r = dict(eos_in_budget=eos is not None, eos_pos=(eos + 1) if eos is not None else None, think_closed=THINK_END in body, think_end_pos=None, answer_complete=False, answer_complete_pos=None, final_correct=0, first_boxed_correct=0, n_boxes=0)
    if not r["think_closed"]: return r
    jt = body.index(THINK_END); r["think_end_pos"] = jt + 1; post = body[jt + 1:]; text = tok.decode(post); bx = closed_boxes(text)
    r["n_boxes"] = len(bx)
    if not bx: return r
    r["answer_complete"] = True
    lo, hi = 0, len(post)                                  # smallest k with decode(post[:k]) covering the first closed box
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(post[:mid])) >= bx[0][1]: hi = mid
        else: lo = mid + 1
    r["answer_complete_pos"] = jt + 1 + lo
    r["final_correct"] = int(grade("\\boxed{" + bx[-1][2] + "}", gold) == 1); r["first_boxed_correct"] = int(grade("\\boxed{" + bx[0][2] + "}", gold) == 1)
    return r

def build_table():
    anchors = json.load(open(f"{LAB}/dur_study/data/anchors32.json")); steps = {}
    for l in open(f"{LAB}/dur_study/data/b7_steps.jsonl"): r = json.loads(l); steps[r["traj_id"]] = r
    rows = []
    for l in open(f"{LAB}/dur_study/data/final_b7.jsonl"):
        r = json.loads(l); a = anchors[r["anchor_id"]]; ids = r["token_ids_all"]; n = len(ids)
        info = answer_info(ids, a["gold"]); o = outcomes(steps[r["run_id"]]); W = r["n_window"]
        fl = o["first_loop_tok"]; ac = info["answer_complete_pos"]; primary = int(info["answer_complete"] and info["final_correct"] == 1)
        if r["W"] > 0:
            if r["window_ended_generation"]: cls = "window_ended_generation"
            elif info["answer_complete"] and ac <= W: cls = "complete_in_window"
            elif info["answer_complete"]: cls = "complete_after_switch"
            elif o["det_entry"] and fl is not None and fl >= W - 1: cls = "repeat_after_switch_incomplete"
            elif o["det_entry"]: cls = "repeat_in_window_incomplete"
            else: cls = "incomplete_other"
        else: cls = "complete" if info["answer_complete"] else ("repeat_incomplete" if o["det_entry"] else "incomplete_other")
        fw = r["fwd"]; tot_sim = sum(v["prefill"] + v["decode"] for v in fw.values())
        rows.append(dict(run_id=r["run_id"], group=r["group"], anchor_id=r["anchor_id"], task_id=a["task_id"], arm=r["arm"], seed=r["seed"], T=r["T"], W=r["W"], n_total=n, n_window=W, finish=r["finish_reason"], window_ended_generation=r["window_ended_generation"],
                         primary=primary, **info, det_entry=int(o["det_entry"]), first_loop_tok=fl, loop_in_window=int(bool(o["det_entry"]) and fl is not None and fl < max(W, 1) - 1 and W > 0), tail_in_loop=int(o["tail_in_loop"]),
                         len_to_complete=(ac if primary or info["answer_complete"] else r["T"]), cls=cls, prefix_len=r["prefix_len"],
                         bf16_prefill=fw.get("bf16", {}).get("prefill", 0), bf16_decode=fw.get("bf16", {}).get("decode", 0), awq3_prefill_sim=fw.get("awq3", {}).get("prefill", 0), awq3_decode=fw.get("awq3", {}).get("decode", 0),
                         fwd_tokens_total_sim=tot_sim, lat_stageA=r["lat"]["stageA"], lat_stageB=r["lat"]["stageB"]))
    D = pd.DataFrame(rows); D.to_csv(f"{LAB}/dur_study/data/b7_outcomes.csv", index=False); return D
if __name__ == "__main__": D = build_table(); print(D.groupby(["group", "arm"]).agg(n=("run_id", "count"), primary=("primary", "mean"), closed=("think_closed", "mean"), det=("det_entry", "mean")).round(3).to_string())

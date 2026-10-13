"""Audit 4: is the BF16 window inside the 4096 new-token budget of the stage-2 (B5) interventions? Verified from the raw request/response files, not from code reading."""
import json, glob, collections
import numpy as np, pandas as pd
B = {b["branch_id"]: b for b in map(json.loads, open("data/b5/branches.jsonl"))}
s2req = {r["req_id"]: r for r in map(json.loads, open("data/b5/stage2_awq3.jsonl"))}
s1 = {}
for p in glob.glob("data/b5/s1_*.jsonl"):
    for l in open(p): r = json.loads(l); s1[r["req_id"]] = r
s2 = {}
for p in glob.glob("data/b5/s2_awq3_*.jsonl"):
    for l in open(p): r = json.loads(l); s2[r["req_id"]] = r
F = {r["req_id"]: r for r in map(json.loads, open("data/b5/final_b5.jsonl"))}
rows = []; viol = 0
for rid, f in F.items():
    b = B[rid]; w = f["n_window"]; tot = len(f["token_ids_all"])
    if rid in s2:
        chk = (s2req[rid]["max_tokens"] == 4096 - w) and (len(s2[rid]["token_ids"]) <= 4096 - w) and tot == w + len(s2[rid]["token_ids"])
        concat_ok = s2req[rid]["prompt_token_ids"] == b["prompt_ids"] + b["prefix_ids"] + s1[rid]["token_ids"]
    else:
        chk = tot <= 4096; concat_ok = True
    viol += (not chk) + (not concat_ok)
    rows.append(dict(branch_id=rid, family=b["family"], cond=b["cond"], W=b["W"], n_window=w, total_new_tokens=tot, hit_cap=tot >= 4096, window_ended_generation=(rid not in s2 and b["cond"] in ("CW", "C5")),
                     budget_rule_ok=chk, concat_exact_ok=concat_ok, bf16_prefill=f["bf16_prefill_tokens"], bf16_decode=f["bf16_decode_tokens"], quant_prefill_sim=f["quant_prefill_tokens"], quant_decode=tot - w if f["model"] != "bf16->awq3" else 0))
D = pd.DataFrame(rows); D.to_csv("dur_study/audit/audit4_budget_per_branch.csv", index=False)
S = D.groupby(["cond", "W"]).agg(n=("branch_id", "count"), mean_total_new=("total_new_tokens", "mean"), max_total_new=("total_new_tokens", "max"), frac_hit_cap=("hit_cap", "mean"), window_ended_gen=("window_ended_generation", "sum"),
                                 rule_ok=("budget_rule_ok", "all"), concat_ok=("concat_exact_ok", "all"), mean_bf16_decode=("bf16_decode", "mean"), mean_quant_decode=("quant_decode", "mean")).reset_index()
S.to_csv("dur_study/audit/audit4_budget_summary.csv", index=False); pd.set_option("display.width", 220); print(S.round(1).to_string()); print("violations:", viol)

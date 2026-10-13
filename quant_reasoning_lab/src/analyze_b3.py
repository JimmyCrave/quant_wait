"""B3: same-prefix BF16 vs AWQ3 continuation, paired by prefix, 3 seeds averaged within prefix, bootstrap clustered by question."""
import json, sys, os, collections, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from outcomes import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{LAB}/counterfactual_results.csv"); a = ap.parse_args()
trs = {t["traj_id"]: t for t in load_trajectories()} if False else None
rows = {}
texts = {}
for l in open(f"{LAB}/data/b3/cont_steps.jsonl"):
    r = json.loads(l); ex = r["extra"]
    rid = r["traj_id"]; base = rid.replace("#8192", "")
    ex["budget"] = 8192 if rid.endswith("#8192") else 4096
    o = outcomes(r)
    rows[rid] = dict(req_id=rid, prefix_id=ex["prefix_id"], cat=ex["cat"], prefix_origin=ex["origin"], model=ex["model"], seed=ex["seed"], budget=ex["budget"], task_id=r["task_id"], **o)
    texts[rid] = r
# final-answer grading needs the text: reconstruct from steps (join) is lossy -> grade from stored continuation text
bank = {b["prefix_id"]: b for b in map(json.loads, open(f"{LAB}/prefix_bank.jsonl"))}
tok = tokenizer(); outs = {}
for p in os.listdir(f"{LAB}/data/b3"):
    if p.startswith("out_"):
        for l in open(f"{LAB}/data/b3/{p}"): r = json.loads(l); outs[r["req_id"]] = r
for rid, row in rows.items():
    base = rid.replace("#8192", ""); o = outs[base]; ids = o["token_ids"][: (8192 if rid.endswith("#8192") else 4096)]
    b = bank[row["prefix_id"]]; txt = tok.decode(b["prefix_ids"]) + tok.decode(ids)
    fake = dict(extra=dict(prefix_chars=len(tok.decode(b["prefix_ids"]))), _text=txt, gold=b["gold"])
    fin, anyc, wrongp = grade_final(fake); row.update(correct_final=fin, any_correct_prop=anyc, wrong_prop=wrongp, correct_by_end=bool(row["submitted"] and fin == 1))
D = pd.DataFrame(rows.values()); D.to_csv(f"{LAB}/data/b3/continuation_outcomes.csv", index=False)
METRICS = {"G": ["det_entry", "rep_rate", "submitted", "correct_by_end", "any_correct_prop", "n_think_steps"],
           "L": ["escaped", "progress_run", "det_entry", "rep_rate", "submitted", "correct_by_end", "any_correct_prop"],
           "CA": ["submitted", "correct_by_end", "t_sub_rmst", "wrong_prop", "rep_rate", "det_entry"]}
D["t_sub_rmst"] = np.where(D.submitted, D.t_sub, D.budget).astype(float)
rng = np.random.default_rng(0); out = []
for budget in (4096, 8192):
    for cat in ("G", "L", "CA"):
        for origin in ("all", "bf16", "awq3"):
            d = D[(D.cat == cat) & (D.budget == budget)]
            if budget == 8192: d = d[d.prefix_id.isin(D[D.budget == 8192].prefix_id)]
            if origin != "all": d = d[d.prefix_origin == origin]
            if d.prefix_id.nunique() < 3: continue
            for m in METRICS[cat]:
                g = d.groupby(["prefix_id", "task_id", "model"])[m].mean().unstack("model").dropna()
                if len(g) == 0 or not {"bf16", "awq3"} <= set(g.columns): continue
                g = g.reset_index(); diff = g.awq3 - g.bf16; per_q = diff.groupby(g.task_id).mean(); qs = per_q.index.values
                bs = [per_q.loc[rng.choice(qs, len(qs))].mean() for _ in range(2000)]
                out.append(dict(category=cat, prefix_origin=origin, budget=budget, metric=m, mean_bf16=g.bf16.mean(), mean_awq3=g.awq3.mean(), diff_awq3_minus_bf16=per_q.mean(),
                                ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n_prefixes=len(g), n_questions=len(qs), n_continuations=len(d), frac_unfinished_bf16=float((~d[d.model == "bf16"].submitted).mean()), frac_unfinished_awq3=float((~d[d.model == "awq3"].submitted).mean())))
R = pd.DataFrame(out); R.to_csv(a.out, index=False); pd.set_option("display.width", 220)
print(R[(R.prefix_origin == "all")].round(3).to_string())

# ---- descriptive (post-treatment conditioning => CORRELATIONAL, not same-prefix causal): submission given loop status, CA/G prefixes, budget 4096
rng2 = np.random.default_rng(1); ex = []
for cat in ("CA", "G", "L"):
    d = D[(D.cat == cat) & (D.budget == 4096)]
    for loop in (False, True):
        for model in ("bf16", "awq3"):
            s = d[(d.model == model) & (d.det_entry == loop)]
            if len(s) < 5: continue
            qs = s.task_id.unique(); bs = []
            for _ in range(1000):
                pick = rng2.choice(qs, len(qs)); ss = pd.concat([s[s.task_id == q] for q in pick]); bs.append(ss.submitted.mean())
            ex.append(dict(category=cat, det_loop_in_continuation=loop, model=model, n_continuations=len(s), n_questions=len(qs), p_submitted_by_4096=s.submitted.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), p_correct_by_end=s.correct_by_end.mean()))
E = pd.DataFrame(ex); E.to_csv(f"{LAB}/results/b3_submission_by_loop_status.csv", index=False); print(E.round(3).to_string())

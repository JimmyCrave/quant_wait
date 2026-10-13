"""B5 analysis. Effects are within-anchor, same-prefix differences (3 seeds averaged), bootstrap clustered by question (one anchor per question per family).
 effect_pos  = CW - C0       (BF16 window at the oracle position vs no intervention from the SAME prefix)
 effect_pen  = C1 - C0       (marker penalty lambda=1 over the whole continuation, same prefix)
 effect_rand = C5 - C0r      (BF16 window at a randomly placed, entropy-matched position vs no intervention from ITS prefix)
 contrast    = effect_pos - effect_rand   (is position better than an equal-budget random window?)"""
import json, sys, os, collections, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from outcomes import *
d = f"{LAB}/data/b5"; tok = tokenizer()
anchors = {a["anchor_id"]: a for a in map(json.loads, open(f"{d}/anchors.jsonl"))}; branches = {b["branch_id"]: b for b in map(json.loads, open(f"{d}/branches.jsonl"))}
final = {r["req_id"]: r for r in map(json.loads, open(f"{d}/final_b5.jsonl"))}
rows = []
for l in open(f"{d}/cont_steps.jsonl"):
    r = json.loads(l); ex = r["extra"]
    if r["traj_id"].endswith("#8192"): continue
    ex["budget"] = 4096; o = outcomes(r); b = branches[ex["branch_id"]]; a = anchors[b["anchor_id"]]
    ids = final[ex["branch_id"]]["token_ids_all"][:4096]; txt = tok.decode(b["prefix_ids"]) + tok.decode(ids)
    fin, anyc, wrongp = grade_final(dict(extra=dict(prefix_chars=ex["prefix_chars"]), _text=txt, gold=a["gold"]))
    f = final[ex["branch_id"]]
    rows.append(dict(branch_id=ex["branch_id"], anchor_id=b["anchor_id"], family=b["family"], cond=b["cond"], W=b["W"], seed=b["seed"], task_id=b["task_id"], split=b["split"], origin=a["origin"],
                     correct_by_end=bool(o["submitted"] and fin == 1), any_correct_prop=anyc, wrong_prop=wrongp, t_sub_rmst=float(o["t_sub"] if o["submitted"] else 4096), **o,
                     n_window=f["n_window"], bf16_prefill=f["bf16_prefill_tokens"], bf16_decode=f["bf16_decode_tokens"], quant_prefill=f["quant_prefill_tokens"], lat_bf16=f["lat_bf16"], lat_quant=f["lat_quant"], n_prefix=len(b["prefix_ids"])))
D = pd.DataFrame(rows); D.to_csv(f"{d}/branch_outcomes.csv", index=False)
FAM_METRICS = {"E": ["det_entry", "rep_rate", "submitted", "correct_by_end"], "I": ["escaped", "progress_run", "rep_rate", "submitted", "correct_by_end"],
               "A": ["submitted", "correct_by_end", "t_sub_rmst", "wrong_prop", "det_entry"], "V": ["correct_by_end", "any_correct_prop", "submitted", "wrong_prop", "det_entry"]}
rng = np.random.default_rng(0); out = []
def per_anchor(df, cond, W, m): return df[(df.cond == cond) & (df.W == W)].groupby("anchor_id")[m].mean()
def boot(x, q):   # x: Series indexed by anchor_id, q: map anchor->question
    g = x.groupby(q.reindex(x.index).values).mean(); qs = g.index.values
    bs = [g.loc[rng.choice(qs, len(qs))].mean() for _ in range(2000)]; return g.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(qs)
for fam in ("E", "I", "A", "V"):
    for W in (32, 8, 64):
        df = D[D.family == fam]
        if W != 32: df = df[df.split == "dev"]
        if fam == "V" and W != 32: continue
        qmap = df.drop_duplicates("anchor_id").set_index("anchor_id").task_id
        for m in FAM_METRICS[fam]:
            c0 = per_anchor(df, "C0", 0, m); cw = per_anchor(df, "CW", W, m); c1 = per_anchor(df, "C1", 0, m); c0r = per_anchor(df, "C0r", 0, m); c5 = per_anchor(df, "C5", W, m)
            idx = c0.index.intersection(cw.index); idr = c0r.index.intersection(c5.index); both = idx.intersection(idr)
            res = {"effect_pos(CW-C0)": (cw.loc[idx] - c0.loc[idx]), "effect_pen(C1-C0)": (c1.loc[c0.index.intersection(c1.index)] - c0.loc[c0.index.intersection(c1.index)]),
                   "effect_rand(C5-C0r)": (c5.loc[idr] - c0r.loc[idr]), "contrast_pos_minus_rand": ((cw.loc[both] - c0.loc[both]) - (c5.loc[both] - c0r.loc[both]))}
            for name, x in res.items():
                if len(x) < 5: continue
                est, lo, hi, nq = boot(x, qmap); out.append(dict(family=fam, W=W, metric=m, comparison=name, estimate=est, ci_lo=lo, ci_hi=hi, n_anchors=len(x), n_questions=nq,
                                                               mean_C0=c0.mean(), mean_CW=cw.mean(), mean_C1=c1.mean(), mean_C0r=c0r.mean(), mean_C5=c5.mean() if len(c5) else np.nan))
R = pd.DataFrame(out); R.to_csv(f"{LAB}/intervention_results.csv", index=False)
# cost table
cw = D[(D.cond == "CW")].groupby(["family", "W"]).agg(n=("branch_id", "count"), bf16_prefill_tokens=("bf16_prefill", "mean"), bf16_decode_tokens=("bf16_decode", "mean"), quant_prefill_tokens_sim=("quant_prefill", "mean"),
                                                         mean_prefix_tokens=("n_prefix", "mean"), mean_cont_tokens=("n_cont", "mean"), lat_bf16_s=("lat_bf16", "mean")).reset_index()
cw["bf16_forward_tokens_total"] = cw.bf16_prefill_tokens + cw.bf16_decode_tokens
cw["bf16_extra_as_frac_of_cont_tokens"] = cw.bf16_forward_tokens_total / cw.mean_cont_tokens
cw["bf16_kv_MB_at_window_start"] = cw.mean_prefix_tokens * 28672 / 1e6   # Qwen2-1.5B: 28 layers x 2(K,V) x 2 kv-heads x 128 x 2 bytes
cw.to_csv(f"{LAB}/results/intervention_cost.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 400)
print(R[(R.W == 32)].round(3).to_string()); print(cw.round(1).to_string())
# figure: effect vs random per family/metric
key = {"E": "det_entry", "I": "escaped", "A": "submitted", "V": "correct_by_end"}
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for k, (fam, m) in enumerate(key.items()):
    s = R[(R.family == fam) & (R.W == 32) & (R.metric == m)]
    names = ["effect_pos(CW-C0)", "effect_rand(C5-C0r)", "effect_pen(C1-C0)", "contrast_pos_minus_rand"]; xs = np.arange(len(names))
    est = [s[s.comparison == n].estimate.values[0] if (s.comparison == n).any() else np.nan for n in names]
    lo = [s[s.comparison == n].ci_lo.values[0] if (s.comparison == n).any() else np.nan for n in names]; hi = [s[s.comparison == n].ci_hi.values[0] if (s.comparison == n).any() else np.nan for n in names]
    ax[k].bar(xs, est, yerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)], color=["C3", "C7", "C2", "C1"]); ax[k].axhline(0, color="k", lw=.6)
    ax[k].set_xticks(xs); ax[k].set_xticklabels(["BF16 window\n(oracle pos)", "BF16 window\n(random)", "marker pen.\nlambda=1", "pos - random"], fontsize=7)
    ax[k].set_title(f"family {fam}: change in P({m}) vs same-prefix C0", fontsize=8)
fig.tight_layout(); fig.savefig(f"{LAB}/figs/intervention_effects.png", dpi=130)
fig, ax = plt.subplots(figsize=(5.5, 4))
for fam, c in zip("EIAV", ["C0", "C1", "C2", "C3"]):
    s = R[(R.family == fam) & (R.comparison == "effect_pos(CW-C0)") & (R.metric == key[fam])]
    cc = cw[cw.family == fam]
    for _, r in s.iterrows():
        ccr = cc[cc.W == r.W]
        if len(ccr): ax.errorbar(float(ccr.bf16_forward_tokens_total.iloc[0]), r.estimate, yerr=[[r.estimate - r.ci_lo], [r.ci_hi - r.estimate]], fmt="o", color=c, label=f"{fam} W={int(r.W)}")
ax.set_xscale("log"); ax.set_xlabel("BF16 forward tokens per intervention (prefill+window)"); ax.set_ylabel("effect on primary metric"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{LAB}/figs/intervention_cost.png", dpi=130)

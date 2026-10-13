"""B7 main analysis: correction duration vs final submission. Paired differences vs D0 (AWQ3 all the way), anchors are the cluster (one anchor per question), seeds averaged within anchor."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DD = f"{LAB}/dur_study/data"; OUT = f"{LAB}/dur_study"; rng = np.random.default_rng(0)
O = pd.read_csv(f"{DD}/b7_outcomes.csv"); O["len_to_complete"] = O.len_to_complete.astype(float)
METR = ["primary", "think_closed", "answer_complete", "det_entry", "len_to_complete"]; ARMS = ["DP", "DB", "W32", "W256", "W1024"]; ORDER = ["D0", "W32", "W256", "W1024", "DB", "DP"]
def table(g):
    d = O[O.group == g]; pa = {a: d[d.arm == a].groupby("anchor_id")[METR].mean() for a in ORDER}; rows = []
    for a in ARMS:
        x = pa[a] - pa["D0"]; x = x.dropna(); idx = x.index.values
        for m in METR:
            bs = [x[m].loc[rng.choice(idx, len(idx))].mean() for _ in range(5000)]
            rows.append(dict(group=g, arm=a, metric=m, mean_arm=pa[a][m].mean(), mean_D0=pa["D0"][m].mean(), diff_vs_D0=x[m].mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n_anchors=len(idx), n_runs_arm=int((d.arm == a).sum())))
    for a, b in (("W1024", "W32"), ("W1024", "W256"), ("W256", "W32"), ("DB", "W1024")):        # monotonicity checks
        x = (pa[a] - pa[b]).dropna(); idx = x.index.values
        for m in METR:
            bs = [x[m].loc[rng.choice(idx, len(idx))].mean() for _ in range(5000)]
            rows.append(dict(group=g, arm=f"{a}-{b}", metric=m, mean_arm=pa[a][m].mean(), mean_D0=pa[b][m].mean(), diff_vs_D0=x[m].mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n_anchors=len(idx), n_runs_arm=int((d.arm == a).sum())))
    return pd.DataFrame(rows), pa
R4, pa4 = table("main"); RL, _ = table("long"); R = pd.concat([R4, RL]); R.to_csv(f"{OUT}/dur_results.csv", index=False)
# decomposition
m = O[O.group == "main"]; dec = m.groupby(["arm", "cls"]).agg(n=("run_id", "count"), primary_correct=("primary", "sum")).reset_index(); dec["frac_of_arm_runs"] = dec.n / dec.groupby("arm").n.transform("sum")
dec["n_arm_runs"] = dec.groupby("arm").n.transform("sum"); dec.to_csv(f"{OUT}/dur_decomposition.csv", index=False)
ca = m[m.cls.isin(["complete_in_window", "complete_after_switch"])].groupby(["arm", "cls"]).agg(n=("run_id", "count"), correct=("final_correct", "sum"), mean_pos=("answer_complete_pos", "mean")).reset_index(); ca.to_csv(f"{OUT}/dur_completion_split.csv", index=False)
# DESCRIPTIVE ONLY (selection bias): runs whose answer was not completed inside the BF16 window
sel = m[(m.W > 0) & ~((m.cls == "complete_in_window") | (m.cls == "window_ended_generation"))]
cond = sel.groupby("arm").agg(n_runs_not_completed_in_window=("run_id", "count"), complete_after_switch=("answer_complete", "mean"), primary_after_switch=("primary", "mean"), repeat_after_switch=("cls", lambda s: (s == "repeat_after_switch_incomplete").mean())).reset_index()
d0 = m[m.arm == "D0"]; cond.loc[len(cond)] = dict(arm="D0 (all runs, reference)", n_runs_not_completed_in_window=len(d0), complete_after_switch=d0.answer_complete.mean(), primary_after_switch=d0.primary.mean(), repeat_after_switch=(d0.cls == "repeat_incomplete").mean())
cond.to_csv(f"{OUT}/dur_conditional_DESCRIPTIVE_selection_biased.csv", index=False)
# cost
cost = m.groupby("arm").agg(n=("run_id", "count"), mean_new_tokens=("n_total", "mean"), bf16_prefill=("bf16_prefill", "mean"), bf16_decode=("bf16_decode", "mean"), awq3_prefill_sim=("awq3_prefill_sim", "mean"), awq3_decode=("awq3_decode", "mean"),
                            fwd_total_sim=("fwd_tokens_total_sim", "mean"), mean_window=("n_window", "mean"), prefix_len=("prefix_len", "mean"), lat_stageA_s=("lat_stageA", "mean"), lat_stageB_s=("lat_stageB", "mean")).reindex(ORDER).reset_index()
cost["awq3_catchup_deployed"] = np.where(cost.arm.isin(["W32", "W256", "W1024"]), cost.mean_window, 0.0)
cost["extra_forward_tokens_deployed_vs_D0"] = np.where(cost.arm.isin(["W32", "W256", "W1024"]), cost.bf16_prefill + cost.bf16_decode + cost.awq3_catchup_deployed - cost.mean_window, np.where(cost.arm == "DB", cost.bf16_prefill + cost.bf16_decode - (cost.loc[cost.arm == "D0", "awq3_prefill_sim"].iloc[0] + cost.loc[cost.arm == "D0", "awq3_decode"].iloc[0]), 0.0))
cost.to_csv(f"{OUT}/dur_cost.csv", index=False)
# figures
fig, ax = plt.subplots(1, 4, figsize=(19, 3.9)); xs = {"D0": 0.5, "W32": 32, "W256": 256, "W1024": 1024}
for k, (met, ttl) in enumerate((("primary", "final answer complete & correct within 4096"), ("think_closed", "</think> emitted"), ("det_entry", "verbatim repetition detected"), ("len_to_complete", "tokens until answer complete (cap 4096)"))):
    t = R4[(R4.metric == met)]; pts = [(0.5, pa4["D0"][met].mean(), pa4["D0"][met].mean(), pa4["D0"][met].mean())]
    for a in ("W32", "W256", "W1024"):
        r = t[t.arm == a].iloc[0]; pts.append((xs[a], r.mean_arm, r.mean_arm - r.diff_vs_D0 + r.ci_lo, r.mean_arm - r.diff_vs_D0 + r.ci_hi))
    ax[k].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="C3", label="BF16 window then AWQ3"); ax[k].fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts], alpha=0.0)
    for a in ("W32", "W256", "W1024"):                                    # CI of the PAIRED difference, drawn around the arm mean
        r = t[t.arm == a].iloc[0]; ax[k].errorbar(xs[a], r.mean_arm, yerr=[[r.diff_vs_D0 - r.ci_lo], [r.ci_hi - r.diff_vs_D0]], color="C3", capsize=3, fmt="none")
    for a, c, lab in (("DB", "C0", "remaining all BF16"), ("DP", "C2", "AWQ3 + marker penalty")):
        r = t[t.arm == a].iloc[0]; ax[k].axhline(r.mean_arm, color=c, ls="--", lw=1, label=lab)
    ax[k].axhline(pa4["D0"][met].mean(), color="k", lw=0.8, label="AWQ3 only (D0)"); ax[k].set_xscale("log"); ax[k].set_xlim(0.3, 2000); ax[k].set_xticks([0.5, 32, 256, 1024]); ax[k].set_xticklabels(["0 (D0)", "32", "256", "1024"]); ax[k].set_xlabel("BF16 window length (tokens)"); ax[k].set_title(ttl, fontsize=8)
    if k == 0: ax[k].legend(fontsize=6)
fig.suptitle("Correction duration vs outcome (32 anchors x 3 seeds; bars = 95% CI of the paired difference vs D0, clustered by question)", fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/figs/duration_curve.png", dpi=130)
fig, ax = plt.subplots(figsize=(8.5, 4)); cls_order = ["complete_in_window", "complete", "complete_after_switch", "repeat_after_switch_incomplete", "repeat_in_window_incomplete", "repeat_incomplete", "incomplete_other", "window_ended_generation"]
bottom = np.zeros(len(ORDER))
for c, col in zip(cls_order, ["C2", "C2", "C9", "C3", "C1", "C3", "C7", "C4"]):
    v = np.array([dec[(dec.arm == a) & (dec.cls == c)].frac_of_arm_runs.sum() for a in ORDER]); 
    if v.sum() > 0: ax.bar(ORDER, v, bottom=bottom, label=c, color=col, alpha=0.9 if c in ("complete_in_window", "complete") else 0.6); bottom += v
ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1)); ax.set_ylabel("fraction of runs"); ax.set_title("Where the answer gets completed (main set, T=4096)"); fig.tight_layout(); fig.savefig(f"{OUT}/figs/duration_decomposition.png", dpi=130)
pd.set_option("display.width", 250, "display.max_rows", 300)
print(R4[R4.metric.isin(["primary", "think_closed", "det_entry", "len_to_complete"])].round(3).to_string()); print(dec.round(3).to_string()); print(ca.round(2).to_string()); print(cond.round(3).to_string()); print(cost.round(1).to_string())
print("LONG subset"); print(RL[RL.metric == "primary"].round(3).to_string())

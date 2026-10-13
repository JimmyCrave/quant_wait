"""B7 extra DESCRIPTIVE analyses (selection-biased by construction, labelled as such): (1) survival-matched 'not yet completed after k tokens' comparison between arms,
(2) in-window vs after-switch share of the gain, (3) anchor heterogeneity, (4) effect-vs-cost figure."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); OUT = f"{LAB}/dur_study"; rng = np.random.default_rng(1)
O = pd.read_csv(f"{OUT}/data/b7_outcomes.csv"); M = O[O.group == "main"].copy(); M["ac"] = M.answer_complete_pos.fillna(1e9)
def cond_mask(d, arm, k):
    if arm.startswith("W"): return (d.arm == arm) & (d.cls != "window_ended_generation") & (d.cls != "complete_in_window")     # answer not finished (and generation not ended) inside the window
    return (d.arm == arm) & (d.ac > k)                                                                                       # not completed within the first k tokens
rows = []
anchors = M.anchor_id.unique()
for k, wa in ((32, "W32"), (256, "W256"), (1024, "W1024")):
    for arm in ("D0", "DP", "DB", wa):
        def stat(d):
            c = d[cond_mask(d, arm, k)]; return len(c), c.primary.sum(), c.det_entry.sum(), c.think_closed.sum()
        n, p, r, t = stat(M); bs = []
        for _ in range(3000):
            pick = rng.choice(anchors, len(anchors)); d = pd.concat([M[M.anchor_id == a] for a in pick]); nn, pp, rr, tt = stat(d)
            bs.append((pp / nn if nn else np.nan, rr / nn if nn else np.nan))
        bs = np.array(bs)
        rows.append(dict(k_tokens=k, arm=arm if arm != wa else f"{wa} (after switch)", n_runs_in_condition=n, n_anchors=M[cond_mask(M, arm, k)].anchor_id.nunique(), P_primary_given_cond=p / n, ci_lo=np.nanpercentile(bs[:, 0], 2.5), ci_hi=np.nanpercentile(bs[:, 0], 97.5),
                         P_repeat_given_cond=r / n, rep_ci_lo=np.nanpercentile(bs[:, 1], 2.5), rep_ci_hi=np.nanpercentile(bs[:, 1], 97.5)))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/dur_survival_matched_DESCRIPTIVE.csv", index=False)
# (2) shares for W1024 / W256 / W32 relative to D0
pa = M.groupby(["anchor_id", "arm"]).agg(primary=("primary", "mean"), in_win=("cls", lambda s: (s.isin(["window_ended_generation", "complete_in_window"])).mean()), after=("cls", lambda s: (s == "complete_after_switch").mean())).reset_index()
sh = []
d0 = pa[pa.arm == "D0"].set_index("anchor_id")
for a in ("W32", "W256", "W1024"):
    x = pa[pa.arm == a].set_index("anchor_id"); idx = x.index
    for name, v in (("total_primary_gain", x.primary - d0.primary.loc[idx]), ("share_completed_inside_window(BF16 only)", x.in_win), ("completed_after_switch_minus_D0_primary", x.after - d0.primary.loc[idx])):
        bs = [v.loc[rng.choice(idx, len(idx))].mean() for _ in range(5000)]; sh.append(dict(arm=a, quantity=name, estimate=v.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
S = pd.DataFrame(sh); S.to_csv(f"{OUT}/dur_gain_split_inwindow_vs_afterswitch.csv", index=False)
# (3) heterogeneity
het = []
P = pa.pivot(index="anchor_id", columns="arm", values="primary")
for a in ("W32", "W256", "W1024", "DB", "DP"):
    d = P[a] - P["D0"]; het.append(dict(arm=a, n_anchors=len(d), n_better=int((d > 0.01).sum()), n_worse=int((d < -0.01).sum()), n_same=int((d.abs() <= 0.01).sum()), D0_zero_anchors=int((P["D0"] == 0).sum()), arm_zero_anchors=int((P[a] == 0).sum())))
H = pd.DataFrame(het); H.to_csv(f"{OUT}/dur_anchor_heterogeneity.csv", index=False)
# DB completion-time profile
db = M[M.arm == "DB"]; prof = {f"DB_complete_by_{k}": float((db.ac <= k).mean()) for k in (32, 256, 1024, 2048, 4096)}; d0r = M[M.arm == "D0"]; prof.update({f"D0_complete_by_{k}": float((d0r.ac <= k).mean()) for k in (32, 256, 1024, 2048, 4096)})
pd.Series(prof).to_csv(f"{OUT}/dur_completion_time_profile.csv", header=["fraction_of_runs"])
# (4) effect vs cost figure
R = pd.read_csv(f"{OUT}/dur_results.csv"); R = R[(R.group == "main") & (R.metric == "primary") & R.arm.isin(["W32", "W256", "W1024", "DB", "DP"])]; cost = pd.read_csv(f"{OUT}/dur_cost.csv").set_index("arm")
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for _, r in R.iterrows():
    x = max(cost.loc[r.arm, "extra_forward_tokens_deployed_vs_D0"], 1) if r.arm != "DB" else cost.loc["DB", "bf16_prefill"] + cost.loc["DB", "bf16_decode"]
    ax.errorbar(x, r.diff_vs_D0, yerr=[[r.diff_vs_D0 - r.ci_lo], [r.ci_hi - r.diff_vs_D0]], fmt="o", capsize=3, label=r.arm)
ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log"); ax.set_xlabel("extra BF16-model forward tokens per request (prefill + window; DB = whole BF16 run)"); ax.set_ylabel("gain in P(answer complete & correct) vs AWQ3"); ax.legend(fontsize=7)
ax.set_title("Gain vs actual BF16 compute (dual-model deployment accounting)", fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/figs/duration_cost_tradeoff.png", dpi=130)
pd.set_option("display.width", 250); print(C.round(3).to_string()); print(S.round(3).to_string()); print(H.to_string()); print(pd.Series(prof).round(3).to_string())

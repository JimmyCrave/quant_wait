"""B2: entry / exit / submission analysis with right-censoring, cluster bootstrap over questions.
Usage: python analyze_b2.py --labels 'data/labels_qwen14b_*.jsonl' --out results/state_transition_results.csv"""
import argparse, collections, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from records import *
from stats import *

def metrics(recs):
    n = len(recs); out = {}
    if n == 0: return out
    out["n_traj"] = n
    out["P_ever_loop"] = np.mean([len(r["spells"]) > 0 for r in recs])
    at = sum(r["G_atrisk"] for r in recs); ev = sum(r["entries"] for r in recs); gt = sum(r["G_tokens"] for r in recs)
    out["entry_events"] = ev; out["entry_atrisk_steps"] = at
    out["entry_per_1000_G_steps"] = 1000 * ev / at if at else np.nan
    out["entry_per_100k_G_tokens"] = 1e5 * ev / gt if gt else np.nan
    out["GA_per_1000_G_steps"] = 1000 * sum(r["GA_events"] for r in recs) / at if at else np.nan
    out["loops_per_traj"] = np.mean([len(r["spells"]) for r in recs])
    out["frac_L_steps"] = sum(r["n_L_steps"] for r in recs) / max(1, sum(r["n_steps"] for r in recs))
    # first-entry KM (time in tokens; censored at end of thinking or trajectory end)
    t = [r["first_entry_tok"] if r["first_entry_tok"] is not None else (r["think_end_tok"] or r["n_tokens"]) for r in recs]
    e = [r["first_entry_tok"] is not None for r in recs]
    ts, S = km(t, e)
    for h in (2000, 8000, 32000): out[f"first_entry_cum_by_{h}tok"] = 1 - surv_at(ts, S, h)
    # L spells
    sp = [s for r in recs for s in r["spells"]]
    out["n_spells"] = len(sp); out["n_spells_censored"] = sum(s["censored"] for s in sp)
    out["n_exit_G"] = sum(s["exit"] == "G" for s in sp); out["n_exit_A"] = sum(s["exit"] == "A" for s in sp)
    out["n_exit_amb"] = sum(s["exit"] == "amb" for s in sp); out["n_spell_end_of_traj"] = sum(s["exit"] == "end" for s in sp)
    if sp:
        d = [s["dur_tok"] for s in sp]; ev_ = [not s["censored"] for s in sp]
        ts, S = km(d, ev_)
        out["spell_KM_median_tok"] = km_median(ts, S)
        for h in (500, 2000, 8000): out[f"spell_S_{h}tok"] = surv_at(ts, S, h)
        cause = [{"G": 1, "A": 2}.get(s["exit"], 0) for s in sp]
        for h in (2000, 8000):
            out[f"spell_CIF_G_{h}tok"] = cif(d, cause, 1, h); out[f"spell_CIF_A_{h}tok"] = cif(d, cause, 2, h)
        out["spell_frac_exit_G_among_observed"] = out["n_exit_G"] / max(1, out["n_exit_G"] + out["n_exit_A"])
        # step-time hazards for the toy model
        Ls = sum(s["dur_steps"] for s in sp); out["beta_hat_LtoG_per_Lstep"] = out["n_exit_G"] / Ls if Ls else np.nan
    # delay from first correct proposal to submission
    c = [r for r in recs if r["first_correct_tok"] is not None]
    out["n_reached_correct"] = len(c)
    if c:
        dl = [(r["think_end_tok"] if r["submitted"] else r["n_tokens"]) - r["first_correct_tok"] for r in c]
        ev_ = [r["submitted"] for r in c]; ts, S = km(dl, ev_)
        out["delay_KM_median_tok"] = km_median(ts, S)
        for h in (500, 2000, 8000): out[f"delay_S_{h}tok"] = surv_at(ts, S, h)
        out["frac_reached_correct_notsubmitted"] = np.mean([not r["submitted"] for r in c])
        out["frac_reached_correct_final_wrong_among_submitted"] = (np.mean([r["score"] == 0 for r in c if r["submitted"]]) if any(r["submitted"] for r in c) else np.nan)
        out["frac_correct_then_later_wrong_proposal"] = np.mean([r["n_wrong_after_correct"] > 0 for r in c])
        out["frac_reached_correct_final_score1"] = np.mean([r["score"] == 1 for r in c])
    out["frac_traj_reached_correct"] = len(c) / n
    out["accuracy"] = np.mean([r["score"] for r in recs]); out["frac_budget_exhausted"] = np.mean([r["finish_reason"] == "length" for r in recs])
    return out

def toy_check(recs):
    """alpha=G->L, beta=L->G, gamma=G->A per step. Check assumptions of E[T_A|G]=(1/gamma)(1+alpha/beta)."""
    at = sum(r["G_atrisk"] for r in recs); alpha = sum(r["entries"] for r in recs) / max(1, at); gamma = sum(r["GA_events"] for r in recs) / max(1, at)
    sp = [s for r in recs for s in r["spells"]]; Ls = sum(s["dur_steps"] for s in sp)
    beta = sum(s["exit"] == "G" for s in sp) / max(1, Ls)
    pred = (1 / gamma) * (1 + alpha / beta) if gamma > 0 and beta > 0 else np.nan
    # observed: restricted mean steps until first A (KM, censored at trajectory end)
    t = [next((i for i, ch in enumerate(r["states_str"]) if ch == "A"), r["n_steps"]) for r in recs]
    e = [("A" in r["states_str"]) for r in recs]; ts, S = km(t, e)
    tau = max(t); out = dict(alpha=alpha, beta=beta, gamma=gamma, E_TA_pred_steps=pred, E_TA_obs_rmst_steps=rmst(ts, S, tau), tau_steps=tau)
    out["LtoA_direct_frac_of_observed_exits"] = sum(s["exit"] == "A" for s in sp) / max(1, sum(s["exit"] in ("A", "G") for s in sp))
    # non-stationarity: entry hazard by trajectory-position tercile (in steps)
    for q, (lo, hi) in enumerate([(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)]):
        a = ev = 0
        for r in recs:
            S_ = r["states_str"]; n = len(S_); ent = set(r["entry_idx"])
            for i in range(n - 1):
                if S_[i] == "G" and lo <= i / n < hi and (S_[i + 1] not in ("U", "X")): a += 1; ev += (i + 1) in ent
        out[f"entry_hazard_per1000_posTercile{q}"] = 1000 * ev / a if a else np.nan
    # memory: exit-to-G hazard by loop index and by spell age (first 8 steps vs later)
    for name, sel in (("loop1", lambda s: s["index"] == 0), ("loop2plus", lambda s: s["index"] >= 1)):
        ss = [s for s in sp if sel(s)]; L_ = sum(s["dur_steps"] for s in ss)
        out[f"beta_{name}"] = sum(s["exit"] == "G" for s in ss) / max(1, L_)
    early = sum(min(s["dur_steps"], 8) for s in sp); late = sum(max(0, s["dur_steps"] - 8) for s in sp)
    out["beta_age_le8steps"] = sum((s["exit"] == "G") and s["dur_steps"] <= 8 for s in sp) / max(1, early)
    out["beta_age_gt8steps"] = sum((s["exit"] == "G") and s["dur_steps"] > 8 for s in sp) / max(1, late)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--labels"); ap.add_argument("--out"); ap.add_argument("--figdir"); ap.add_argument("--B", type=int, default=500); ap.add_argument("--labels2", default=None)
    a = ap.parse_args()
    steps, lab = load_steps(), load_labels(a.labels); lab2 = load_labels(a.labels2) if a.labels2 else None; VARS = ('auto', 'hc', 'cons', 'det') if lab2 else ('auto', 'hc', 'det')
    have = {k for k, _ in lab}; steps = {k: v for k, v in steps.items() if any((k, i) in lab for i in (0, 1, 2, 3))} 
    recs = collections.defaultdict(list); curves = {}
    for tid, tr in steps.items():
        ann = annotate(tr, lab, lab2)
        for v in VARS: recs[(tr["config"], v)].append(build_record(tr, ann, v))
    rows = []
    for (cfg, v), rs in sorted(recs.items()):
        g = collections.defaultdict(list); [g[r["task_id"]].append(r) for r in rs]
        est, ci = cluster_boot(g, metrics, B=a.B)
        for k, val in est.items():
            rows.append(dict(config=cfg, label_variant=v, metric=k, estimate=val, ci_lo=ci[k][0], ci_hi=ci[k][1], n_traj=len(rs), n_questions=len(g)))
        tc = toy_check(rs)
        for k, val in tc.items(): rows.append(dict(config=cfg, label_variant=v, metric="toy:" + k, estimate=val, ci_lo=np.nan, ci_hi=np.nan, n_traj=len(rs), n_questions=len(g)))
        curves[(cfg, v)] = rs
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    # survival curves of L durations and of correct-answer->submission delay
    for v in VARS:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for cfg, colr in (("bf16", "C0"), ("awq3", "C3"), ("awq3_pen", "C2")):
            rs = curves.get((cfg, v));
            if not rs: continue
            sp = [s for r in rs for s in r["spells"]]
            if sp:
                ts, S = km([s["dur_tok"] for s in sp], [not s["censored"] for s in sp]); ax[0].step(np.r_[0, ts], np.r_[1, S], where="post", color=colr, label=f"{cfg} (n={len(sp)}, cens={sum(s['censored'] for s in sp)})")
            c = [r for r in rs if r["first_correct_tok"] is not None]
            if c:
                ts, S = km([(r["think_end_tok"] if r["submitted"] else r["n_tokens"]) - r["first_correct_tok"] for r in c], [r["submitted"] for r in c]); ax[1].step(np.r_[0, ts], np.r_[1, S], where="post", color=colr, label=f"{cfg} (n={len(c)})")
            t = [r["first_entry_tok"] if r["first_entry_tok"] is not None else (r["think_end_tok"] or r["n_tokens"]) for r in rs]; ts, S = km(t, [r["first_entry_tok"] is not None for r in rs]); ax[2].step(np.r_[0, ts], np.r_[1, S], where="post", color=colr, label=f"{cfg} (n={len(rs)})")
        ax[0].set_title("P(still in loop) vs time in loop (KM, censored at trajectory end)"); ax[1].set_title("P(not yet submitted) after first correct answer"); ax[2].set_title("P(no loop entered yet) vs token position")
        for x in ax: x.set_xscale("symlog", linthresh=100); x.set_xlabel("tokens"); x.set_ylim(0, 1.02); x.legend(fontsize=7)
        fig.suptitle(f"label variant: {v}"); fig.tight_layout(); fig.savefig(f"{a.figdir}/survival_{v}.png", dpi=130); plt.close(fig)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    piv = df[~df.metric.str.startswith("toy:")].pivot_table(index="metric", columns=["label_variant", "config"], values="estimate")
    print(piv.round(4).to_string()); print(df[df.metric.str.startswith("toy:")].pivot_table(index="metric", columns=["label_variant", "config"], values="estimate").round(5).to_string())
if __name__ == "__main__": main()

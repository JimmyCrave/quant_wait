"""B4 analysis on streamed teacher-forced statistics (data/tf/*.npz). Reports raw-logit and sampling-distribution (T=.6,top-p=.95) quantities,
event-window contrasts against position-matched controls (paired, question-clustered bootstrap) and step-level prediction of state transitions
(group-CV logistic regression; position-only baseline vs +distribution features)."""
import argparse, glob, collections, json
import numpy as np, pandas as pd
sys_path = __import__("sys").path.insert(0, __import__("os").path.dirname(__file__))
from records import *
from stats import cluster_boot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = ["KL_bq", "KL_qb", "sTV", "sKL_bq", "sOut_b", "H_b", "H_q", "sH_b", "sH_q", "mk_b", "mk_q", "smk_b", "smk_q", "end_b", "end_q", "send_b", "send_q",
        "mg_end_b", "mg_end_q", "mg_mk_b", "mg_mk_q", "mg_top_b", "mg_top_q", "lp_real_b", "lp_real_q"]
DER = {"dH": ("H_q", "H_b"), "dmk": ("mk_q", "mk_b"), "dsmk": ("smk_q", "smk_b"), "dend": ("end_q", "end_b"), "dsend": ("send_q", "send_b"),
       "dmg_end": ("mg_end_q", "mg_end_b"), "dmg_mk": ("mg_mk_q", "mg_mk_b"), "dlp_real": ("lp_real_q", "lp_real_b"), "dsH": ("sH_q", "sH_b")}
FEATS = BASE + list(DER)
W = 128

def load_tf(tid):
    p = f"{LAB}/data/tf/{tid.replace('/', '_')}.npz"
    try: d = np.load(p)
    except Exception: return None
    names = list(d["names"]); F = d["feats"].astype(np.float32)
    cols = {n: F[:, i] for n, i in zip(names, range(len(names)))}
    for k, (a, b) in DER.items(): cols[k] = cols[a] - cols[b]
    return {k: np.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4) for k, v in cols.items()}

def csum(cols): return {k: np.concatenate([[0.0], np.cumsum(v.astype(np.float64))]) for k, v in cols.items()}
def wmean(cs, a, b): return {k: (c[b] - c[a]) / max(1, b - a) for k, c in cs.items()}

def events_for(tr, ann, variant="hc"):
    S, st = ann[variant], tr["steps"]; n = len(st); ntok = tr["n_tokens_retok"]; ev = []
    for a_, b_ in runs(S, "L", 2):
        ev.append(("pre_entry", st[a_]["t0"] - W, st[a_]["t0"]))
        if a_ + 3 < b_ and st[min(b_, n - 1)]["t0"] - st[a_ + 3]["t0"] >= W: ev.append(("in_loop", st[a_ + 3]["t0"], st[a_ + 3]["t0"] + W))
    ca = next((p for p in ann["props"] if p["correct"]), None)
    if ca: ev.append(("post_correct", st[ca["step"]]["t1"], st[ca["step"]]["t1"] + W))
    if tr["think_end_tok"]: ev.append(("pre_submit", tr["think_end_tok"] - W, tr["think_end_tok"]))
    return [(k, a, b) for k, a, b in ev if a >= 0 and b <= ntok]

def control_window(tr, ann, evs, near, rng, variant="hc"):
    """G_hc-only window of W tokens, >=256 tokens from every event window, nearest in position to `near`"""
    S, st = ann[variant], tr["steps"]; ntok = tr["n_tokens_retok"]
    bad = [(a - 256, b + 256) for _, a, b in evs]
    okG = np.zeros(ntok + 1, bool)
    for i, s in enumerate(st):
        if S[i] == "G": okG[s["t0"]:min(s["t1"], ntok)] = True
    best, bd = None, 1e9
    for a in range(0, ntok - W, 64):
        if not okG[a:a + W].all() or any(a < y and a + W > x for x, y in bad): continue
        d = abs(np.log1p(a) - np.log1p(near))
        if d < bd: best, bd = a, d
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--labels"); ap.add_argument("--out"); ap.add_argument("--B", type=int, default=300); a = ap.parse_args()
    steps, lab = load_steps(), load_labels(a.labels); rng = np.random.default_rng(0)
    rows_state, win_rows, step_rows = [], [], []
    for tid, tr in steps.items():
        if tr["config"] not in ("bf16", "awq3") or (tid, 0) not in lab: continue
        cols = load_tf(tid)
        if cols is None or len(next(iter(cols.values()))) != tr["n_tokens_retok"]: continue
        cs = csum(cols); ann = annotate(tr, lab); st = tr["steps"]; n = len(st)
        # (1) per-state means (token weighted)
        for var in ("hc",):
            S = ann[var]
            for state in ("G", "L", "X", "A"):
                idx = [i for i in range(n) if S[i] == state and st[i]["phase"] == "think"]
                if not idx: continue
                tot = sum(st[i]["t1"] - st[i]["t0"] for i in idx)
                m = {k: sum(cs[k][min(st[i]["t1"], len(cols[k]))] - cs[k][st[i]["t0"]] for i in idx) / tot for k in FEATS}
                rows_state.append(dict(traj_id=tid, origin=tr["config"], task_id=tr["task_id"], state=state, n_tok=tot, **m))
        # (2) windows
        evs = events_for(tr, ann)
        for kind, x, y in evs:
            c = control_window(tr, ann, evs, x, rng)
            if c is None: continue
            e, k_ = wmean(cs, x, y), wmean(cs, c, c + W)
            win_rows.append(dict(traj_id=tid, origin=tr["config"], task_id=tr["task_id"], kind=kind, start=x, **{f"ev_{f}": e[f] for f in FEATS}, **{f"ct_{f}": k_[f] for f in FEATS}))
        # (3) step-level at-risk G -> entry / L -> exit G / G -> A
        S = ann["hc"]; starts = {a_ for a_, _ in runs(S, "L", 2)}
        for i in range(n - 1):
            if S[i] not in ("G", "L") or S[i + 1] == "X" or st[i]["phase"] != "think": continue
            m = wmean(cs, st[i]["t0"], max(st[i]["t1"], st[i]["t0"] + 1))
            if S[i] == "G": y = {"entry": int((i + 1) in starts), "G2A": int(S[i + 1] == "A")}
            else: y = {"L2G": int(S[i + 1] == "G"), "L2A": int(S[i + 1] == "A")}
            step_rows.append(dict(traj_id=tid, origin=tr["config"], task_id=tr["task_id"], cur=S[i], logpos=np.log1p(st[i]["t0"]), **y, **{f: m[f] for f in FEATS}))
    pd.DataFrame(rows_state).groupby(["origin", "state"])[FEATS].mean().reset_index().to_csv(a.out.replace(".csv", "_by_state.csv"), index=False)
    # window contrasts
    W_ = pd.DataFrame(win_rows); out = []
    for (kind, origin), g in W_.groupby(["kind", "origin"]):
        for f in FEATS:
            d = (g[f"ev_{f}"] - g[f"ct_{f}"]); per_q = d.groupby(g["task_id"]).mean()
            rng_ = np.random.default_rng(1); bs = [per_q.sample(len(per_q), replace=True, random_state=int(rng_.integers(1e9))).mean() for _ in range(a.B)]
            out.append(dict(analysis="window_minus_matched_control", kind=kind, origin=origin, feature=f, estimate=per_q.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                            n_events=len(g), n_questions=len(per_q), event_mean=g[f"ev_{f}"].mean(), control_mean=g[f"ct_{f}"].mean()))
    # transition prediction
    D = pd.DataFrame(step_rows)
    for target, cur in (("entry", "G"), ("G2A", "G"), ("L2G", "L"), ("L2A", "L")):
        d = D[D.cur == cur]
        for origin in ("all", "bf16", "awq3"):
            dd = d if origin == "all" else d[d.origin == origin]
            if dd[target].sum() < 15 or dd[target].nunique() < 2 or dd.task_id.nunique() < 10: continue
            grp = dd.task_id.values; y = dd[target].values
            for name, cols_ in (("position_only", ["logpos"]), ("pos+KL/entropy", ["logpos", "KL_bq", "sTV", "H_b", "dH"]),
                                ("pos+markers", ["logpos", "mk_b", "dmk", "smk_b", "dsmk"]), ("pos+termination", ["logpos", "end_b", "dend", "mg_end_b", "dmg_end"]),
                                ("pos+all", ["logpos"] + FEATS)):
                X = dd[cols_].values; p = np.zeros(len(y))
                for tr_i, te_i in GroupKFold(5).split(X, y, grp):
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(X[tr_i], y[tr_i]); p[te_i] = clf.predict_proba(X[te_i])[:, 1]
                out.append(dict(analysis="predict_transition_groupCV_AUC", kind=target, origin=origin, feature=name, estimate=roc_auc_score(y, p), n_events=int(y.sum()), n_questions=len(set(grp)),
                                event_mean=y.mean(), control_mean=len(y)))
    pd.DataFrame(out).to_csv(a.out, index=False); print(pd.DataFrame(out).query("analysis=='predict_transition_groupCV_AUC'").round(3).to_string())
if __name__ == "__main__": main()

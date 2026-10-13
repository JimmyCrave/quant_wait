import numpy as np
def km(times, events):
    """Kaplan-Meier. returns sorted unique event times and survival after each"""
    t = np.asarray(times, float); e = np.asarray(events, int)
    if len(t) == 0: return np.array([]), np.array([])
    ts = np.unique(t[e == 1]); S, s = [], 1.0
    for u in ts:
        n_at = (t >= u).sum(); d = ((t == u) & (e == 1)).sum(); s *= 1 - d / n_at; S.append(s)
    return ts, np.array(S)
def surv_at(ts, S, x):
    if len(ts) == 0: return 1.0
    i = np.searchsorted(ts, x, "right") - 1
    return 1.0 if i < 0 else float(S[i])
def km_median(ts, S):
    i = np.where(S <= 0.5)[0]
    return float(ts[i[0]]) if len(i) else np.nan
def cif(times, cause, k, horizon):
    """Aalen-Johansen cumulative incidence of cause k at horizon. cause: 0=censored."""
    t = np.asarray(times, float); c = np.asarray(cause, int)
    if len(t) == 0: return np.nan
    S, F = 1.0, 0.0
    for u in np.unique(t[(c > 0) & (t <= horizon)]):
        n_at = (t >= u).sum(); dk = ((t == u) & (c == k)).sum(); da = ((t == u) & (c > 0)).sum()
        F += S * dk / n_at; S *= 1 - da / n_at
    return F
def rmst(ts, S, tau):
    if len(ts) == 0: return float(tau)
    x = np.concatenate([[0], ts[ts < tau], [tau]]); s = np.concatenate([[1.0], S[ts < tau]])
    return float(np.sum(s * np.diff(x)))
def cluster_boot(groups, fn, B=500, seed=0):
    """groups: dict question -> list of records; fn(list)->dict of scalars. percentile CI of each key."""
    rng = np.random.default_rng(seed); keys = list(groups)
    allr = [r for k in keys for r in groups[k]]; est = fn(allr); out = {k: [] for k in est}
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        r = fn([x for i in pick for x in groups[keys[i]]])
        for k in out: out[k].append(r[k])
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}
    return est, ci

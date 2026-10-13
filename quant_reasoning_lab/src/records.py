"""Per-trajectory event records (entry / exit / submission) derived from step states. Used by B2 and by prefix-bank selection."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from states import *

def build_record(tr, ann, variant):
    S = ann[variant]; steps = tr["steps"]; n = len(steps)
    t0 = np.array([s["t0"] for s in steps] + [tr["n_tokens_retok"]]); ntok = tr["n_tokens_retok"]
    unk = "U" if variant == "auto" else "X"
    # at-risk G steps (state G, next step known and still in think or the first answer-phase step)
    L_runs = runs(S, "L", 2); starts = {a for a, _ in L_runs}
    atrisk, events, g_tok, ga_events = 0, 0, 0, 0
    entry_idx = []
    for i in range(n - 1):
        if S[i] != "G": continue
        nxt = S[i + 1]
        if nxt == unk: continue
        atrisk += 1; g_tok += int(t0[i + 1] - t0[i])
        if (i + 1) in starts: events += 1; entry_idx.append(i + 1)
        if nxt == "A": ga_events += 1
    spells = []
    for k, (a, b) in enumerate(L_runs):
        if b < n:
            nx = S[b]; ex = {"G": "G", "A": "A"}.get(nx, "amb"); dur = int(t0[b] - t0[a]); cens = ex == "amb"
        else:
            ex, dur, cens = "end", int(ntok - t0[a]), True
        spells.append(dict(start_step=a, end_step=b, start_tok=int(t0[a]), dur_tok=dur, dur_steps=b - a, exit=ex, censored=cens, index=k))
    n_L_steps = sum(b - a for a, b in L_runs)
    props = ann["props"]; ca = [p for p in props if p["correct"]]
    te = tr["think_end_tok"]
    rec = dict(traj_id=tr["traj_id"], config=tr["config"], seed=tr["seed"], task_id=tr["task_id"], split=tr["split"], variant=variant,
               n_tokens=ntok, n_steps=n, finish_reason=tr["finish_reason"], score=tr["score"], submitted=te is not None,
               think_end_tok=te, G_atrisk=atrisk, G_tokens=g_tok, entries=events, entry_idx=entry_idx, GA_events=ga_events,
               n_L_steps=n_L_steps, spells=spells, n_props=len(props), first_prop_tok=(props[0]["tok"] if props else None),
               first_correct_tok=(ca[0]["tok"] if ca else None), first_correct_step=(ca[0]["step"] if ca else None),
               n_wrong_after_correct=(sum(1 for p in props if (not p["correct"]) and ca and p["tok"] > ca[0]["tok"])),
               first_entry_tok=(int(t0[entry_idx[0]]) if entry_idx else None),
               states_str="".join(S))
    return rec

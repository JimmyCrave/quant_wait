"""List of pre-specified event windows (+ position-matched G_hc controls) for the directional (token-level) analysis."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from analyze_b4 import *
steps, lab = load_steps(), load_labels("data/labels_qwen14b_*.jsonl"); rng = np.random.default_rng(0); out = []
for tid, tr in steps.items():
    if tr["config"] not in ("bf16", "awq3") or (tid, 0) not in lab: continue
    ann = annotate(tr, lab); evs = events_for(tr, ann)
    for kind, x, y in evs:
        c = control_window(tr, ann, evs, x, rng)
        out.append(dict(traj_id=tid, origin=tr["config"], kind=kind, start=x, end=y))
        if c is not None: out.append(dict(traj_id=tid, origin=tr["config"], kind="control_for_" + kind, start=c, end=c + W))
json.dump(out, open("data/b4_windows.json", "w")); print(len(out))

"""Ingest filled human annotation CSV(s) and compare with the AUTOMATIC annotators (Qwen2.5-14B, phi-4). Refuses to report anything if no human rows are filled.
usage: ingest_human_labels.py annotation_template_filled_A.csv [annotation_template_filled_B.csv]"""
import sys, os, json, itertools
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from states import *
from sklearn.metrics import cohen_kappa_score
files = sys.argv[1:]; H = [pd.read_csv(f).dropna(subset=["label_GLAU"]) for f in files]
if not files or all(len(h) == 0 for h in H): print("NO HUMAN LABELS FOUND - nothing to report. Status stays NOT COMPLETED."); sys.exit(0)
key = json.load(open(f"{LAB}/dur_study/human_pack_60_KEY_private.json")); q = load_labels(f"{LAB}/data/labels_qwen14b_*.jsonl"); p = load_labels(f"{LAB}/data/labels_phi4_all_sh*.jsonl")
def merged(h):
    h = h.copy(); h["step"] = [key[a]["start"] + j for a, j in zip(h.anon_id, h.local_step)]; h["traj_id"] = [key[a]["traj_id"] for a in h.anon_id]
    h["qwen"] = [q.get((t, s), ("U", None))[0] for t, s in zip(h.traj_id, h.step)]; h["phi4"] = [p.get((t, s), ("U", None))[0] for t, s in zip(h.traj_id, h.step)]; h["stratum"] = [key[a]["stratum"] for a in h.anon_id]
    h["config"] = [key[a]["config"] for a in h.anon_id]; return h
M = [merged(h) for h in H]
for i, m in enumerate(M):
    print(files[i], "rows", len(m), "U share", (m.label_GLAU == "U").mean())
    for a in ("qwen", "phi4"): print("  kappa(human, %s) = %.3f ; L-vs-notL kappa = %.3f" % (a, cohen_kappa_score(m.label_GLAU, m[a]), cohen_kappa_score(m.label_GLAU == "L", m[a] == "L")))
    print(m.groupby(["config"]).apply(lambda d: pd.Series(dict(frac_L_human=(d.label_GLAU == "L").mean(), frac_L_qwen=(d.qwen == "L").mean(), frac_L_phi4=(d.phi4 == "L").mean()))))
for (i, a), (j, b) in itertools.combinations(list(enumerate(M)), 2):
    mm = a.merge(b, on=["anon_id", "local_step"]); print("inter-human kappa", cohen_kappa_score(mm.label_GLAU_x, mm.label_GLAU_y))

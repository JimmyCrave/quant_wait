"""B1e: reliability of the automatic annotators on the audit windows. NO human labels exist; all numbers are annotator-vs-annotator / annotator-vs-detector."""
import json, itertools, collections, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from states import *
from sklearn.metrics import cohen_kappa_score
key = json.load(open(f"{LAB}/audit_key_private.json")); steps = load_steps()
ann = {"qwen14b_ev": load_labels(f"{LAB}/data/audit_qwen14b_ev.jsonl"), "qwen14b_noev": load_labels(f"{LAB}/data/audit_qwen14b_noev.jsonl"),
       "phi4_ev": load_labels(f"{LAB}/data/audit_phi4_ev.jsonl"), "qwen14b_fullrun": load_labels(f"{LAB}/data/labels_qwen14b_*.jsonl")}
rows = []
for aid, w in key.items():
    st = steps[w["traj_id"]]["steps"]
    for i in range(w["start"], w["end"]):
        r = dict(anon=aid, stratum=w["stratum"], config=w["config"], traj_id=w["traj_id"], step=i, cand=st[i]["cand"], cand_text=st[i]["cand_text"], cand_sem=st[i]["cand_sem"], phase=st[i]["phase"])
        for k, a in ann.items(): r[k] = a.get((w["traj_id"], i), ("U", None))[0]
        rows.append(r)
D = pd.DataFrame(rows); D = D[D.phase == "think"]; D.to_csv(f"{LAB}/results/audit_step_labels_with_key.csv", index=False)
out = []
names = list(ann)
for a, b in itertools.combinations(names, 2):
    for sub, d in (("all", D), ("bf16", D[D.config == "bf16"]), ("awq3", D[D.config == "awq3"])):
        out.append(dict(kind="pairwise", a=a, b=b, subset=sub, n=len(d), agree=(d[a] == d[b]).mean(), kappa4=cohen_kappa_score(d[a], d[b]),
                        kappa_L_vs_notL=cohen_kappa_score(d[a] == "L", d[b] == "L"), both_L=int(((d[a] == "L") & (d[b] == "L")).sum()), either_L=int(((d[a] == "L") | (d[b] == "L")).sum())))
for k in names:
    for sub, d in (("all", D), ("bf16", D[D.config == "bf16"]), ("awq3", D[D.config == "awq3"])):
        out.append(dict(kind="label_dist", a=k, subset=sub, n=len(d), frac_G=(d[k] == "G").mean(), frac_L=(d[k] == "L").mean(), frac_A=(d[k] == "A").mean(), frac_U=(d[k] == "U").mean(),
                        L_given_cand=(d[k][d.cand] == "L").mean(), L_given_nocand=(d[k][~d.cand] == "L").mean(), kappa_vs_detector_L=cohen_kappa_score(d[k] == "L", d.cand)))
# by stratum
for (s, c), d in D.groupby(["stratum", "config"]):
    out.append(dict(kind="by_stratum", a="qwen14b_ev", subset=f"{s}|{c}", n=len(d), frac_L=(d.qwen14b_ev == "L").mean(), frac_U=(d.qwen14b_ev == "U").mean(), frac_G=(d.qwen14b_ev == "G").mean(),
                    pairwise_agree_qwen_phi=(d.qwen14b_ev == d.phi4_ev).mean(), cand_rate=d.cand.mean()))
# high confidence coverage & agreement: consensus of the 3 audit annotators
cons = D[["qwen14b_ev", "qwen14b_noev", "phi4_ev"]]
D["consensus_L3"] = (cons == "L").all(1); D["consensus_G3"] = (cons == "G").all(1); D["any_disagree"] = cons.nunique(axis=1) > 1
D["hc_L"] = (D.qwen14b_ev == "L") & D.cand; D["hc_G"] = (D.qwen14b_ev == "G") & ~D.cand
for sub, d in (("all", D), ("bf16", D[D.config == "bf16"]), ("awq3", D[D.config == "awq3"])):
    out.append(dict(kind="highconf", subset=sub, n=len(d), cov_hcL=d.hc_L.mean(), cov_hcG=d.hc_G.mean(), cov_hc_total=(d.hc_L | d.hc_G).mean(),
                    hcL_phi_agrees=(d.phi4_ev[d.hc_L] == "L").mean(), hcL_noev_agrees=(d.qwen14b_noev[d.hc_L] == "L").mean(),
                    hcG_phi_agrees=(d.phi4_ev[d.hc_G] == "G").mean(), hcG_noev_agrees=(d.qwen14b_noev[d.hc_G] == "G").mean(),
                    frac_steps_any_disagree=d.any_disagree.mean(), frac_consensusL3=d.consensus_L3.mean(), frac_consensusG3=d.consensus_G3.mean()))
pd.DataFrame(out).to_csv(f"{LAB}/results/audit_results.csv", index=False)
pd.set_option("display.width", 250); print(pd.DataFrame(out).round(3).to_string())
# disagreement list for human reviewers (priority review order); keys are NOT part of the anonymous pack
D[D.any_disagree][["anon", "step", "stratum", "qwen14b_ev", "qwen14b_noev", "phi4_ev"]].rename(columns={"step": "global_step"}).to_csv(f"{LAB}/results/audit_disagreements_for_human_review.csv", index=False)

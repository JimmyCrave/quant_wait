"""Audit 2: why 'difference of group means' != 'paired difference' in mechanism_report.
Reported (counterfactual_results.csv): mean_bf16 / mean_awq3 = unweighted means over PREFIXES (3 seeds averaged within prefix);
diff_awq3_minus_bf16 = mean over QUESTIONS of (mean over that question's prefixes of the paired diff) = question-weighted. When questions own different numbers
of prefixes the two differ. For B5 additionally each comparison uses a different anchor set (anchors that have both arms)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
LAB = "."
D = pd.read_csv("data/b3/continuation_outcomes.csv"); D["t_sub_rmst"] = np.where(D.submitted, D.t_sub, D.budget).astype(float)
rows = []
for cat, ms in (("G", ["det_entry", "submitted"]), ("L", ["escaped", "det_entry"]), ("CA", ["submitted", "correct_by_end", "t_sub_rmst"])):
    d = D[(D.cat == cat) & (D.budget == 4096)]
    for m in ms:
        g = d.groupby(["prefix_id", "task_id", "model"])[m].mean().unstack("model").dropna().reset_index()
        g["diff"] = g.awq3 - g.bf16; nq = g.groupby("task_id").size()
        perq = g.groupby("task_id").agg(b=("bf16", "mean"), a=("awq3", "mean"), d=("diff", "mean"))
        rows.append(dict(category=cat, metric=m, n_prefixes=len(g), n_questions=len(nq), prefixes_per_question_min=nq.min(), prefixes_per_question_max=nq.max(),
                         mean_bf16_prefixwt=g.bf16.mean(), mean_awq3_prefixwt=g.awq3.mean(), diff_of_means_prefixwt=g.awq3.mean() - g.bf16.mean(), mean_diff_prefixwt=g["diff"].mean(),
                         diff_questionwt_REPORTED=perq.d.mean(), mean_bf16_questionwt=perq.b.mean(), mean_awq3_questionwt=perq.a.mean(), gap_reported_minus_prefixwt=perq.d.mean() - g["diff"].mean()))
A = pd.DataFrame(rows); A.to_csv("dur_study/audit/audit2_b3_mean_vs_paired.csv", index=False)
# B5 : set-membership effect
B = pd.read_csv("data/b5/branch_outcomes.csv"); out = []
for fam, m in (("A", "submitted"), ("A", "correct_by_end"), ("E", "det_entry"), ("I", "escaped"), ("V", "correct_by_end")):
    d = B[(B.family == fam) & (((B.cond == "CW") & (B.W == 32)) | B.cond.isin(["C0", "C1"]) | ((B.cond == "C5") & (B.W == 32)) | (B.cond == "C0r"))]
    pa = d.groupby(["anchor_id", "cond"])[m].mean().unstack("cond")
    sets = {c: pa[c].dropna().index for c in pa.columns}
    both_pos = sets["C0"].intersection(sets["CW"]); both_rand = sets["C0r"].intersection(sets["C5"]); both_all = both_pos.intersection(both_rand)
    q = d.drop_duplicates("anchor_id").set_index("anchor_id").task_id
    def qw(x): return x.groupby(q.reindex(x.index).values).mean().mean()
    r = dict(family=fam, metric=m, n_anchors_C0=len(sets["C0"]), n_anchors_CW=len(sets["CW"]), n_anchors_C5=len(sets["C5"]), n_anchors_C0r=len(sets["C0r"]), n_both_pos=len(both_pos), n_both_rand=len(both_rand), n_all4=len(both_all),
             mean_C0_allanchors=pa.C0.mean(), mean_CW_allanchors=pa.CW.mean(), diff_of_means_pos=pa.CW.mean() - pa.C0.mean(),
             paired_pos_anchorwt=(pa.CW - pa.C0).dropna().mean(), paired_pos_questionwt_REPORTED=qw((pa.CW - pa.C0).dropna()),
             mean_C5_all=pa.C5.mean(), mean_C0r_all=pa.C0r.mean(), paired_rand_anchorwt=(pa.C5 - pa.C0r).dropna().mean(), paired_rand_questionwt_REPORTED=qw((pa.C5 - pa.C0r).dropna()))
    cont = ((pa.CW - pa.C0) - (pa.C5 - pa.C0r)).dropna(); r.update(contrast_anchorwt=cont.mean(), contrast_questionwt_REPORTED=qw(cont), n_contrast=len(cont),
                                                                  pos_effect_on_contrast_subset=(pa.CW - pa.C0).loc[cont.index].mean(), rand_effect_on_contrast_subset=(pa.C5 - pa.C0r).loc[cont.index].mean())
    out.append(r)
Bd = pd.DataFrame(out); Bd.to_csv("dur_study/audit/audit2_b5_mean_vs_paired.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(A.round(3).to_string()); print(Bd.round(3).T.to_string())

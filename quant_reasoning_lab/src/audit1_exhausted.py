"""Audit 1: budget-exhausted trajectories graded correct by the stage-1 scorer. Splits each trajectory into four separate facts:
 (a) stage-1 scorer says correct  (b) </think> emitted  (c) final answer complete (closed \\boxed{} AFTER </think>)  (d) post-</think> answer graded correct.
The stage-1 scorer extracts from the WHOLE generated text (boxed first), so a boxed answer inside the unfinished thinking can earn credit."""
import sys, os, json, collections
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
rows = []
for t in load_trajectories():
    text = t["text"]; i = text.find("</think>"); closed = i >= 0
    post = text[i + 8:] if closed else ""
    think = text[:i] if closed else text
    bx_post = boxed_answers(post); bx_think = boxed_answers(think)
    last_think = bx_think[-1][1] if bx_think else None
    rows.append(dict(traj_id=t["traj_id"], config=t["config"], seed=t["seed"], task_id=t["task_id"], finish=t["finish_reason"], exhausted=t["finish_reason"] == "length",
                     scorer_correct=int(t["score"] == 1), think_closed=closed, answer_complete=bool(bx_post), n_gen=t["n_gen"],
                     final_correct=int(bool(bx_post) and grade(post, t["gold"]) == 1),
                     boxed_in_think=bool(bx_think), last_think_boxed_correct=int(last_think is not None and grade("\\boxed{" + last_think + "}", t["gold"]) == 1)))
D = pd.DataFrame(rows); D.to_csv("dur_study/audit/audit1_per_trajectory.csv", index=False)
out = []
for (cfg), d in D.groupby("config"):
    for name, s in (("all", d), ("exhausted", d[d.exhausted]), ("not_exhausted", d[~d.exhausted])):
        if len(s) == 0: continue
        out.append(dict(config=cfg, subset=name, n=len(s), scorer_correct=s.scorer_correct.sum(), think_closed=int(s.think_closed.sum()), answer_complete=int(s.answer_complete.sum()),
                        final_correct=int(s.final_correct.sum()), scorer_correct_and_not_closed=int(((s.scorer_correct == 1) & ~s.think_closed).sum()),
                        scorer_correct_and_closed_no_complete_answer=int(((s.scorer_correct == 1) & s.think_closed & ~s.answer_complete).sum()),
                        scorer_correct_and_final_correct=int(((s.scorer_correct == 1) & (s.final_correct == 1)).sum()),
                        scorer_correct_and_final_wrong=int(((s.scorer_correct == 1) & (s.final_correct == 0)).sum()),
                        final_correct_but_scorer_wrong=int(((s.scorer_correct == 0) & (s.final_correct == 1)).sum())))
O = pd.DataFrame(out); O.to_csv("dur_study/audit/audit1_summary.csv", index=False)
# contingency over exhausted & scorer-correct only
E = D[D.exhausted & (D.scorer_correct == 1)]
E2 = E.groupby(["config", "think_closed", "answer_complete", "final_correct", "boxed_in_think"]).size().rename("n").reset_index(); E2.to_csv("dur_study/audit/audit1_exhausted_correct_breakdown.csv", index=False)
# headline accuracy: stage-1 scorer vs strict (final answer complete after </think> and correct), per config, with question-clustered SE
acc = []
for cfg, d in D.groupby("config"):
    q = d.groupby("task_id").agg(s=("scorer_correct", "mean"), f=("final_correct", "mean"))
    acc.append(dict(config=cfg, n_traj=len(d), acc_scorer=d.scorer_correct.mean(), acc_strict=d.final_correct.mean(), diff=d.scorer_correct.mean() - d.final_correct.mean(),
                    se_q_scorer=q.s.std(ddof=1) / np.sqrt(len(q)), se_q_strict=q.f.std(ddof=1) / np.sqrt(len(q))))
A = pd.DataFrame(acc); A.to_csv("dur_study/audit/audit1_accuracy_scorer_vs_strict.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(O.to_string()); print(E2.to_string()); print(A.round(4).to_string())

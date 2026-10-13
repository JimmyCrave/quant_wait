"""Outcome extraction for continuation branches (B3/B5). Primary loop outcome is LABELER-FREE: det-loop = >=2 consecutive continuation steps flagged by the
verbatim-repetition detector (Jaccard>=0.6 vs any earlier step incl. the prefix, or zlib ratio<0.25). Submission = the </think> token. Correctness is graded
offline against gold with the stage-1 grader and is never used online."""
import sys, os, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def outcomes(r, budget=None):
    st, ex = r["steps"], r["extra"]; pc, npre = ex["prefix_chars"], ex["n_prefix_tokens"]; budget = budget or ex["budget"]
    cs = [s for s in st if s["c0"] >= pc - 1]; think = [s for s in cs if s["phase"] == "think"]
    flags = [bool(s["cand_text"] or s["cand_intra"]) for s in think]
    te = r["think_end_tok"]; submitted = te is not None and te >= npre - 1
    ncont = ex["n_cont_tokens"]; t_sub = (te - npre) if submitted else None
    runs2 = [i for i in range(len(flags) - 1) if flags[i] and flags[i + 1]]
    det_entry = len(runs2) > 0
    # first token offset of a det run
    first_loop_tok = (think[runs2[0]]["t0"] - npre) if det_entry else None
    tail = flags[-4:] if len(flags) >= 2 else flags
    tail_loop = any(tail[i] and tail[i + 1] for i in range(len(tail) - 1)) if len(tail) >= 2 else False
    # progress run: >=3 consecutive unflagged continuation steps
    prog = any(not any(flags[i:i + 3]) for i in range(0, max(0, len(flags) - 2))) if len(flags) >= 3 else False
    correct_final = None
    if submitted:
        after = r["text_after_think"] if "text_after_think" in r else None
    return dict(n_cont=ncont, submitted=submitted, t_sub=t_sub, det_entry=det_entry, first_loop_tok=first_loop_tok, tail_in_loop=tail_loop,
                escaped=(submitted or not tail_loop), progress_run=prog, rep_rate=float(np.mean(flags)) if flags else 0.0, n_think_steps=len(think),
                finish=ex["cont_finish"])

def grade_final(r):
    """correctness of the final answer after </think> (None if not submitted) + any correct boxed proposal in the continuation"""
    ex = r["extra"]; text = r["_text"]; i = text.find("</think>", ex["prefix_chars"] - 1)
    fin = None
    if i >= 0: fin = grade(text[i + 8:], r["gold"][0] if isinstance(r["gold"], list) else r["gold"])
    cont_think = text[ex["prefix_chars"]:(i if i >= 0 else len(text))]
    props = [b for _, b in boxed_answers(cont_think)]
    any_correct = any(grade("\\boxed{" + p + "}", r["gold"]) for p in props) if props else False
    wrong_prop = any(not grade("\\boxed{" + p + "}", r["gold"]) for p in props) if props else False
    return fin, any_correct, wrong_prop

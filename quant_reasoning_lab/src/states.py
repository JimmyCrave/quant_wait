"""B1c/B2 data layer: merge steps + labels, derive per-step states, answer proposals (graded OFFLINE), loop runs.
Gold answers are used only for offline labels (answer correctness); nothing here feeds an online intervention."""
import sys, json, glob, re, os, functools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def load_steps(pattern=None):
    out = {}
    for p in sorted(glob.glob(pattern or f"{LAB}/data/steps_*.jsonl")):
        for l in open(p):
            r = json.loads(l); out[r["traj_id"]] = r
    return out

def load_labels(pattern):
    lab = {}
    for p in sorted(glob.glob(pattern)):
        for l in open(p):
            r = json.loads(l); lab[(r["traj_id"], r["step"])] = (r["label"], r["ans"])
    return lab

@functools.lru_cache(None)
def _g(ans, gold):
    a = ans.strip().strip("$").strip()
    return grade("\\boxed{" + a + "}", gold) if a else 0.0

def clean_ans(a): return re.sub(r"^\\\(|\\\)$", "", a.strip()).strip("$ ")

def annotate(tr, lab, lab2=None):
    """returns dict with per-step arrays. states: auto in G/L/A/U ; hc in G/L/A/X ; answer-phase steps = 'A'."""
    steps = tr["steps"]; n = len(steps); auto, hcs, props, det = [], [], [], []
    for s in steps:
        i = s["step"]
        if s["phase"] == "answer":
            auto.append("A"); hcs.append("A"); det.append("A"); continue
        det.append("L" if (s["cand_text"] or s["cand_intra"]) else "G")   # labeler-free verbatim-repetition variant (POST-HOC sensitivity)
        l, a = lab.get((tr["traj_id"], i), ("U", None))
        auto.append(l)
        if l == "L" and s["cand"]: hcs.append("L")
        elif l == "G" and not s["cand"]: hcs.append("G")
        elif l == "A": hcs.append("A")
        else: hcs.append("X")
        cands = []
        if a: cands.append(clean_ans(a))
        cands += [b for _, b in boxed_answers(s["text"])] if s["phase"] == "think" else []
        for c in dict.fromkeys(cands):
            props.append(dict(step=i, tok=s["t0"], ans=c, correct=bool(_g(c, tr["gold"]))))
    # hc smoothing: a single ambiguous step between two L steps is L
    hs = list(hcs)
    for i in range(1, n - 1):
        if hcs[i] == "X" and hcs[i - 1] == "L" and hcs[i + 1] == "L": hs[i] = "L"
    out = dict(auto=auto, hc=hs, det=det, props=props)
    if lab2 is not None:   # consensus of two independent LLM annotators (+ detector): L = both L & candidate ; G = both G & no candidate
        cs = []
        for s, a0 in zip(steps, auto):
            if s["phase"] == "answer" or a0 == "A": cs.append("A"); continue
            b0 = lab2.get((tr["traj_id"], s["step"]), ("U", None))[0]
            cs.append("L" if (a0 == "L" and b0 == "L" and s["cand"]) else "G" if (a0 == "G" and b0 == "G" and not s["cand"]) else "X")
        cc = list(cs)
        for i in range(1, n - 1):
            if cs[i] == "X" and cs[i - 1] == "L" and cs[i + 1] == "L": cc[i] = "L"
        out["cons"] = cc
    return out

def runs(states, target="L", minlen=2):
    """maximal runs of `target` with length>=minlen -> list of (start_idx, end_idx_exclusive)"""
    out, i, n = [], 0, len(states)
    while i < n:
        if states[i] == target:
            j = i
            while j < n and states[j] == target: j += 1
            if j - i >= minlen: out.append((i, j))
            i = j
        else: i += 1
    return out

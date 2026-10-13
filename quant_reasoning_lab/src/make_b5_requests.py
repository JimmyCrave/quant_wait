"""B5a: anchors + branch tables for position-controlled BF16-correction interventions. Rules fixed before any branch is run.
Families (anchors come from AWQ3-generated trajectories, high-confidence labels; V also from BF16-generated):
  E : window = the W tokens right BEFORE the first loop entry (first L_hc run >=2 steps starting after token 200)   [oracle: uses full trajectory]
  I : window starts at the 4th step of the first L_hc run that lasts >= W more tokens                                [oracle]
  A : window starts right after the step that first proposes a CORRECT answer (needs gold)                           [oracle]
  V : valid-correction harm check: prefix ends after the last WRONG proposal that precedes the first CORRECT proposal; window = next W tokens [oracle]
Branches per anchor (all share the identical prefix except random controls):
  C0 : quantized, no intervention          C1 : quantized + marker penalty (lambda=1.0) over the whole continuation
  CW : BF16 samples the W window tokens, then the quantized model continues (C2/C3/C4 for E/I/A, CV for V)
  C0r/C5 : a randomly placed window (same W, same trajectory, G_hc-only, >=128 tokens from the anchor, one of the 8 windows with closest mean quantized entropy) with its own no-intervention control C0r.
One anchor per question per family (random eligible trajectory/seed). 3 sampling seeds per branch. W=32 main; W in {8,64} on dev-split anchors of E,I,A."""
import sys, os, json, random, collections, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from records import *
from analyze_b4 import load_tf
from tf_stats import prompt_ids
rng = random.Random(4242); WMAIN = 32; MAXTOK = 4096; NPER = {"E": 48, "I": 48, "A": 48, "V": 24}
labels = sys.argv[1]; outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
steps, lab = load_steps(), load_labels(labels); tok = tokenizer(); trs = {t["traj_id"]: t for t in load_trajectories()}
cand = collections.defaultdict(list)
for tid, tr in steps.items():
    if tr["config"] not in ("bf16", "awq3") or (tid, 0) not in lab: continue
    ann = annotate(tr, lab); S = ann["hc"]; st = tr["steps"]; n = len(st); ntok = tr["n_tokens_retok"]; te = tr["think_end_tok"] or ntok
    R = runs(S, "L", 2)
    if tr["config"] == "awq3":
        e = next(((a_, b_) for a_, b_ in R if st[a_]["t0"] >= 200), None)
        if e: cand["E"].append((tid, st[e[0]]["t0"] - WMAIN, dict(entry_tok=st[e[0]]["t0"])))
        i_ = next(((a_, b_) for a_, b_ in R if b_ - a_ >= 5 and st[min(b_, n - 1)]["t0"] - st[a_ + 3]["t0"] >= 64), None)
        if i_: cand["I"].append((tid, st[i_[0] + 3]["t0"], dict(loop_start_tok=st[i_[0]]["t0"])))
        ca = next((p for p in ann["props"] if p["correct"]), None)
        if ca and st[ca["step"]]["t1"] + WMAIN < te: cand["A"].append((tid, st[ca["step"]]["t1"], dict(correct_step=ca["step"])))
    pr = ann["props"]; k = next((j for j, p in enumerate(pr) if p["correct"]), None)
    if k and any(not p["correct"] for p in pr[:k]):
        w = max(j for j in range(k) if not pr[j]["correct"]); s = st[pr[w]["step"]]["t1"]
        if s + WMAIN < te: cand["V"].append((tid, s, dict(wrong_step=pr[w]["step"], correct_step=pr[k]["step"])))

def pick(fam):
    byq = collections.defaultdict(list)
    for c in cand[fam]: byq[trs[c[0]]["task_id"]].append(c)
    qs = sorted(byq); rng.shuffle(qs)
    if fam == "V":   # balance origins
        chosen = []
        for o in ("bf16", "awq3"):
            qq = [q for q in qs if any(trs[c[0]]["config"] == o for c in byq[q])]
            used = {trs[c[0]]["task_id"] for c in chosen}
            chosen += [rng.choice([c for c in byq[q] if trs[c[0]]["config"] == o]) for q in qq if q not in used][:NPER["V"] // 2]
        return chosen
    return [rng.choice(byq[q]) for q in qs[:NPER[fam]]]

def rand_start(tid, anchor_start, W, h_target, ann):
    cols = load_tf(tid); H = np.concatenate([[0], np.cumsum(cols["H_q"].astype(np.float64))]); st = steps[tid]["steps"]; S = ann["hc"]; ntok = len(cols["H_q"])
    okG = np.zeros(ntok + 1, bool)
    for i, s in enumerate(st):
        if S[i] == "G" and s["phase"] == "think": okG[s["t0"]:min(s["t1"], ntok)] = True
    te = steps[tid]["think_end_tok"] or ntok
    c = [a for a in range(200, min(ntok, te) - W, 16) if okG[a:a + W].all() and abs(a - anchor_start) >= 128]
    if not c: return None
    c.sort(key=lambda a: abs((H[a + W] - H[a]) / W - h_target)); return rng.choice(c[:8])

anchors, branches = [], []
for fam in ("E", "I", "A", "V"):
    for tid, start, info in pick(fam):
        tr, full = steps[tid], trs[tid]; ann = annotate(tr, lab)
        ids_all = tok(full["text"], add_special_tokens=False).input_ids; split = full["split"]
        Ws = [WMAIN] + ([8, 64] if (split == "dev" and fam != "V") else [])
        aid = f"{fam}_{len(anchors):03d}"; cols = load_tf(tid)
        a = dict(anchor_id=aid, family=fam, traj_id=tid, origin=full["config"], task_id=full["task_id"], split=split, start_tok=start, gold=full["gold"], **info)
        anchors.append(a)
        def add(cond, W, s_tok, seed, aid_=aid):
            branches.append(dict(branch_id=f"{aid_}|{cond}|W{W}|s{seed}", anchor_id=aid, family=fam, cond=cond, W=W, seed=seed, start_tok=s_tok, task_id=full["task_id"], split=split,
                                 prompt_ids=prompt_ids(full["task_id"]), prefix_ids=ids_all[:s_tok]))
        for seed in (1, 2, 3):
            add("C0", 0, start, seed); add("C1", 0, start, seed)
            for W in Ws: add("CW", W, start, seed)
        for W in Ws:
            hH = float(cols["H_q"][start:start + W].mean()) if start + W <= len(cols["H_q"]) else float(cols["H_q"][-W:].mean())
            rs = rand_start(tid, start, W, hH, ann)
            a.setdefault("random_start", {})[str(W)] = rs
            if rs is None: continue
            for seed in (1, 2, 3):
                if W == WMAIN: add("C0r", 0, rs, seed)
                add("C5", W, rs, seed)
with open(f"{outdir}/anchors.jsonl", "w") as f:
    for a in anchors: f.write(json.dumps(a) + "\n")
with open(f"{outdir}/branches.jsonl", "w") as f:
    for b in branches: f.write(json.dumps(b) + "\n")
print("anchors", collections.Counter(a["family"] for a in anchors), "available", {k: len(v) for k, v in cand.items()})
print("branches", len(branches), collections.Counter((b["cond"], b["W"]) for b in branches))

"""B3a: build the prefix bank. Rules fixed in configs/prereg.yaml / here BEFORE any continuation is run. Labels used: 'hc' high-confidence states.
Categories: G (normal progress: G_hc step, last 5 steps without candidates, no L_hc step in the last 30 steps, no correct answer yet), L (cut after the 3rd consecutive L_hc step of a run), CA (cut right after the step that first proposes a correct answer).
Quotas: G 16+16, L 8+24, CA 16+16 (bf16-origin + awq3-origin); anchor-driven matching: for each L anchor, G/CA prefixes are taken from the same question & origin with the closest token position."""
import sys, os, json, random, math, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from records import *
ap = argparse.ArgumentParser(); ap.add_argument("--labels"); ap.add_argument("--out"); a = ap.parse_args()
rng = random.Random(31337)
steps, lab = load_steps(), load_labels(a.labels); tok = tokenizer()
trs = {t["traj_id"]: t for t in load_trajectories()}
cands = collections.defaultdict(list)   # (cat, origin) -> list of dict
for tid, tr in steps.items():
    if tr["config"] not in ("bf16", "awq3") or (tid, 0) not in lab: continue
    ann = annotate(tr, lab); S = ann["hc"]; st = tr["steps"]; n = len(st)
    L_runs = runs(S, "L", 3); first_L = min([a_ for a_, _ in runs(S, "L", 2)], default=None)
    base = dict(traj_id=tid, origin=tr["config"], task_id=tr["task_id"], seed=tr["seed"], split=tr["split"])
    for a_, b_ in L_runs:    # cut after 3rd L step of the run
        i = a_ + 2; cands[("L", tr["config"])].append(dict(base, cut_step=i, cut_tok=st[i]["t1"], cat="L"))
    ca = next((p for p in ann["props"] if p["correct"]), None)
    first_prop = next((p["step"] for p in ann["props"] if p["correct"]), n)   # no correct answer proposed yet
    if ca: cands[("CA", tr["config"])].append(dict(base, cut_step=ca["step"], cut_tok=st[ca["step"]]["t1"], cat="CA"))
    for i in range(3, n - 1):
        if st[i]["phase"] != "think" or S[i] != "G" or any(c for c in (x["cand"] for x in st[max(0, i - 4):i + 1])): continue
        if i >= first_prop or st[i]["t1"] < 300 or any(x == "L" for x in S[max(0, i - 30):i + 1]): continue   # not in/near a loop (last 30 steps)
        cands[("G", tr["config"])].append(dict(base, cut_step=i, cut_tok=st[i]["t1"], cat="G"))
QUOTA = {("G", "bf16"): 16, ("G", "awq3"): 16, ("L", "bf16"): 8, ("L", "awq3"): 24, ("CA", "bf16"): 16, ("CA", "awq3"): 16}
chosen, avail = [], {k: len(v) for k, v in cands.items()}
def pick_one_per_question(lst, k, near=None):
    if k <= 0: return []
    byq = collections.defaultdict(list); [byq[c["task_id"]].append(c) for c in lst]
    qs = sorted(byq); rng.shuffle(qs); out = []
    for q in qs:
        c = byq[q]; c = min(c, key=lambda x: abs(math.log(x["cut_tok"]) - math.log(near[q]))) if near and q in near else rng.choice(c)
        out.append(c)
        if len(out) == k: break
    return out
L_sel = {o: pick_one_per_question(cands[("L", o)], QUOTA[("L", o)]) for o in ("bf16", "awq3")}
for o in ("bf16", "awq3"):
    chosen += L_sel[o]
    anchors = {c["task_id"]: c["cut_tok"] for c in L_sel[o]}
    for cat in ("G", "CA"):
        lst = cands[(cat, o)]; k = QUOTA[(cat, o)]
        m = [c for c in lst if c["task_id"] in anchors]          # same-question matched first
        got = pick_one_per_question(m, k, near=anchors)
        rest = [c for c in lst if c["task_id"] not in {g["task_id"] for g in got}]
        got += pick_one_per_question(rest, k - len(got))
        chosen += got
bank = []
for i, c in enumerate(chosen):
    tr, full = steps[c["traj_id"]], trs[c["traj_id"]]
    from tf_stats import prompt_ids
    ids = tok(full["text"], add_special_tokens=False).input_ids[:c["cut_tok"]]
    pid = f"P{i:03d}"; c.update(prefix_id=pid, prompt_ids=prompt_ids(c["task_id"]), prefix_ids=ids, n_prefix_tokens=len(ids), gold=full["gold"],
                                heldout_sensitivity8192=(int(__import__("hashlib").sha1(pid.encode()).hexdigest(), 16) % 4 == 0),
                                origin_final_score=full["score"], prefix_text_tail=tok.decode(ids[-60:]))
    bank.append(c)
with open(a.out, "w") as f:
    for c in bank: f.write(json.dumps(c) + "\n")
print("available:", dict(avail)); print("selected:", collections.Counter((c["cat"], c["origin"]) for c in bank), "n questions:", len({c["task_id"] for c in bank}))
for cat in ("G", "L", "CA"): print(cat, "prefix len median", np.median([c["n_prefix_tokens"] for c in bank if c["cat"] == cat]))

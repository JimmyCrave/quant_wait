"""B1d: sample >=100 audit windows (10 steps) from detector-defined strata ONLY (no labeler output), seed 777; export anonymous pack."""
import sys, os, json, re, random, csv, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from states import *
rng = random.Random(777); W = 10; PER = 12
steps = load_steps(); cor = re.compile(r"^\s*(Wait|But|Hmm|Alternatively|Actually|However|Let me double|Let me check)", re.I)
pool = {}   # (stratum, config) -> list of (traj_id, start)
for tid, tr in steps.items():
    if tr["config"] not in ("bf16", "awq3"): continue
    S = tr["steps"]; th = [s["step"] for s in S if s["phase"] == "think"]; n = len(th)
    if n < 20: continue
    def add(stratum, s):
        s = max(4, min(s, n - W))
        pool.setdefault((stratum, tr["config"]), []).append((tid, s))
    cand = [S[i]["cand"] for i in range(n)]
    # normal_G: windows with zero candidates (take every 10th start to avoid overlap)
    for s in range(4, n - W, 10):
        c = sum(cand[s:s + W])
        if c == 0: add("normal_G", s)
        if c >= 5: add("repetition_candidate", s)
    for i in range(4, n - W):
        if cor.match(S[i]["text"]) and S[i]["n_novel_symbols"] >= 2 and not S[i]["cand"]: add("valid_correction_candidate", i - 3)
    add("near_end", n - W)
    for i in range(n):
        if "\\boxed" in S[i]["text"]: add("near_answer", i - 5); break
windows, key, rows = [], {}, []
for (stratum, cfg), lst in sorted(pool.items()):
    by_q = {}
    for tid, s in lst: by_q.setdefault(steps[tid]["task_id"] + "|" + tid, []).append((tid, s))
    keys = sorted(by_q); rng.shuffle(keys); chosen = []; seenq = set()
    for k in keys:   # prefer distinct questions
        q = k.split("|")[0]
        if q in seenq: continue
        seenq.add(q); chosen.append(rng.choice(by_q[k]))
        if len(chosen) == PER: break
    for tid, s in chosen: windows.append(dict(traj_id=tid, start=s, end=s + W, stratum=stratum, config=cfg))
rng.shuffle(windows)
os.makedirs(f"{LAB}/annotation_pack", exist_ok=True)
with open(f"{LAB}/annotation_pack/windows.jsonl", "w") as f, open(f"{LAB}/annotation_pack/annotation_template.csv", "w", newline="") as g:
    cw = csv.writer(g); cw.writerow(["anon_id", "local_step", "label", "ans", "notes", "annotator"])
    for w in windows:
        tr = steps[w["traj_id"]]; aid = hashlib.sha1((str(rng.random()) + w["traj_id"]).encode()).hexdigest()[:10]
        S = tr["steps"]; ctx = [S[i]["text"] for i in range(max(0, w["start"] - 4), w["start"])]
        win = [S[i]["text"] for i in range(w["start"], w["end"])]
        f.write(json.dumps(dict(anon_id=aid, problem=problems()[tr["task_id"]], context=ctx, steps=win)) + "\n")
        for j in range(W): cw.writerow([aid, j, "", "", "", ""])
        key[aid] = dict(w)
json.dump(key, open(f"{LAB}/audit_key_private.json", "w"), indent=1)
json.dump([{k: w[k] for k in ("traj_id", "start", "end")} for w in windows], open(f"{LAB}/data/audit_windows.json", "w"))
import collections
print(len(windows), collections.Counter((w["stratum"], w["config"]) for w in windows))

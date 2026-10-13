"""Audit item 6: 60 windows for BLIND HUMAN review. Selection uses detector candidates + the first CORRECT answer proposal (Qwen ANS/boxed graded offline; used ONLY to place windows, never shown), seed 888.
Strata (15 windows each): after_correct_repeating / after_correct_not_repeating / mid_reasoning_no_candidates / late_repeat_no_correct.
Pack hides model configuration, seed, trajectory id, position and every automatic label. Human review status: NOT COMPLETED until the template is filled."""
import sys, os, json, random, csv, hashlib, collections
sys.path.insert(0, os.path.dirname(__file__))
from states import *
rng = random.Random(888); W = 10; PER = 15; OUT = f"{LAB}/dur_study/human_pack_60"; os.makedirs(OUT, exist_ok=True)
LAB_Q = load_labels(f"{LAB}/data/labels_qwen14b_*.jsonl")
steps = {}
for l in open(f"{LAB}/data/steps_all.jsonl"):
    r = json.loads(l)
    if r["config"] in ("bf16", "awq3"): steps[r["traj_id"]] = r
pool = collections.defaultdict(list)
for tid, tr in steps.items():
    S = tr["steps"]; th = [s["step"] for s in S if s["phase"] == "think"]; n = len(th)
    if n < 24 or (tid, 0) not in LAB_Q: continue
    flag = [bool(S[i]["cand_text"] or S[i]["cand_intra"]) for i in range(n)]
    ann = annotate(tr, LAB_Q); ca = next((p for p in ann["props"] if p["correct"]), None); first_c = ca["step"] if ca else None
    if first_c is not None:
        for s in range(first_c + 1, n - W + 1, 5):    # any window starting after the first correct answer proposal
            c = sum(flag[s:s + W]); pool[("after_correct_repeating" if c >= 4 else "after_correct_not_repeating" if c <= 2 else None, tr["config"])].append((tid, s))
    for s in range(4, n - W, 10):
        c = sum(flag[s:s + W]); early = first_c is None or s + W <= first_c
        if c == 0 and early: pool[("mid_reasoning_no_candidates", tr["config"])].append((tid, s))
        if c >= 5 and first_c is None: pool[("late_repeat_no_correct", tr["config"])].append((tid, s))
win, avail = [], {}
for st in ("after_correct_repeating", "after_correct_not_repeating", "mid_reasoning_no_candidates", "late_repeat_no_correct"):
    quota = {"bf16": PER // 2 + PER % 2, "awq3": PER // 2}; got = []
    for cfg in ("bf16", "awq3"):
        lst = pool.get((st, cfg), []); avail[(st, cfg)] = len(lst); rng.shuffle(lst); seenq = set(); k = 0
        for tid, s in lst:
            q = steps[tid]["task_id"]
            if q in seenq: continue
            seenq.add(q); got.append(dict(traj_id=tid, start=s, end=s + W, stratum=st, config=cfg)); k += 1
            if k == quota[cfg]: break
    for cfg_fill in ("awq3", "bf16"):          # fill shortfalls from the other configuration; recorded
        while len([g for g in got if True]) < PER:
            used = {(g["traj_id"], g["start"]) for g in got}; usedq = {steps[g["traj_id"]]["task_id"] for g in got}
            c = [(t, s) for cfg in (cfg_fill,) for (t, s) in pool.get((st, cfg), []) if (t, s) not in used and steps[t]["task_id"] not in usedq]
            if not c: break
            t, s = rng.choice(c); got.append(dict(traj_id=t, start=s, end=s + W, stratum=st, config=steps[t]["config"], filled=True))
    win += got
rng.shuffle(win); key = {}
with open(f"{OUT}/windows.jsonl", "w") as f, open(f"{OUT}/annotation_template.csv", "w", newline="") as g:
    cw = csv.writer(g); cw.writerow(["anon_id", "local_step", "label_GLAU", "states_an_answer_YN", "stated_answer_correct_YNNA", "notes", "annotator_id"])
    for w in win:
        tr = steps[w["traj_id"]]; S = tr["steps"]; aid = hashlib.sha1((str(rng.random()) + w["traj_id"]).encode()).hexdigest()[:10]
        f.write(json.dumps(dict(anon_id=aid, problem=problems()[tr["task_id"]], reference_solution_for_checking_answers_only=tr["gold"], context=[S[i]["text"] for i in range(max(0, w["start"] - 4), w["start"])],
                                steps=[S[i]["text"] for i in range(w["start"], w["end"])])) + "\n")
        for j in range(W): cw.writerow([aid, j, "", "", "", "", ""])
        key[aid] = w
json.dump(key, open(f"{LAB}/dur_study/human_pack_60_KEY_private.json", "w"), indent=1)
json.dump(dict(n_windows=len(win), n_steps=len(win) * W, n_human_reviewed_windows=0, status="NOT COMPLETED - no human has labelled these windows. Any model labels computed later are AUTOMATIC and must not be called human.",
               availability={f"{k[0]}|{k[1]}": v for k, v in avail.items()}, strata=dict(collections.Counter((w["stratum"], w["config"]) for w in win).items().__iter__().__class__ and {f"{a}|{b}": c for (a, b), c in collections.Counter((w["stratum"], w["config"]) for w in win).items()})),
          open(f"{OUT}/human_review_status.json", "w"), indent=1)
print(len(win), collections.Counter((w["stratum"], w["config"]) for w in win), {f"{k[0]}|{k[1]}": v for k, v in avail.items()})

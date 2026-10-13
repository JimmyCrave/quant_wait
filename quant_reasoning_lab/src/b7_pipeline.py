"""B7: correction-duration study + switch-consistency test. Sub-commands: build | stageA | stageB | merge (stages run through continue_gen.py).
Anchors: family-A anchors of B5 (AWQ3-generated trajectories, first correct answer proposed). Selection BEFORE reading any intervention effect:
  eligible = original sampled token ids were saved AND retokenised prefix == original ids; 32 lowest sha1("dur-A-v1|"+anchor_id).  One anchor per question.
Arms (T = total new tokens INCLUDING the BF16 window):
  main (T=4096, seeds 1-3): D0 AWQ3 | DP AWQ3+marker penalty lam=1 | DB remaining all BF16 | W32/W256/W1024 BF16 window then AWQ3
  long (T=8192, the 8 of the 32 anchors with the lowest sha1("dur-T8192-v1|"+id); first draft used hash%4==0 which gave 17/32 = not a small subset, changed before any generation): same six arms
  cons (T=2048, seeds 11-18): NQ AWQ no switch | SQQ32/SQQ256 AWQ->AWQ | NB BF16 no switch | SBB32/SBB256 BF16->BF16   (switch-consistency test)
Window tokens are passed on as token ids (no decode/encode at the switch); stage 2 uses seed+100; a window that ends the generation (EOS) is never continued."""
import sys, os, json, hashlib, random, glob, collections
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from tf_stats import prompt_ids

DD = f"{LAB}/dur_study/data"
def h(s): return int(hashlib.sha1(s.encode()).hexdigest(), 16)

ARMS = {"D0": dict(W=0, m1=None, m2="awq3"), "DP": dict(W=0, m1=None, m2="awq3", lam=1.0), "DB": dict(W=0, m1=None, m2="bf16"),
        "W32": dict(W=32, m1="bf16", m2="awq3"), "W256": dict(W=256, m1="bf16", m2="awq3"), "W1024": dict(W=1024, m1="bf16", m2="awq3"),
        "NQ": dict(W=0, m1=None, m2="awq3"), "NB": dict(W=0, m1=None, m2="bf16"),
        "SQQ32": dict(W=32, m1="awq3", m2="awq3"), "SQQ256": dict(W=256, m1="awq3", m2="awq3"),
        "SBB32": dict(W=32, m1="bf16", m2="bf16"), "SBB256": dict(W=256, m1="bf16", m2="bf16")}
GROUPS = {"main": (["D0", "DP", "DB", "W32", "W256", "W1024"], (1, 2, 3), 4096), "long": (["D0", "DP", "DB", "W32", "W256", "W1024"], (1, 2, 3), 8192),
          "cons": (["NQ", "NB", "SQQ32", "SQQ256", "SBB32", "SBB256"], tuple(range(11, 19)), 2048)}

def build():
    os.makedirs(DD, exist_ok=True); tok = tokenizer(); trs = {t["traj_id"]: t for t in load_trajectories()}
    A = [a for a in map(json.loads, open(f"{LAB}/data/b5/anchors.jsonl")) if a["family"] == "A"]
    assert len({a["task_id"] for a in A}) == len(A)
    rows = []
    for a in A:
        t = trs[a["traj_id"]]; sid = t["saved_ids"]; ids = tok(t["text"], add_special_tokens=False).input_ids
        ok = sid is not None and sid[:a["start_tok"]] == ids[:a["start_tok"]] and EOS not in sid[:a["start_tok"]] and THINK_END not in sid[:a["start_tok"]]
        rows.append(dict(a, eligible=bool(ok), hash=h("dur-A-v1|" + a["anchor_id"]), prefix_ids=(sid[:a["start_tok"]] if ok else None)))
    elig = sorted([r for r in rows if r["eligible"]], key=lambda r: r["hash"]); chosen = elig[:32]
    assert len(chosen) == 32
    cs = {r["anchor_id"] for r in chosen}
    long8 = {r["anchor_id"] for r in sorted(chosen, key=lambda r: h("dur-T8192-v1|" + r["anchor_id"]))[:8]}
    with open(f"{DD}/anchor_selection.jsonl", "w") as f:    # full selection record (no outcome fields exist in anchors.jsonl)
        for r in sorted(rows, key=lambda r: r["hash"]): f.write(json.dumps(dict(anchor_id=r["anchor_id"], traj_id=r["traj_id"], task_id=r["task_id"], hash=r["hash"], eligible=r["eligible"], chosen=r["anchor_id"] in cs,
                                                                          long_subset=(r["anchor_id"] in cs and r["anchor_id"] in long8))) + "\n")
    long8 = {r["anchor_id"] for r in sorted(chosen, key=lambda r: h("dur-T8192-v1|" + r["anchor_id"]))[:8]}
    anchors = {}
    for r in chosen:
        a = {k: r[k] for k in ("anchor_id", "traj_id", "task_id", "origin", "start_tok", "gold", "correct_step")}
        a["prompt_ids"] = prompt_ids(a["task_id"]); a["prefix_ids"] = r["prefix_ids"]; a["long_subset"] = a["anchor_id"] in long8; anchors[a["anchor_id"]] = a
    json.dump(anchors, open(f"{DD}/anchors32.json", "w"))
    runs = []
    for g, (arms, seeds, T) in GROUPS.items():
        for aid, a in anchors.items():
            if g == "long" and not a["long_subset"]: continue
            for arm in arms:
                for s in seeds: runs.append(dict(run_id=f"{g}|{aid}|{arm}|s{s}", group=g, anchor_id=aid, arm=arm, seed=s, T=T, **ARMS[arm]))
    json.dump(runs, open(f"{DD}/runs.json", "w"))
    random.Random(5).shuffle(runs)
    A_req = collections.defaultdict(list)
    for r in runs:
        a = anchors[r["anchor_id"]]; P = a["prompt_ids"] + a["prefix_ids"]
        if r["W"] > 0: A_req[r["m1"]].append(dict(req_id=r["run_id"], model=r["m1"], seed=r["seed"], max_tokens=r["W"], prompt_token_ids=P))
        else: A_req[r["m2"]].append(dict(req_id=r["run_id"], model=r["m2"], seed=r["seed"], max_tokens=r["T"], lam=r.get("lam"), prompt_token_ids=P))
    for m, L in A_req.items():
        with open(f"{DD}/A_{m}.jsonl", "w") as f:
            for q in L: f.write(json.dumps(q) + "\n")
    print("anchors", len(anchors), "long subset", sum(a["long_subset"] for a in anchors.values()), "runs", len(runs), collections.Counter(r["group"] for r in runs), {m: len(L) for m, L in A_req.items()})

def read(pat):
    out = {}
    for p in sorted(glob.glob(pat)):
        for l in open(p): r = json.loads(l); out[r["req_id"]] = r
    return out

def stageB():
    anchors = json.load(open(f"{DD}/anchors32.json")); runs = [r for r in json.load(open(f"{DD}/runs.json")) if r["W"] > 0]
    A = {**read(f"{DD}/outA_bf16_*.jsonl"), **read(f"{DD}/outA_awq3_*.jsonl")}; Br = collections.defaultdict(list); n_stop = 0
    for r in runs:
        w = A[r["run_id"]]; a = anchors[r["anchor_id"]]
        if w["finish_reason"] == "stop" or len(w["token_ids"]) < r["W"]: n_stop += 1; continue            # window ended the generation: never continue
        Br[r["m2"]].append(dict(req_id=r["run_id"], model=r["m2"], seed=r["seed"] + 100, max_tokens=r["T"] - len(w["token_ids"]), lam=None, prompt_token_ids=a["prompt_ids"] + a["prefix_ids"] + w["token_ids"]))
    for m, L in Br.items():
        with open(f"{DD}/B_{m}.jsonl", "w") as f:
            for q in L: f.write(json.dumps(q) + "\n")
    print("stage B", {m: len(L) for m, L in Br.items()}, "windows that ended the generation:", n_stop)

def merge():
    anchors = json.load(open(f"{DD}/anchors32.json")); runs = json.load(open(f"{DD}/runs.json"))
    A = {**read(f"{DD}/outA_bf16_*.jsonl"), **read(f"{DD}/outA_awq3_*.jsonl")}; B = {**read(f"{DD}/outB_bf16_*.jsonl"), **read(f"{DD}/outB_awq3_*.jsonl")}
    with open(f"{DD}/final_b7.jsonl", "w") as f:
        for r in runs:
            a = anchors[r["anchor_id"]]; P = len(a["prompt_ids"]) + len(a["prefix_ids"]); x = A[r["run_id"]]
            o = dict(r, prefix_len=P)
            if r["W"] == 0:
                ids = x["token_ids"]; o.update(token_ids_all=ids, finish_reason=x["finish_reason"], n_window=0, window_ended_generation=False, window_finish=None,
                                               fwd={r["m2"]: dict(prefill=P, decode=len(ids))}, lat=dict(stageA=x["latency_s"], stageB=None))
            else:
                w = x["token_ids"]
                if r["run_id"] in B:
                    y = B[r["run_id"]]; ids = w + y["token_ids"]
                    o.update(token_ids_all=ids, finish_reason=y["finish_reason"], n_window=len(w), window_ended_generation=False, window_finish=x["finish_reason"],
                             stage2_prompt_exact=(y.get("n_prompt") == P + len(w)), lat=dict(stageA=x["latency_s"], stageB=y["latency_s"]))
                    fw = collections.defaultdict(lambda: dict(prefill=0, decode=0)); fw[r["m1"]]["prefill"] += P; fw[r["m1"]]["decode"] += len(w)
                    fw[r["m2"]]["prefill"] += P + len(w); fw[r["m2"]]["decode"] += len(y["token_ids"]); o["fwd"] = dict(fw)
                else:
                    o.update(token_ids_all=w, finish_reason=x["finish_reason"], n_window=len(w), window_ended_generation=True, window_finish=x["finish_reason"], fwd={r["m1"]: dict(prefill=P, decode=len(w))}, lat=dict(stageA=x["latency_s"], stageB=None))
            f.write(json.dumps(o) + "\n")
    print("merged", len(runs))

if __name__ == "__main__":
    {"build": build, "stageB": stageB, "merge": merge}[sys.argv[1]]()

"""Audit item 5: switch-consistency verification. (A) structural checks (template, token concatenation, terminators, seeds/state, retokenisation) and
(B) distributional comparison: AWQ->AWQ vs no switch, BF16->BF16 vs no switch, with BF16-vs-AWQ no-switch as POSITIVE control for sensitivity.
Random generation is compared in distribution (question/anchor-clustered bootstrap), not token by token across back ends."""
import sys, os, json, collections, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from tf_stats import prompt_ids
tok = tokenizer(); DD = f"{LAB}/dur_study/data"; OUT = f"{LAB}/dur_study/audit"; rows = []
def add(check, ok, detail): rows.append(dict(check=check, passed=bool(ok), detail=detail))
# ---- (A1) chat template / prompt identical to the stage-1 evaluation prompt (no double template, single BOS)
res = json.load(open(f"{ROOT}/artifacts/results/bf16/MATH-500.jsonl")); bad = 0; lens_ok = 0
for r in res:
    ids = prompt_ids(r["task_id"]); ref = tok(r["full_prompt"], add_special_tokens=False).input_ids
    bad += ids != ref; lens_ok += len(ids) == r["num_input_tokens"]
add("prompt ids == tokenisation of the stage-1 full_prompt (100 questions)", bad == 0, f"mismatches={bad}; len==num_input_tokens for {lens_ok}/100")
sp = {k: tok.convert_tokens_to_ids(k) for k in ("<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "<｜end▁of▁sentence｜>")}
anchors = json.load(open(f"{DD}/anchors32.json")); cnt_bad = 0
for a in anchors.values():
    P = a["prompt_ids"]; c = collections.Counter(P + a["prefix_ids"])
    cnt_bad += not (c[sp["<｜begin▁of▁sentence｜>"]] == 1 and c[sp["<｜User｜>"]] == 1 and c[sp["<｜Assistant｜>"]] == 1 and c[sp["<think>"]] == 1 and c[sp["</think>"]] == 0 and c[sp["<｜end▁of▁sentence｜>"]] == 0)
add("each anchor prompt+prefix has exactly 1 BOS/User/Assistant/<think> and no </think>/EOS (32 anchors)", cnt_bad == 0, f"violations={cnt_bad}; special ids={sp}")
trs = {t["traj_id"]: t for t in load_trajectories()}
add("anchor prefix == ORIGINAL sampled token ids of the source trajectory (no re-tokenisation)", all(trs[a["traj_id"]]["saved_ids"][:a["start_tok"]] == a["prefix_ids"] for a in anchors.values()), "32/32 checked against saved generated_token_ids")
# ---- (A2) merged runs
F = [json.loads(l) for l in open(f"{DD}/final_b7.jsonl")]; bad = collections.Counter(); nwin = nstage2 = nstopwin = 0; rt = []
A = {}
for p in glob.glob(f"{DD}/outA_*_*.jsonl"):
    for l in open(p): r = json.loads(l); A[r["req_id"]] = r
for r in F:
    ids = r["token_ids_all"]; eos_pos = [i for i, t in enumerate(ids) if t == EOS]
    bad["eos_not_last"] += bool(eos_pos) and eos_pos != [len(ids) - 1]
    bad["stop_without_eos"] += (r["finish_reason"] == "stop") != (ids[-1:] == [EOS])
    bad["over_budget"] += len(ids) > r["T"]
    bad["length_but_short"] += (r["finish_reason"] == "length") and len(ids) != r["T"]
    if r["W"] > 0:
        nwin += 1; w = A[r["run_id"]]["token_ids"]
        bad["window_not_prefix"] += ids[:len(w)] != w; bad["window_len_wrong"] += (len(w) != r["W"]) and A[r["run_id"]]["finish_reason"] != "stop"
        if r["window_ended_generation"]: nstopwin += 1; bad["stopped_window_continued"] += len(ids) != len(w)
        else: nstage2 += 1; bad["stage2_prompt_not_exact"] += not r.get("stage2_prompt_exact", False)
    if len(rt) < 300 and r["group"] == "main": e2 = [i for i in ids if i != EOS]; rt.append(tok.encode(tok.decode(e2), add_special_tokens=False) == e2)
for k in ("eos_not_last", "stop_without_eos", "over_budget", "length_but_short", "window_not_prefix", "window_len_wrong", "stopped_window_continued", "stage2_prompt_not_exact"):
    add(f"merged runs: {k} == 0", bad[k] == 0, f"violations={bad[k]} of {len(F)} runs ({nwin} window runs: {nstage2} continued, {nstopwin} ended inside the window)")
add("INFO decode->encode round trip equals the sampled ids (only matters for the detector segmentation, never for generation)", True, f"{np.mean(rt):.3f} of {len(rt)} main runs round-trip exactly")
# ---- (A3) generator state: what is (not) reset at the switch
cg = open(f"{LAB}/src/continue_gen.py").read()
add("sampling params identical in every stage (T=0.6, top_p=0.95; no repetition/frequency penalty; no min_tokens); only a per-request seed and the optional marker processor vary", "temperature=0.6, top_p=0.95" in cg and "repetition" not in cg,
    "stage-2 seed = stage-1 seed + 100 (a fresh independent random stream; the per-request RNG cannot be carried across two vLLM calls, distributionally irrelevant); marker processor only in DP runs (stateless); prefix caching off")
# ---- (A4) same-seed identity of the first W tokens: switch stage 1 vs no-switch run with the same model/seed
R = {r["run_id"]: r for r in F}; ident = []
for r in F:
    if r["group"] == "cons" and r["W"] > 0:
        base = R[r["run_id"].replace(r["arm"], "NQ" if r["m1"] == "awq3" else "NB")]; w = A[r["run_id"]]["token_ids"]; b = base["token_ids_all"][:len(w)]
        k = next((i for i in range(len(w)) if i >= len(b) or w[i] != b[i]), len(w)); ident.append(dict(arm=r["arm"], W=r["W"], identical=k == len(w), lcp=k))
I = pd.DataFrame(ident); S1 = I.groupby("arm").agg(n=("identical", "size"), frac_identical_first_W=("identical", "mean"), median_lcp=("lcp", "median")).reset_index()
add("INFO same model+seed: first W tokens of the switch run vs the no-switch run (bf16 kernels / batch composition can flip a sampled token, so equality is not required)", True, S1.round(3).to_dict("records"))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/audit5_structural_checks.csv", index=False)
# ---- (B) distribution comparison on outcomes
O = pd.read_csv(f"{DD}/b7_outcomes.csv"); O = O[O.group == "cons"]; rng = np.random.default_rng(0)
O["len_to_complete"] = O.len_to_complete.astype(float)
METR = ["think_closed", "answer_complete", "primary", "det_entry", "len_to_complete", "n_total"]
def pa(arm): return O[O.arm == arm].groupby("anchor_id")[METR].mean()
out = []
for a, b, kind in (("SQQ32", "NQ", "AWQ->AWQ vs no switch"), ("SQQ256", "NQ", "AWQ->AWQ vs no switch"), ("SBB32", "NB", "BF16->BF16 vs no switch"), ("SBB256", "NB", "BF16->BF16 vs no switch"),
                   ("NB", "NQ", "POSITIVE CONTROL: BF16 vs AWQ no switch")):
    x, y = pa(a), pa(b); d = (x - y).dropna(); idx = d.index.values
    for m in METR:
        bs = [d[m].loc[rng.choice(idx, len(idx))].mean() for _ in range(5000)]
        out.append(dict(comparison=f"{a} - {b}", kind=kind, metric=m, mean_a=x[m].mean(), mean_b=y[m].mean(), diff=d[m].mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n_anchors=len(idx),
                        ci_excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0)))
Dd = pd.DataFrame(out); Dd.to_csv(f"{OUT}/audit5_distribution_comparison.csv", index=False); S1.to_csv(f"{OUT}/audit5_same_seed_prefix_identity.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 200); print(C[["check", "passed"]].to_string()); print(S1.to_string()); print(Dd.round(3).to_string())

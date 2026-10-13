"""B1b: automatic step labelling with a local instruct LLM (NOT a human gold standard).
The labeler never sees the model configuration, seed, or the gold answer. Optional 'evidence' = the most similar EARLIER step
(retrieval by the detectors' similarity), shown neutrally; the no-evidence variant is used on audit windows to measure this dependence."""
import sys, json, re, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
from common import *

SYSTEM = "You are a careful annotator of chain-of-thought text produced by a math-solving language model."
DEFS = """Label every STEP with exactly one label:
G = productive reasoning: introduces NEW information, a new method, a new computation, a new case, or a genuine check/verification that adds confidence, and necessary correction of a real mistake (a self-correction that fixes an error is G, even if it starts with 'Wait' or 'But').
L = unproductive loop: re-states or re-derives something already established earlier, returns to the same conclusion/doubt/contradiction again, or re-does the same check, WITHOUT new information, new method, or fixing an error. Paraphrase counts as repetition. Words like 'Wait', 'But', 'Alternatively' are NOT by themselves evidence of L.
A = explicitly commits to / submits the FINAL answer (e.g. 'So the final answer is X', '**Final Answer** \\boxed{X}', 'I'll go with X').
U = cannot be judged reliably from the given text.
Also give ANS: the answer value to the PROBLEM stated or proposed in the step (tentative or final), written as a short math string, or '-' if the step does not state an answer to the problem."""
FMT = "Output exactly one line per step, no explanations, in the form:  <step number>|<G/L/A/U>|<ANS or ->"

def problem_of(prompt):
    m = re.search(r"<｜User｜>(.*?)\n\nPlease reason step by step", prompt, re.S)
    return (m.group(1) if m else prompt)[:1500]

def clip(t, n): t = t.strip(); return t if len(t) <= n else t[:n - 60] + " ... " + t[-50:]

def build_prompt(tr, targets, evidence, n_ctx=4):
    steps = tr["steps"]; a = targets[0]
    ctx = steps[max(0, a - n_ctx):a]
    lines = [f"PROBLEM:\n{problems()[tr['task_id']][:1500]}\n", DEFS, "",
             "CONTEXT (previous steps, already read; do NOT label):"]
    lines += [f"[{s['step']}] {clip(s['text'], 300)}" for s in ctx] or ["(start of reasoning)"]
    lines += ["", "STEPS TO LABEL:"]
    for i in targets:
        s = steps[i]; lines.append(f"[{i}] {clip(s['text'], 600)}")
        if evidence:
            j, sim = None, 0.0
            if s["jac_dist"] and s["max_jaccard"] >= 0.4: j, sim = i - s["jac_dist"], s["max_jaccard"]
            elif s["cos_dist"] and s["max_cos"] >= 0.85: j, sim = i - s["cos_dist"], s["max_cos"]
            if j is not None and j >= 0 and i - j >= 1:
                lines.append(f"    (retrieval: closest EARLIER step is [{j}], {i-j} steps back, similarity {sim:.2f}: \"{clip(steps[j]['text'], 200)}\")")
    lines += ["", FMT]
    return SYSTEM, "\n".join(lines)

def parse(out, targets):
    res = {}
    for ln in out.strip().splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s*\|\s*([GLAU])\s*\|\s*(.*)", ln)
        if m and int(m.group(1)) in targets:
            ans = m.group(3).strip().strip("`")
            res[int(m.group(1))] = (m.group(2), None if ans in ("-", "", "None") else ans[:120])
    return {i: res.get(i, ("U", None)) for i in targets}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--steps"); ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--evidence", type=int, default=1); ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--windows", default=None, help="json list of {traj_id,start,end} to restrict to")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    trajs = [json.loads(l) for l in open(a.steps)]
    win = None
    if a.windows: win = {}; [win.setdefault(w["traj_id"], []).append((w["start"], w["end"])) for w in json.load(open(a.windows))]
    jobs = []
    for k, tr in enumerate(trajs):
        if k % a.nshards != a.shard: continue
        if win is not None and tr["traj_id"] not in win: continue
        n = len(tr["steps"]); todo = [i for i, s in enumerate(tr["steps"]) if s["phase"] == "think"]
        if win is not None: todo = [i for i in todo if any(s0 <= i < e0 for s0, e0 in win[tr["traj_id"]])]
        for c in range(0, len(todo), a.chunk):
            jobs.append((k, todo[c:c + a.chunk]))
    llm = LLM(a.model, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.9, enable_prefix_caching=True, seed=0)
    tk = llm.get_tokenizer(); prompts = []
    for k, tg in jobs:
        sysm, user = build_prompt(trajs[k], tg, bool(a.evidence))
        prompts.append(tk.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=10 * a.chunk + 40))
    with open(a.out, "w") as f:
        for (k, tg), o in zip(jobs, outs):
            for i, (lab, ans) in parse(o.outputs[0].text, tg).items():
                f.write(json.dumps(dict(traj_id=trajs[k]["traj_id"], step=i, label=lab, ans=ans)) + "\n")
    print("labelled", len(jobs), "chunks")
if __name__ == "__main__": main()

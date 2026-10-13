"""Turn merged B7 runs into pseudo-trajectories (decoded prefix + decoded continuation, EOS stripped), segment them and run the candidate detectors (build_steps.run).
Used ONLY for the detector-based repetition flags / positions. Generation itself never decodes/re-encodes (token ids are passed through)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import build_steps
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
tok = tokenizer(); anchors = json.load(open(f"{LAB}/dur_study/data/anchors32.json")); trajs = []
for l in open(f"{LAB}/dur_study/data/final_b7.jsonl"):
    r = json.loads(l); a = anchors[r["anchor_id"]]; pre = a["prefix_ids"]; ids = [i for i in r["token_ids_all"] if i != EOS]
    pt = tok.decode(pre); ct = tok.decode(ids)
    ex = dict(prefix_chars=len(pt), n_prefix_tokens=len(pre), n_cont_tokens=len(ids), cont_finish=r["finish_reason"], budget=r["T"], run_id=r["run_id"])
    trajs.append(dict(traj_id=r["run_id"], config=r["arm"], seed=r["seed"], task_id=a["task_id"], split=None, score=None, finish_reason=r["finish_reason"], n_gen=len(ids), gold=a["gold"], text=pt + ct, extra=ex))
print("pseudo-trajectories", len(trajs)); build_steps.run(trajs, f"{LAB}/dur_study/data/b7_steps.jsonl", dev)

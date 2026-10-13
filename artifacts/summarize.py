import json, os, statistics, sys
import numpy as np
root = "Quantized-Reasoning-Models/outputs/final"
groups = {"BF16": "bf16", "AWQ3 (numerical simulation)": "awq3", "AWQ3 + marker penalty lambda=1.0": "awq3_lambda1.0"}
out, per_q = {}, {}
for name, d in groups.items():
    p = f"{root}/{d}/MATH-500.jsonl"
    if not os.path.exists(p):
        out[name] = {"status": "NOT COMPLETED (no result file)"}; continue
    data = json.load(open(p))          # JSON array (despite .jsonl suffix)
    cfg = json.load(open(p.replace(".jsonl", ".run_config.json")))
    n = np.array([r["num_generated_tokens"] for r in data]); sc = np.array([r["score"] for r in data], float)
    ex = np.array([r["budget_exhausted"] for r in data])
    out[name] = {"status": "completed", "n_questions": len(data), "accuracy": float(sc.mean()),
                 "accuracy_se_binomial": float(np.sqrt(sc.mean() * (1 - sc.mean()) / len(sc))),
                 "mean_tokens": float(n.mean()), "median_tokens": float(np.median(n)), "p90_tokens": float(np.percentile(n, 90)),
                 "budget_exhausted_frac": float(ex.mean()), "budget_exhausted_n": int(ex.sum()),
                 "wall_time_s": cfg["wall_time_s"], "marker_lambda": cfg["marker_lambda"]}
    per_q[name] = {r["task_id"]: (r["score"], r["num_generated_tokens"]) for r in data}
ids = [set(v) for v in per_q.values()]
out["same_questions_across_groups"] = bool(ids) and all(i == ids[0] for i in ids)
json.dump(out, open("artifacts/results_summary.json", "w"), indent=2)
print(json.dumps(out, indent=2))

import json, sys, torch
from safetensors import safe_open
base = safe_open("modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors", "pt")
q = safe_open("outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1/model.safetensors", "pt")
rows, summary = [], {"quantized": 0, "unchanged": 0, "max_unique_per_group_quantized": 0}
for k in sorted(q.keys()):
    a, b = base.get_tensor(k).float(), q.get_tensor(k).float()
    changed = not torch.equal(a, b)
    entry = {"name": k, "shape": list(b.shape), "changed_vs_bf16": changed}
    if k.endswith(".weight") and b.ndim == 2 and ".layers." in k and changed:
        g = b.reshape(b.shape[0], -1, 128)
        # distinct values per (row, group of 128). 3-bit asym => <= 8
        mx = max(int(torch.unique(r).numel()) for r in g[:, torch.randperm(g.shape[1])[:1].item() if g.shape[1] > 1 else 0, :][:256])
        gs = g[:64]  # first 64 rows, all groups
        u = torch.stack([torch.tensor([torch.unique(gs[i, j]).numel() for j in range(gs.shape[1])]) for i in range(gs.shape[0])])
        entry["max_unique_per_group"] = int(u.max()); entry["rel_err"] = float((a - b).norm() / a.norm())
        summary["quantized"] += 1
        summary["max_unique_per_group_quantized"] = max(summary["max_unique_per_group_quantized"], int(u.max()))
    elif not changed:
        summary["unchanged"] += 1
    rows.append(entry)
print(json.dumps(summary))
for r in rows:
    if not r["changed_vs_bf16"] and ".layers." not in r["name"]: print("unchanged non-layer:", r["name"])
    if ".layers.0." in r["name"]: print(r)
json.dump({"summary": summary, "tensors": rows}, open("../artifacts/awq_weight_check.json", "w"), indent=1)

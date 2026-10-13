"""B1a: segment every trajectory into steps (char+token spans), run the two candidate-loop detectors.
Detectors only produce CANDIDATES; thresholds are fixed in configs/prereg.yaml."""
import sys, json, re, zlib, collections, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *

WORD = re.compile(r"\w+|[^\w\s]")
NUM = re.compile(r"\d+(?:\.\d+)?|\\[a-zA-Z]+|[a-zA-Z]\^?\d?")   # numbers, latex commands, symbols
def shingles(ws, n=5): return {" ".join(ws[i:i+n]) for i in range(max(0, len(ws)-n+1))}

def embed_all(texts, dev="cuda:0", bs=256):
    from transformers import AutoTokenizer, AutoModel
    p = f"{LAB}/models/bge-small-en-v1.5"
    tk, m = AutoTokenizer.from_pretrained(p), AutoModel.from_pretrained(p).to(dev).half().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tk(texts[i:i+bs], padding=True, truncation=True, max_length=192, return_tensors="pt").to(dev)
            e = m(**b).last_hidden_state[:, 0]; out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out).numpy()

def main(out_path):
    only = set(map(int, sys.argv[2].split(","))) if len(sys.argv) > 2 else None
    cfgs = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    run([t for t in load_trajectories() if (only is None or t["seed"] in only) and (cfgs is None or t["config"] in cfgs)], out_path)

def run(trajs, out_path, dev="cuda:0"):
    rows, all_texts = [], []
    for k, tr in enumerate(trajs):
        steps, ids, te = segment(tr["text"])
        for s in steps: s["traj_idx"] = k
        rows.append((tr, steps, ids, te)); all_texts += [s["text"] for s in steps]
        if k % 100 == 0: print("segmented", k, len(trajs), flush=True)
    print("embedding", len(all_texts), flush=True)
    E = embed_all(all_texts, dev); off = 0
    with open(out_path, "w") as f:
        for tr, steps, ids, te in rows:
            n = len(steps); e = E[off:off+n]; off += n
            idx = collections.defaultdict(set); sh = []; numsets = []
            sims = e @ e.T if n <= 4000 else None
            for i, s in enumerate(steps):
                ws = WORD.findall(s["text"].lower()); S = shingles(ws); sh.append(S)
                ns = set(NUM.findall(s["text"])); numsets.append(ns)
                best_j, best_d, best_c = 0.0, None, 0.0
                if len(ws) >= 8 and S:
                    cnt = collections.Counter(j for g in S for j in idx[g])
                    for j, c in cnt.items():
                        u = len(S) + len(sh[j]) - c; jac = c / u
                        if jac > best_j: best_j, best_d, best_c = jac, i - j, c / len(S)
                for g in S: idx[g].add(i)
                # semantic
                bc, bd = 0.0, None
                if len(ws) >= 8 and i >= 2:
                    row = sims[i, :i-1] if sims is not None else e[:i-1] @ e[i]; j = int(row.argmax()); bc, bd = float(row[j]), i - j
                seen = set().union(*numsets[max(0, i-30):i]) if i else set()
                novel = len(ns - seen)
                raw = s["text"].encode(); zr = len(zlib.compress(raw)) / max(1, len(raw))
                s.update(dict(max_jaccard=round(best_j, 3), jac_dist=best_d, max_contain=round(best_c, 3),
                              max_cos=round(bc, 3), cos_dist=bd, n_novel_symbols=novel, zlib_ratio=round(zr, 3),
                              cand_text=bool(best_j >= 0.6), cand_intra=bool(len(raw) >= 200 and zr < 0.25),
                              cand_sem=bool(bc >= 0.90 and novel == 0), n_words=len(ws)))
                s["cand"] = s["cand_text"] or s["cand_intra"] or s["cand_sem"]
            meta = {k: tr.get(k) for k in ("traj_id", "config", "seed", "task_id", "split", "score", "finish_reason", "n_gen", "gold")}
            meta["extra"] = tr.get("extra")
            meta.update(n_tokens_retok=len(ids), think_end_tok=te, retok_matches_saved=(None if tr.get("saved_ids") is None else
                        [t for t in tr["saved_ids"] if t != EOS] == ids), steps=steps)
            f.write(json.dumps(meta) + "\n")
    print("done")
if __name__ == "__main__": main(sys.argv[1])

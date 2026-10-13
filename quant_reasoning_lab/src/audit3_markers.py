"""Audit 3: the 49 penalised token ids vs the 50 entries of paper Appendix C (Table 8), plus the frequency of space-free variants (not penalised)."""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
sys.path.insert(0, REPO)
tok = tokenizer()
PAPER = ("perhaps maybe wait Wait actually hold Hmm hmm Alternatively alternatively However however instead Instead But but though although yet rather unless otherwise "
         "nonetheless nevertheless regardless still anyway Or or either whether uncertain unsure possibly might could another different reconsider rethink backtrack retry recheck "
         "revisit doubt confused wrong mistake error incorrect").split()
assert len(PAPER) == 50
used = json.load(open(f"{REPO}/penalty/qwen_marker_token_ids.json"))["token_ids"]          # what the penalty processor actually uses
rows = []
for w in PAPER:
    sp = tok.encode(" " + w, add_special_tokens=False); nsp = tok.encode(w, add_special_tokens=False)
    rows.append(dict(paper_entry="_" + w, string_with_space=" " + w, ids_with_space=sp, pieces_with_space=tok.convert_ids_to_tokens(sp), single_token=len(sp) == 1,
                     id_used_by_penalty=used.get(" " + w), roundtrip_ok=(len(sp) == 1 and tok.decode(sp) == " " + w),
                     space_free_ids=nsp, space_free_single=len(nsp) == 1, space_free_id=(nsp[0] if len(nsp) == 1 else None)))
M = pd.DataFrame(rows)
assert set(M.id_used_by_penalty.dropna().astype(int)) == set(used.values()) and len(set(used.values())) == 49 and len(used) == 49
# frequency in sampled trajectories (think part only), per 1000 generated tokens
pen_ids = {int(i) for i in used.values()}; free = {int(r.space_free_id): r.paper_entry[1:] for r in M.itertuples() if r.space_free_single and r.space_free_id is not None}
free_only = {i: w for i, w in free.items() if i not in pen_ids}
cnt = collections.Counter(); ntok = collections.Counter(); id2s = {v: k for k, v in used.items()}
for t in load_trajectories():
    if t["config"] not in ("bf16", "awq3"): continue
    c0 = t["config"]
    ids = t["saved_ids"] if t["saved_ids"] is not None else tok(t["text"], add_special_tokens=False).input_ids
    ids = [i for i in ids if i != EOS]; te = ids.index(THINK_END) if THINK_END in ids else len(ids); ids = ids[:te]
    ntok[c0] += len(ids)
    for i, n in collections.Counter(ids).items():
        if i in pen_ids: cnt[(c0, "pen_total")] += n; cnt[(c0, "pen", id2s[i])] += n
        if i in free_only: cnt[(c0, "free_total")] += n; cnt[(c0, "free", free_only[i])] += n
    cnt[(c0, "recheck2")] += sum(1 for k in range(len(ids) - 1) if ids[k] == 312 and ids[k + 1] == 2028)
for c0 in ("bf16", "awq3"):
    M[f"pen_per1k_{c0}"] = [1000 * cnt[(c0, "pen", r.string_with_space)] / ntok[c0] if r.single_token else np.nan for r in M.itertuples()]
    M[f"free_per1k_{c0}"] = [1000 * cnt[(c0, "free", r.paper_entry[1:])] / ntok[c0] if (r.space_free_single and r.space_free_id not in pen_ids) else np.nan for r in M.itertuples()]
M.to_csv("dur_study/audit/audit3_marker_mapping.csv", index=False)
S = pd.DataFrame([dict(config=c, think_tokens=ntok[c], penalised_markers_per1k=1000 * cnt[(c, "pen_total")] / ntok[c], space_free_variants_per1k=1000 * cnt[(c, "free_total")] / ntok[c],
                       recheck_two_token_occurrences=cnt[(c, "recheck2")]) for c in ("bf16", "awq3")]); S.to_csv("dur_study/audit/audit3_marker_frequency.csv", index=False)
pd.set_option("display.width", 250)
print("single-token:", int(M.single_token.sum()), "of 50; not single:", M[~M.single_token][["paper_entry", "ids_with_space", "pieces_with_space"]].values.tolist())
print("space-free single-token:", int(M.space_free_single.sum()), "; space-free ids that collide with penalised ids:", [w for i, w in free.items() if i in pen_ids])
print(S.round(3).to_string())
print(M[["paper_entry", "id_used_by_penalty", "space_free_id", "pen_per1k_bf16", "pen_per1k_awq3", "free_per1k_bf16", "free_per1k_awq3"]].round(3).to_string())

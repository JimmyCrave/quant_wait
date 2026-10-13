"""B4 pass 2 summary: which vocabulary items gain / lose probability mass under the quantized model (mean per position), per origin x window kind, raw vs sampling distribution."""
import sys, os, glob, json, collections
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
tok = tokenizer(); acc = collections.defaultdict(lambda: dict(raw=0, samp=0, base=0, n=0))
mk = set(json.load(open(f"{REPO}/penalty/qwen_marker_token_ids.json"))["token_ids"].values())
for p in glob.glob(f"{LAB}/data/b4_direction_sh*.pt"):
    for k, v in torch.load(p).items():
        a = acc[k]; a["raw"] = a["raw"] + v["raw"]; a["samp"] = a["samp"] + v["samp"]; a["base"] = a["base"] + v["base_raw"]; a["n"] += v["n"]
rows = []; summ = []
for k, a in sorted(acc.items()):
    origin, kind = k.split("|"); n = a["n"]
    for dist in ("raw", "samp"):
        d = (a[dist] / n).numpy(); base = (a["base"] / n).numpy()
        top_up, top_dn = np.argsort(-d)[:15], np.argsort(d)[:15]
        mk_ids = np.array(sorted(mk))
        summ.append(dict(origin=origin, kind=kind, dist=dist, n_positions=n, total_abs_mass_moved=float(np.abs(d).sum() / 2), marker_mass_change=float(d[mk_ids].sum()), marker_mass_bf16=float(base[mk_ids].sum()),
                         thinkend_change=float(d[THINK_END]), eos_change=float(d[EOS]), gain_top15_mass=float(d[top_up].sum()), loss_top15_mass=float(d[top_dn].sum())))
        for sign, ids in (("gain", top_up), ("loss", top_dn)):
            for r, i in enumerate(ids): rows.append(dict(origin=origin, kind=kind, dist=dist, sign=sign, rank=r, token=tok.convert_ids_to_tokens(int(i)), token_id=int(i), mean_delta_p=float(d[i]), mean_bf16_p=float(base[i]), is_marker=int(i) in mk))
pd.DataFrame(rows).to_csv(f"{LAB}/results/b4_direction_top_tokens.csv", index=False); S = pd.DataFrame(summ); S.to_csv(f"{LAB}/results/b4_direction_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 200); print(S.round(5).to_string())
T = pd.DataFrame(rows); print(T[(T.origin == "awq3") & (T.kind == "in_loop") & (T.dist == "raw")][["sign", "rank", "token", "mean_delta_p", "mean_bf16_p"]].round(5).head(30).to_string())

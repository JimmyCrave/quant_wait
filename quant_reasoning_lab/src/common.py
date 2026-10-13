import json, os, re, hashlib, random, functools
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAB = os.path.join(ROOT, "quant_reasoning_lab")
REPO = os.path.join(ROOT, "Quantized-Reasoning-Models")
BF16 = os.path.join(REPO, "modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B")
AWQ3 = os.path.join(REPO, "outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1")
THINK_END, EOS = 151649, 151643
MAXCH = 700
MINCH = 150

@functools.lru_cache(None)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BF16)

def question_split():
    ids = sorted({r["task_id"] for r in json.load(open(f"{ROOT}/artifacts/results/bf16/MATH-500.jsonl"))})
    random.Random(20261011).shuffle(ids)
    return {t: ("dev" if i < 50 else "test") for i, t in enumerate(ids)}

def load_trajectories():
    """Unified list of trajectories. config in {bf16,awq3,awq3_pen}; seed 42 = stage-1 runs, 43-45 = replicates."""
    out = []
    srcs = [("bf16", 42, f"{ROOT}/artifacts/results/bf16"), ("awq3", 42, f"{ROOT}/artifacts/results/awq3"),
            ("awq3_pen", 42, f"{ROOT}/artifacts/results/awq3_lambda1.0")]
    for s in (43, 44, 45):
        for c in ("bf16", "awq3"):
            srcs.append((c, s, f"{REPO}/outputs/replicates/{c}_s{s}"))
    split = question_split()
    for cfg, seed, d in srcs:
        p = f"{d}/MATH-500.jsonl"
        if not os.path.exists(p): continue
        for r in json.load(open(p)):
            out.append(dict(traj_id=f"{cfg}_s{seed}_{r['task_id']}", config=cfg, seed=seed, task_id=r["task_id"],
                            split=split[r["task_id"]], gold=r["gold"][0], text=r["generated_text"], score=r["score"],
                            finish_reason=r["finish_reason"], n_gen=r["num_generated_tokens"],
                            saved_ids=r.get("generated_token_ids"), prompt=r["full_prompt"]))
    return out

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(])")
def split_paragraph(par, c0):
    """yield (c0,c1) spans for chunks <= MAXCH chars within paragraph starting at char c0"""
    if len(par) <= MAXCH:
        return [(c0, c0 + len(par))]
    pieces, pos, cur0 = [], 0, 0
    sents, last = [], 0
    for m in _SENT.finditer(par):
        sents.append((last, m.start())); last = m.end()
    sents.append((last, len(par)))
    cs, ce = None, None
    for a, b in sents:
        if cs is None: cs, ce = a, b
        elif b - cs <= MAXCH: ce = b
        else:
            pieces.append((c0 + cs, c0 + ce)); cs, ce = a, b
    pieces.append((c0 + cs, c0 + ce))
    # hard-split any still-too-long piece
    out = []
    for a, b in pieces:
        while b - a > MAXCH * 2:
            out.append((a, a + MAXCH)); a += MAXCH
        out.append((a, b))
    return out

def segment(text):
    """steps with char spans + token spans (retokenized, special tokens off). Returns (steps, ids, offsets)."""
    tok = tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        if m.start() > pos: spans += split_paragraph(text[pos:m.start()], pos)
        pos = m.end()
    if pos < len(text): spans += split_paragraph(text[pos:], pos)
    # merge consecutive tiny paragraphs into steps of >= MINCH chars (never across </think>, never beyond MAXCH)
    tc = text.find("</think>"); tc = tc + len("</think>") if tc >= 0 else 10**12
    merged = []
    for a, b in spans:
        if not text[a:b].strip(): continue
        if merged and (merged[-1][1] - merged[-1][0]) < MINCH and (b - merged[-1][0]) <= MAXCH and (merged[-1][0] >= tc) == (a >= tc) \
                and not (merged[-1][0] < tc <= a):
            merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    spans = merged
    starts = np.array([o[0] for o in offs])
    te = ids.index(THINK_END) if THINK_END in ids else None
    steps = []
    for i, (a, b) in enumerate(spans):
        if not text[a:b].strip(): continue
        t0 = int(np.searchsorted(starts, a, "left")); t1 = int(np.searchsorted(starts, b, "left"))
        phase = "answer" if (te is not None and t0 > te) else "think"
        if te is not None and t0 <= te < t1 and phase == "think": pass
        steps.append(dict(step=len(steps), c0=a, c1=b, t0=t0, t1=t1, phase=phase, text=text[a:b]))
    return steps, ids, te

@functools.lru_cache(None)
def _metric():
    import sys; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "third-party/lighteval/src"))
    from lighteval_custom.tasks.reasoning import latex_gold_metric
    from lighteval.tasks.requests import Doc
    return latex_gold_metric.sample_level_fn, Doc

def grade(pred_text, gold):
    """Same grader as the stage-1 evaluation (lighteval_custom latex_gold_metric). pred_text may be any text containing \\boxed{}."""
    fn, Doc = _metric()
    doc = Doc(task_name="x", query="", choices=[gold], gold_index=0)
    try: return float(fn([gold], [pred_text], doc))
    except Exception: return 0.0

_BOX = re.compile(r"\\boxed\s*\{")
def boxed_answers(text):
    """all \\boxed{...} contents (balanced braces) with char offsets"""
    out = []
    for m in _BOX.finditer(text):
        i, d = m.end(), 1
        while i < len(text) and d:
            d += (text[i] == "{") - (text[i] == "}"); i += 1
        if d == 0: out.append((m.start(), text[m.end():i - 1]))
    return out

@functools.lru_cache(None)
def problems():
    return {r["unique_id"]: r["problem"] for r in map(json.loads, open(f"{REPO}/datasets/MATH-500/test.jsonl"))}

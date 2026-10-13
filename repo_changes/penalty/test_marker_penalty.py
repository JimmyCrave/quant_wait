"""Checks of the marker penalty on the real vLLM 0.7.0 path (first decoding step, fixed prompt/prefix).
(1) lambda=0 is bit-identical to no processor; (2) only marker tokens change, by exactly -lambda/T in the
reported (post-temperature) logprob space => penalty is applied to RAW logits BEFORE temperature scaling;
(3) negative lambda boosts. Also pure-torch unit test of the processor."""
import json, sys, torch
from vllm import LLM, SamplingParams
from penalty.marker_penalty import MarkerPenaltyProcessor, load_marker_token_ids

ids = load_marker_token_ids("penalty/qwen_marker_token_ids.json")
S = set(ids)

# --- pure torch unit test (fixed logits)
for dt, tol in ((torch.float32, 1e-5), (torch.bfloat16, 0.0625)):
    z = torch.randn(151936, dtype=dt, device="cuda") * 5
    for lam in (0.0, 1.0, -2.0, 0.7):
        out = MarkerPenaltyProcessor(ids, lam)([], z.clone())
        diff = out.float() - z.float()
        changed = set(diff.nonzero().flatten().tolist())
        if lam == 0.0:
            assert not changed, "lambda=0 must be exact identity"
        else:
            assert changed <= S and len(changed) > 0
            err = (diff[ids] + lam).abs().max().item()
            assert err <= tol, (dt, lam, err)
            print(f"  unit {dt} lambda={lam}: only marker ids changed ({len(changed)}/{len(S)}); max |shift+lambda|={err:.4f}")
        others = torch.ones(z.numel(), dtype=torch.bool, device="cuda"); others[ids] = False
        assert torch.equal(out[others], z[others]), "non-marker logits modified"
print("torch unit test OK")

model = "modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B"
llm = LLM(model, dtype="bfloat16", max_model_len=4096, enforce_eager=True, gpu_memory_utilization=0.5, max_logprobs=2000)
tok = llm.get_tokenizer()
prompt = ("<｜begin▁of▁sentence｜><｜User｜>What is 2+3? Please reason step by step, and put your final answer within \\boxed{}."
          "<｜Assistant｜><think>\nI think 2+3=5. So the answer is 5.")
res = {}
for T in (1.0, 0.6):
    for lam in (None, 0.0, 1.0, -2.0):
        lp = [] if lam is None else [MarkerPenaltyProcessor(ids, lam)]
        sp = SamplingParams(temperature=T, top_p=1.0, max_tokens=1, logprobs=2000, seed=42, logits_processors=lp)
        o = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0]
        res[(T, lam)] = ({t: l.logprob for t, l in o.logprobs[0].items()}, o.token_ids[0])
for T in (1.0, 0.6):
    base, b_tok = res[(T, None)]
    zero, z_tok = res[(T, 0.0)]
    assert base == zero and b_tok == z_tok, "lambda=0 differs from baseline"
    print(f"T={T}: lambda=0 logprobs identical to no-processor over {len(base)} tokens; sampled token equal")
    for lam in (1.0, -2.0):
        pen, _ = res[(T, lam)]
        common = [t for t in base if t in pen]
        nm = [t for t in common if t not in S]
        mk = [t for t in common if t in S]
        ref = nm[0]
        # relative logit shift vs a non-marker reference token, expressed in raw-logit units (multiply by T)
        non_marker_dev = max(abs((pen[t] - pen[ref]) - (base[t] - base[ref])) for t in nm)
        shifts = [((pen[t] - pen[ref]) - (base[t] - base[ref])) * T for t in mk]
        print(f"  lambda={lam}: {len(nm)} non-marker / {len(mk)} marker tokens compared; "
              f"max |non-marker relative change|={non_marker_dev:.2e}; "
              f"marker raw-logit shifts min/max={min(shifts):.4f}/{max(shifts):.4f} (expected {-lam})")
        assert non_marker_dev < 1e-4 and all(abs(s + lam) < 0.13 for s in shifts), 'bf16 raw-logit rounding bound exceeded'
        print(f'    max |shift+lambda| = {max(abs(s + lam) for s in shifts):.4f} (bf16 rounding of raw logits)')
print("ALL CHECKS PASSED")

# Minimal reproduction: AWQ-W3g128 (numerical simulation) + marker penalty, DeepSeek-R1-Distill-Qwen-1.5B, MATH-500

All three groups COMPLETED. Single seed (42), one sample per question, 100 questions -> binomial SE about 4-5 points.

## Results (same 100 questions, T=0.6, top_p=0.95, max_new_tokens=32768, max_model_len=34816)

| group | acc | mean / median / P90 gen tokens | budget exhausted |
|---|---|---|---|
| BF16 | 0.80 | 6713 / 3306 / 16836 | 5/100 |
| AWQ3 numerical simulation | 0.39 | 25841 / 32768 / 32768 | 70/100 |
| AWQ3 + marker penalty, lambda=1.0 (pre-fixed probe, not tuned, not claimed optimal) | 0.51 | 15856 / 8378 / 32768 | 35/100 |

Paired differences (`paired_comparison.json`): AWQ3-BF16 = -0.41 (paired SE 0.051); AWQ3+pen - AWQ3 = +0.12 (paired SE 0.046; 17 questions fixed, 5 broken); still -0.29 vs BF16.
"Budget exhausted" = vLLM `finish_reason == "length"` (hit 32768 new tokens; max prompt 369 tokens, so model length was never the limiter). Exhausted outputs are almost always scored wrong (acc 0.0 / 0.21 / 0.09 among exhausted), so accuracy and exhaustion are coupled.

## Provenance
- Main repo commit and submodules: `commits.txt` (vLLM submodule v0.7.0, lighteval 0.8.0). Penalty reference repo cloned but NOT used (no patch, no GLM IDs).
- Env: `.venv` Python 3.12, torch 2.5.1+cu124, transformers 4.47.1, vLLM 0.7.0 (`pip_freeze.txt`).
- Code changes: `code.diff` (inference.py, lighteval_custom/models/vllm/vllm_model.py, new `penalty/`). Run command: `run_experiments.sh`; per-run command/config/wall time: `results/*/MATH-500.run_config.json`; logs: `logs/`.
- Trajectories: `results/{bf16,awq3,awq3_lambda1.0}/MATH-500.jsonl`. NOTE: despite the suffix this is ONE JSON ARRAY (verified; read with `json.load`). Original 4 fields kept; added `task_id, num_input_tokens, num_generated_tokens, score, finish_reason, stop_reason, budget_exhausted, request_latency_s`.
- Summaries: `results_summary.json`, `summarize.py`.

## AWQ3 model: what it is
- Made with the repo's own flow (`methods.awq.run_awq --w_bits 3 --w_groupsize 128 --w_asym`, same args as scripts/quantization/awq.sh W3 branch; tp suffix = 1). Weights are stored as dequantized bf16. **This is numerical simulation (fake quantization), not a real 3-bit kernel run; no speed/memory claims.**
- Checked (`awq_weight_check.json`, `logs/verify_loaded_weights.log`): 196 decoder linear weights changed, every 128-group has <=8 distinct values; vLLM's in-memory qkv/gate_up/down weights (layers 0, 13, 27) equal the AWQ checkpoint and differ from BF16. embed_tokens, lm_head, norms-final unchanged (AWQ also rescales input/post-attention layernorm weights). KV cache and activations are bf16.

## Calibration vs paper (Appendix A.1: "128 sequences of length 512 from Pile uncopyrighted", standard AWQ recipe)
| item | paper | here |
|---|---|---|
| dataset | monology/pile-uncopyrighted | `mit-han-lab/pile-val-backup` (repo default, original AWQ data) |
| samples | 128 x 512 | 128 documents of <=512 tokens (shuffle seed 42) are concatenated (57,054 tokens) and cut into **60** blocks of 512 |
| other preprocessing, seed, AWQ version | not stated | repo's own AWQ implementation (not AutoAWQ) |
The dataset difference was not fixed; it may affect AWQ quality.

## Marker penalty
- Paper Appendix C gives only token STRINGS (50, each with leading space), no IDs. IDs were mapped with the Qwen tokenizer (`penalty/build_qwen_marker_ids.py`, `qwen_marker_token_ids.json`): **49/50** are single tokens; `" recheck"` -> [" re","check"] and is excluded (not guessed). Penalty is therefore on 49 tokens.
- Implementation: per-request vLLM logits processor (`penalty/marker_penalty.py`), z[v] -= lambda, no vLLM patch. In vLLM 0.7.0 it runs on raw logits (logits_processor.py:80), before temperature (sampler.py:271) and top-p (sampler.py:274). Verified empirically with T=0.6: marker shift = -lambda in raw-logit units, not -lambda/T.
- Tests (`logs/test_marker_penalty.log`): lambda=0 gives bit-identical logprobs and sampled token vs no processor; with fixed prefix/logits only marker ids change (non-marker relative change <3e-6), for lambda = 1.0 and -2.0 (negative boosts). Caveat: raw logits are bf16, so the shift is quantized to bf16 resolution (up to ~0.06 for |lambda|=2). The lambda=0 end-to-end check was one decoding step on the real path, not a full-trajectory comparison.

## Question set
`--max_samples 100`: lighteval shuffles MATH-500 with Random(42) and takes the first 100 (not the first 100 file rows); identical across groups (verified by task_id).

## Environment fixes (minimal, no framework changes)
- vLLM: installed the official v0.7.0 wheel (matches the submodule tag) instead of source build; submodule `pip install -e` hung for 20+ min fetching a precompiled wheel. Editable submodule install was therefore NOT done; `import vllm` uses the wheel in site-packages.
- Pins to fix import errors: pyarrow 19.0.1, pandas<2.3, xxhash 3.5.0 (4.x crashed lighteval logging after generation), protobuf 3.20.3 left from lighteval (pip check reports unrelated opentelemetry/google conflicts).
- `LD_PRELOAD=/usr/local/libvgpu/libcuda.so.1` (cluster shim) caused a vLLM segfault; all runs use `env -u LD_PRELOAD`.
- Not done: xformers rebuild from submodule, AutoAWQ/GPTQModel (not needed).

## Limits
- n=100, single seed: differences of a few points are noise; the AWQ3 drop is large, the penalty gain is suggestive (about 2.6 paired SE) but not established. No lambda sweep, nothing tuned on these questions.
- Wall times were measured with the three jobs running concurrently on separate GPUs, and the penalty uses a Python per-row processor; do not read them as a speed comparison.
- bf16 GPU sampling is not bitwise reproducible across batch compositions.

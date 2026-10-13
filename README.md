# quant_wait

Reproduction and mechanism experiments for quantization-induced overthinking (DeepSeek-R1-Distill-Qwen-1.5B, AWQ W3 g128 numerical simulation).

- `artifacts/` stage 1: minimal reproduction (BF16 / AWQ3 / AWQ3 + marker penalty), see `artifacts/README.md`. `commits.txt` pins upstream commits; `code.diff` is the change to Quantized-Reasoning-Models.
- `repo_changes/` copies of the files changed or added in Quantized-Reasoning-Models (`inference.py`, `vllm_model.py`, `penalty/`).
- `quant_reasoning_lab/` stage 2 mechanism study, see `mechanism_report.md`; `reproduce.sh`.
- `quant_reasoning_lab/dur_study/` correction-duration study, see `duration_report.md`; `reproduce_dur.sh`.
- `paper.html` target paper snapshot.

Not included (size / regenerable): model weights, datasets, venv, wheels, upstream repos, intermediate `data/` (except annotator labels), `dur_study/data/b7_steps.jsonl`.
Upstream: https://github.com/ruikangliu/Quantized-Reasoning-Models (commit in `artifacts/commits.txt`).

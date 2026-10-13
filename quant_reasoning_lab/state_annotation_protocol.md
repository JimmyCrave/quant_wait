# State annotation protocol (B1)

Applies to DeepSeek-R1-Distill-Qwen-1.5B reasoning traces on MATH-500 (BF16 and AWQ-W3g128 fake-quant; seeds 42-45).
The protocol, thresholds and sampling rules were fixed in `configs/prereg.yaml` before any transition / counterfactual /
intervention result was inspected.

## 1. Units and position mapping
* Text is split into paragraphs on blank lines; consecutive paragraphs are merged until a step has >= 150 chars (never across `</think>`,
  never beyond 700 chars); paragraphs > 700 chars are cut at sentence boundaries.
  Merging was added because quantized loops produce thousands of one-line paragraphs ("Wait, that's correct.").
* Every step keeps char span `[c0,c1)` and token span `[t0,t1)` (tokens = re-tokenisation of the text with the model tokenizer, special tokens off).
  Re-tokenisation reproduced the sampled token ids exactly in 295/300 BF16 replicate trajectories (the rest differ by non-canonical BPE splits);
  all token positions in this study refer to the re-tokenised sequence, and all prefixes fed to both models are taken from it.
* Phase: steps after the `</think>` token (id 151649) are the *answer section* and are labelled A deterministically.

## 2. Labels (what an annotator decides for each step inside `<think>`)
| label | meaning |
|---|---|
| G | Productive reasoning. Introduces new information: a new computation, case, method, a genuine verification that checks something not yet checked, or a necessary correction of a real error. A self-correction that fixes a mistake is **G**, even when it starts with Wait/But/Hmm. |
| L | Unproductive loop. Re-states or re-derives something already established earlier, returns to the same conclusion / doubt / contradiction, or redoes the same check, without new information and without fixing an error. Paraphrase counts as repetition. |
| A | Explicitly commits to the final answer ("So the final answer is X", "**Final Answer** \boxed{X}", "I'll go with X"). Everything after `</think>` is A by rule. |
| U | Cannot be judged reliably from the text shown. |

Rules: (1) A step is not L merely because it contains Wait / But / Alternatively / Hmm. (2) Checking a *different* thing than before is G; checking the same thing again is L.
(3) A first re-statement of a result as part of normal consolidation (once) is G; the second and later identical re-statements with no new content are L.
(4) If unsure, label U (do not guess).

Additional field **ANS**: the answer to the problem stated or proposed in the step (tentative or final), or `-`.
Offline only: ANS and every `\boxed{}` inside `<think>` are graded against the gold answer with the same grader as the stage-1 evaluation
(lighteval_custom `latex_gold_metric`). The gold answer is never shown to annotators and never used by any online procedure.

Derived event positions (all token positions): first answer proposal; first *correct* proposal; later keep / revise / abandon of a correct
proposal (a later incorrect proposal = revision away from the correct answer); submission = `</think>` token; budget exhaustion = trajectory
ends by `finish_reason == "length"` without `</think>`.

## 3. Loop-candidate detectors (candidates only, never labels)
1. Exact / near text repetition: word-5-gram-shingle Jaccard >= 0.6 with *any* earlier step (any distance); or zlib compression ratio < 0.25 for steps >= 200 bytes.
2. Semantic repetition: bge-small-en-v1.5 cosine >= 0.90 with a step >= 2 steps earlier AND no new number / LaTeX-command / symbol token vs the 30 preceding steps.
`Wait`, `But`, `Alternatively` are never used as a feature.

## 4. Automatic annotators (NOT human gold)
* Primary: Qwen2.5-14B-Instruct, temperature 0, chunks of 8 steps, 4 previous steps as context, problem text shown,
  *with* neutral retrieval evidence (the most similar earlier step, if similarity high). Configuration, seed, accuracy and gold are hidden.
* Audit-only variants on the audit windows: (a) same model without retrieval evidence, (b) phi-4 with evidence.
* **No human has labelled anything.** `annotation_pack/` is ready for human review; until humans fill it, all reliability numbers below are
  inter-automatic-annotator agreements and detector agreement, which can share systematic errors.

## 5. High-confidence subset
* L_hc = labeler L AND (text or semantic candidate); G_hc = labeler G AND no candidate; A = A; everything else = X (ambiguous).
  A single X between two L_hc steps is treated as L_hc (smoothing). Spells and hazards are reported for both the full automatic labels (`auto`)
  and the high-confidence labels (`hc`); the coverage of the hc subset and its selection bias are reported in the mechanism report.
* Loop event: >= 2 consecutive L steps (entry = first L step of the run). Exit: first non-L step after the run: G (L->G), A (L->A), ambiguous (treated as censored).
  A run still open when the trajectory ends is right-censored.

## 6. Audit windows and anonymous annotation pack
* >= 100 windows of 10 consecutive steps (+ 4 context steps), sampled with seed 777 *from detector-defined strata only* (no labeler output used):
  normal_G (no candidates), repetition_candidate (>= 5/10 candidate steps), valid_correction_candidate (a Wait/But/Hmm/Alternatively-opening step with >= 2 new symbols and no candidate flag),
  near_end (last 10 thinking steps), near_answer (window around the first boxed answer), for BF16 and AWQ3 equally.
* Windows get random ids; configuration, seed, question id, accuracy, position and stratum are removed from the pack (key kept in `audit_key_private.json`).
* `annotation_pack/annotation_template.csv`: one row per step, columns `anon_id, local_step, label, ans, notes, annotator`.

# Blind human review pack (60 windows x 10 steps)

STATUS: **NOT COMPLETED. No human has labelled these windows.** `annotation_template.csv` is empty on purpose.
Nothing in this pack identifies the model (BF16 vs quantized), seed, trajectory or position, and no automatic label is shown.

For every window (`windows.jsonl`, key `anon_id`) read `problem`, the 4 `context` steps (do not label them) and the 10 `steps`, then fill one row per step in
`annotation_template.csv`:

* `label_GLAU` - G: productive reasoning (new information, new method/case, genuine check, necessary correction of a real mistake - a self-correction that
  fixes an error is G even if it starts with "Wait"); L: unproductive loop (re-states / re-derives something already established, same doubt or conclusion again,
  same check again, without new information or a fixed error; paraphrase counts); A: explicitly commits to the FINAL answer; U: cannot judge.
  "Wait/But/Alternatively" are NOT evidence of L by themselves.  Same computation template with different numbers is G.
* `states_an_answer_YN` - does the step state an answer to the problem (tentative or final)?
* `stated_answer_correct_YNNA` - if yes, is it correct (use `reference_solution_for_checking_answers_only`; answer correctness only, never use it to judge G/L)?
Full definitions: `../../state_annotation_protocol.md`. Please have >= 2 annotators label independently (different `annotator_id`), then run `src/ingest_human_labels.py`.

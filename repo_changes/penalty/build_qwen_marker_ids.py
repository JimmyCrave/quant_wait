"""Map the 50 overthinking markers of paper Appendix C (Table 8) to token IDs of the
DeepSeek-R1-Distill-Qwen tokenizer. Every marker carries a leading space ("_" in the paper)
and must encode to exactly ONE token; otherwise it is reported and excluded (never guessed)."""
import json
import sys

from transformers import AutoTokenizer

MARKERS = ("perhaps maybe wait Wait actually hold Hmm hmm Alternatively alternatively However however "
           "instead Instead But but though although yet rather unless otherwise nonetheless nevertheless "
           "regardless still anyway Or or either whether uncertain unsure possibly might could another "
           "different reconsider rethink backtrack retry recheck revisit doubt confused wrong mistake "
           "error incorrect").split()
assert len(MARKERS) == 50 and len(set(MARKERS)) == 50

model, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model)
mapping, missing = {}, {}
for m in MARKERS:
    ids = tok.encode(" " + m, add_special_tokens=False)
    if len(ids) == 1 and tok.decode(ids) == " " + m:
        mapping[" " + m] = ids[0]
    else:
        missing[" " + m] = {"ids": ids, "pieces": tok.convert_ids_to_tokens(ids)}
json.dump({"tokenizer": model, "n_paper_markers": len(MARKERS), "n_single_token": len(mapping),
           "token_ids": mapping, "not_single_token": missing}, open(out, "w"), indent=1)
print(len(mapping), "single-token markers;", "multi-token:", missing)

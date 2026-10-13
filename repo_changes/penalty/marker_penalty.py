"""Marker penalty as a per-request vLLM logits processor (no vLLM patch).

z'(v) = z(v) - lambda  for v in marker token set, unchanged otherwise (paper Eq. 1).

Order in vLLM 0.7.0 (V0 engine, third-party/vllm):
  LogitsProcessor.forward -> _apply_logits_processors (this class; RAW model logits)  [logits_processor.py:80]
  Sampler.forward         -> temperature division [sampler.py:271] -> top-p/top-k [sampler.py:274]
                             -> softmax -> sampling
i.e. the penalty is applied to raw (pre-temperature) logits, before temperature and top-p.
Note: raw logits are in the model dtype (bf16), so the subtraction is rounded to bf16 resolution.
"""
import json

import torch


def load_marker_token_ids(path):
    with open(path) as f:
        return sorted(set(json.load(f)["token_ids"].values()))


class MarkerPenaltyProcessor:
    def __init__(self, token_ids, penalty):
        self.token_ids = list(token_ids)
        self.penalty = float(penalty)
        self._idx = {}

    def __call__(self, past_token_ids, logits):
        idx = self._idx.get(logits.device)
        if idx is None:
            idx = self._idx[logits.device] = torch.tensor(self.token_ids, dtype=torch.long, device=logits.device)
        logits[idx] -= self.penalty
        return logits

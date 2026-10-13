"""Inspect the weights that vLLM actually holds in GPU memory after loading the AWQ3 directory."""
import torch
from vllm import LLM
from penalty.marker_penalty import MarkerPenaltyProcessor  # noqa (path sanity)
llm = LLM("outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1", dtype="bfloat16",
          max_model_len=2048, enforce_eager=True, gpu_memory_utilization=0.3)
eng = llm.llm_engine
runner = eng.model_executor.driver_worker.model_runner
model = runner.model
print("vLLM model class:", type(model).__name__, "| quant_config:", eng.model_config.quantization)
ref = llm.llm_engine  # placeholder
from safetensors import safe_open
bf = safe_open("modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors", "pt")
qf = safe_open("outputs/modelzoo/awq/DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1/model.safetensors", "pt")
sd = dict(model.named_parameters())
# merged qkv/gate_up params: compare against concatenated checkpoint tensors
def check(vname, parts):
    w = sd[vname].detach().float().cpu()
    q = torch.cat([qf.get_tensor(p).float() for p in parts], 0)
    b = torch.cat([bf.get_tensor(p).float() for p in parts], 0)
    g = w.reshape(w.shape[0], -1, 128)[:32]
    uniq = max(torch.unique(g[i, j]).numel() for i in range(g.shape[0]) for j in range(g.shape[1]))
    print(f"{vname}: equals AWQ checkpoint={torch.equal(w, q)}, equals BF16 original={torch.equal(w, b)}, max distinct values per 128-group={uniq}")
for l in (0, 13, 27):
    P = f"model.layers.{l}."
    check(P + "self_attn.qkv_proj.weight", [P + f"self_attn.{n}_proj.weight" for n in "qkv"])
    check(P + "mlp.gate_up_proj.weight", [P + f"mlp.{n}_proj.weight" for n in ("gate", "up")])
    check(P + "mlp.down_proj.weight", [P + "mlp.down_proj.weight"])

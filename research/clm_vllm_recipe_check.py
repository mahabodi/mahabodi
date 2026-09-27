"""Is research/clm_embed_server.py's recipe the same as vLLM pooling? Compare real vLLM (`runner="pooling"`, fp16 on the
2080 Ti) with the transformers recipe (fp32, final-norm hidden state at the last token, L2) on Qwen3-0.6B: the same
architecture family as CLM's Qwen3-8B, small enough to fit the GPU. Passing means min cosine >= 0.999 over all texts.
Run in a separate venv with vLLM (research/PREREG_CLM.md, parity follow-up).
"""
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt

M = "Qwen/Qwen3-0.6B"
texts = ["Customer: my invoice was charged twice and nobody answers the phone!\n\nIs this urgent?",
         "true: Yes. This is true: Is this urgent?", "false: No. This is false: Is this urgent?",
         "Charges, invoices, refunds", "Bugs and outages", "Calm", "Very angry",
         "What causes tides on Earth?", "The Moon's gravitational pull."] + \
        ["sample text number %d about banking, refunds and card payments" % i for i in range(100)]
tok = AutoTokenizer.from_pretrained(M)
ids = [tok(t, add_special_tokens=True)["input_ids"] for t in texts]
llm = LLM(model=M, runner="pooling", dtype="float16", gpu_memory_utilization=0.5, enforce_eager=True)
V = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in llm.embed([TokensPrompt(prompt_token_ids=i) for i in ids])])
V /= np.linalg.norm(V, axis=1, keepdims=True)
del llm
torch.cuda.empty_cache()
m = AutoModel.from_pretrained(M, dtype=torch.float32).eval()
T = []
with torch.inference_mode():
    for i in ids:
        h = m(input_ids=torch.tensor([i])).last_hidden_state[0, -1].float()
        T.append((h / h.norm()).numpy())
T = np.stack(T)
cos = (V * T).sum(1)
print(json.dumps({"model": M, "n": len(texts), "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
                  "vllm": "runner=pooling, float16, GPU", "ours": "transformers fp32, final-norm last token, L2",
                  "pass": bool(cos.min() >= 0.999)}))

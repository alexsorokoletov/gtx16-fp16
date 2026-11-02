"""End-to-end, GPU-bound case: BERT-style embedding model (bge-base-en-v1.5) on a batch of texts, fp32
vs gtx16_fp16.patch(). Reports throughput and cosine similarity of the embeddings."""
import time

import torch
from transformers import AutoModel, AutoTokenizer

import gtx16_fp16 as g

name = "BAAI/bge-base-en-v1.5"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
texts = [f"Document {i}: the fox named Ember carried a fallen star home through the snowy pine forest." * 4
         for i in range(256)]
batch = tok(texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to("cuda")


def run(label, reps=5):
    with torch.no_grad():
        out = model(**batch).last_hidden_state[:, 0]  # warm-up
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            out = model(**batch).last_hidden_state[:, 0]
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    print(f"{label:6} {len(texts) / dt:7.1f} texts/s ({dt * 1000:.0f} ms per batch of {len(texts)} x 256 tokens)",
          flush=True)
    return torch.nn.functional.normalize(out.float(), dim=-1), dt


ref, t32 = run("fp32")
n = g.patch(model)
torch.cuda.empty_cache()
emb, t16 = run("gtx16")
cos = (ref * emb).sum(-1)
print(f"patched {n} Linear layers | speedup x{t32 / t16:.2f} | cosine vs fp32: min {cos.min():.6f} mean {cos.mean():.6f}")

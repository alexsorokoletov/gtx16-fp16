"""Find Linear layers whose inputs or weights leave the fp16-safe range in a real model (Qwen3-TTS 0.6B)."""
import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel

m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda:0",
                                  dtype=torch.float32, attn_implementation="sdpa")
stats = {}


def hook(name):
    def f(mod, inp, out):
        x = inp[0].detach()
        s = stats.setdefault(name, {"x_amax": 0.0, "w_amax": mod.weight.detach().abs().max().item(),
                                    "slice16_amax": 0.0, "K": mod.in_features, "N": mod.out_features})
        s["x_amax"] = max(s["x_amax"], x.abs().max().item())
        # worst partial sum of 16 consecutive products (what the hybrid kernel accumulates in fp16)
        xs = x.reshape(-1, x.shape[-1])[:64].float()
        K = xs.shape[1] // 16 * 16
        if K:
            prod = xs[:, None, :K] * mod.weight.detach()[None, :, :K].float()  # [rows, N, K]
            part = prod.reshape(prod.shape[0], prod.shape[1], -1, 16).sum(-1)
            s["slice16_amax"] = max(s["slice16_amax"], part.abs().max().item())
    return f


inner = m.model if hasattr(m, "model") else m
for n, mod in inner.named_modules():
    if isinstance(mod, nn.Linear) and min(mod.in_features, mod.out_features) >= 256:
        mod.register_forward_hook(hook(n))
torch.manual_seed(0)
with torch.no_grad():
    m.generate_custom_voice(text="Once upon a time there lived a small fox named Ember.", language="English",
                            speaker="Ryan")
bad = {n: s for n, s in stats.items() if s["x_amax"] > 2048 or s["slice16_amax"] > 8192 or s["w_amax"] > 2048}
print(f"layers checked: {len(stats)}; risky (x>2048, 16-slice partial>8192, or w>2048): {len(bad)}")
for n, s in sorted(stats.items(), key=lambda kv: -kv[1]["slice16_amax"])[:15]:
    print(f"{n:70} K={s['K']:5} N={s['N']:5} x_amax={s['x_amax']:10.1f} w_amax={s['w_amax']:8.2f} "
          f"slice16={s['slice16_amax']:10.1f}")

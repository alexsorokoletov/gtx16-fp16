"""End-to-end check on a real model: Qwen3-TTS 0.6B in fp32, with and without gtx16_fp16.patch().
Reports speed (seconds of audio per second of compute), patched layer count, VRAM, and audio validity.
Needs: pip install qwen-tts soundfile"""
import sys
import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

import gtx16_fp16 as g

TEXTS = [
    "Once upon a time, in a village at the edge of a pine forest, there lived a small fox named Ember.",
    "Every night, Ember climbed the old hill to count the stars, but there was always one star missing.",
    "Жила-была на краю соснового леса маленькая лисичка по имени Искорка.",
]
LANGS = ["English", "English", "Russian"]
out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"


def run(label, model):
    model.generate_custom_voice(text="Warm up.", language="English", speaker="Ryan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    audio_s = gen_s = 0.0
    for i, (t, lang) in enumerate(zip(TEXTS, LANGS)):
        torch.manual_seed(i)
        t0 = time.perf_counter()
        w, sr = model.generate_custom_voice(text=t, language=lang, speaker="Ryan")
        torch.cuda.synchronize()
        gen_s += time.perf_counter() - t0
        audio_s += len(w[0]) / sr
        ok = bool(np.isfinite(w[0]).all()) and float(np.abs(w[0]).max()) > 0.01
        sf.write(f"{out_dir}/e2e_{label}_{i}.wav", w[0], sr)
        print(f"  {label} #{i}: {len(w[0]) / sr:.1f}s audio, valid={ok}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label:10} {audio_s:.1f}s audio in {gen_s:.1f}s -> x{audio_s / gen_s:.2f} realtime, peak VRAM {peak:.2f} GB",
          flush=True)
    return audio_s / gen_s


m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda:0",
                                  dtype=torch.float32, attn_implementation="sdpa")
print("device:", torch.cuda.get_device_name(0), "| affected:", g.is_affected())
base = run("fp32", m)
n = g.patch(m.model if hasattr(m, "model") else m, min_features=256)
print("patched Linear layers:", n)
torch.cuda.empty_cache()
fast = run("gtx16", m)
print(f"speedup x{fast / base:.2f}")

#!/usr/bin/env python3
"""Local GPU test: run CogVideoX-5B on RTX 3060 12GB with int8 quantization."""

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/hf_cache"))

import torch
from PIL import Image

# Verify GPU
assert torch.cuda.is_available(), "No CUDA GPU found"
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

# Use backends.py
sys.path.insert(0, os.path.dirname(__file__))
from backends import get_backend

# Load CogVideoX-5B with int8 quantization
backend = get_backend("cogvideo5b")
cache_dir = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/hf_cache"))

print(f"\n--- Loading {backend.display_name} ---")
t0 = time.time()
backend.load(cache_dir=cache_dir)
load_time = time.time() - t0
print(f"Load time: {load_time:.1f}s")

# Use an existing test image
img_path = "results/avatar_p0_frame_000.png"
if not os.path.exists(img_path):
    # Create a simple test image if none available
    img = Image.new("RGB", (720, 480), color=(100, 150, 200))
    print("Using synthetic test image")
else:
    img = Image.open(img_path).convert("RGB")
    print(f"Using test image: {img_path} ({img.size})")

# Generate
prompt = "A person waves hello slowly, simple motion"
print(f"\n--- Generating 49 frames ---")
print(f"Prompt: {prompt}")
t0 = time.time()

try:
    frames = backend.generate(
        image=img,
        prompt=prompt,
        negative_prompt="blurry, distorted",
        height=480,
        width=720,
        num_frames=49,
        guidance_scale=6.0,
        steps=10,
        seed=42,
    )
    gen_time = time.time() - t0
    print(f"\nGeneration complete: {len(frames)} frames in {gen_time:.1f}s")

    # Save sample frames
    os.makedirs("results", exist_ok=True)
    for i, idx in enumerate([0, 12, 24, 36, 48]):
        if idx < len(frames):
            out_path = f"results/local_gpu_frame_{idx:03d}.png"
            frames[idx].save(out_path)
            print(f"  Saved {out_path}")

    # Peak VRAM
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"\nPeak VRAM: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
    print(f"Load time: {load_time:.1f}s, Generation time: {gen_time:.1f}s")
    print("\nSUCCESS: CogVideoX-5B runs locally on RTX 3060!")

except torch.cuda.OutOfMemoryError as e:
    print(f"\nOOM Error: {e}")
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak VRAM before OOM: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
    print("RESULT: RTX 3060 12GB is insufficient for CogVideoX-5B")
    sys.exit(1)

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

"""Test AI-generated avatars locally with CogVideoX-5B on RTX 3060.

Quick forward-only test (no reset) to evaluate how each avatar animates.
Uses sequential CPU offload, 10 steps, ~6 min per avatar.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_avatars_local.py
"""
import glob
import os
import time
import torch
from PIL import Image

os.environ["DIPPY_NO_QUANTIZE"] = "1"  # torchao + cpu_offload incompatible

from backends import get_backend

SENTENCE = "He jumped"
BACKEND = "cogvideo5b"
STEPS = 10
GUIDANCE = 6.0
SEED = 42
OUTPUT_DIR = "results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    avatar_paths = sorted(glob.glob("avatars/*.png"))
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Avatars: {len(avatar_paths)}")
    print(f"Sentence: \"{SENTENCE}\"")
    print(f"Backend: {BACKEND}, Steps: {STEPS}, Guidance: {GUIDANCE}")

    backend = get_backend(BACKEND)
    print(f"\nLoading {backend.display_name}...")
    t0 = time.time()
    backend.load(cache_dir=".hf_cache")
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    num_frames = backend.valid_num_frames(49)

    for avatar_path in avatar_paths:
        name = os.path.splitext(os.path.basename(avatar_path))[0]
        avatar = Image.open(avatar_path).convert("RGB")
        ground = avatar.resize((720, 480))

        print(f"\n{'='*50}")
        print(f"Avatar: {name} (original: {avatar.size})")

        # Save resized input
        ground.save(os.path.join(OUTPUT_DIR, f"avatar_cog_{name}_input.png"))

        # Forward pass only (saves time — reset is same mechanics)
        prompt = (
            f"The character enthusiastically acts out '{SENTENCE}' with big, "
            "exaggerated body movements, arms moving expressively, full-body "
            "pantomime gestures. Smooth animation, dynamic motion."
        )

        print(f"Generating ({num_frames} frames, {STEPS} steps)...")
        t1 = time.time()
        frames = backend.generate(
            image=ground,
            prompt=prompt,
            negative_prompt="blurry, static, low quality, distorted",
            height=480, width=720,
            num_frames=num_frames,
            guidance_scale=GUIDANCE,
            steps=STEPS,
            seed=SEED,
        )
        gen_time = time.time() - t1
        print(f"  Generated {len(frames)} frames in {gen_time:.1f}s")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        # Save keyframes
        for idx in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]:
            frames[idx].save(os.path.join(
                OUTPUT_DIR, f"avatar_cog_{name}_f{idx:03d}.png"))

        # Save video
        try:
            from diffusers.utils import export_to_video
            clip_path = os.path.join(OUTPUT_DIR, f"avatar_cog_{name}_fwd.mp4")
            export_to_video(frames, clip_path, fps=backend.fps)
            size_mb = os.path.getsize(clip_path) / (1024 * 1024)
            print(f"  Saved: {clip_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Video export failed: {e}")

        # Reset VRAM tracking for next avatar
        torch.cuda.reset_peak_memory_stats()

    backend.unload()
    print("\nDone!")


if __name__ == "__main__":
    main()

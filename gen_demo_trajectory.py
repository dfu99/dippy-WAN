"""
Generate a seamless multi-scene dippy demo video on RTX 3060 (12GB).

Uses CogVideoX-5B with sequential CPU offload. Each sentence gets a forward
pass (act out) + reset pass (return to idle), chained into a smooth trajectory.
"""

import os
import sys
import time

# Configure cache and environment before any HF imports
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.path.dirname(__file__), ".hf_cache"))
os.environ.setdefault("HF_HOME", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# No quantization — torchao + cpu_offload is incompatible on some setups
os.environ["DIPPY_NO_QUANTIZE"] = "1"

import torch
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video

from backends import get_backend, _frame_to_pil

# ── Config ──────────────────────────────────────────────────────────────────

AVATAR_PATH = "avatars/Perplexity.png"
OUTPUT_DIR = "results"
BACKEND_NAME = "cogvideo5b"

SENTENCES = [
    "He jumped",
    "She laughed",
    "She laughed at him",
    "He ran",
    "He ran because it started raining",
]

# CogVideoX-5B defaults
STEPS = 10          # 10 steps ≈ 30 steps quality, 4x faster
GUIDANCE = 6.0      # optimal per sweep
SEED_BASE = 42
NUM_FRAMES = 49     # fixed for CogVideoX
FPS = 8

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards, watermark, text, signature"
)

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load avatar
    avatar = Image.open(AVATAR_PATH).convert("RGB")
    print(f"Avatar loaded: {avatar.size}")

    # Load backend
    print(f"\nLoading {BACKEND_NAME}...")
    t0 = time.time()
    backend = get_backend(BACKEND_NAME)
    backend.load(cache_dir=os.environ["HF_HUB_CACHE"])
    print(f"Backend loaded in {time.time() - t0:.1f}s")

    # CogVideoX works at 480x720
    target_h, target_w = 480, 720
    ground_state = _frame_to_pil(avatar.resize((target_w, target_h)))

    all_frames = []
    clip_paths = []
    total_start = time.time()

    for i, sentence in enumerate(SENTENCES):
        print(f"\n{'='*60}")
        print(f"Sentence {i+1}/{len(SENTENCES)}: {sentence}")
        print(f"{'='*60}")

        fwd_seed = SEED_BASE + (i * 2)
        rst_seed = SEED_BASE + (i * 2) + 1

        # Forward pass
        print(f"  Forward pass (seed={fwd_seed})...")
        fwd_prompt = (
            f"The character enthusiastically acts out '{sentence}' with big, exaggerated "
            "body movements, arms moving expressively, full-body pantomime gestures. "
            "Smooth animation, dynamic motion."
        )
        t1 = time.time()
        fwd_frames = backend.generate(
            image=ground_state,
            prompt=fwd_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=target_h, width=target_w,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE,
            steps=STEPS,
            seed=fwd_seed,
        )
        fwd_time = time.time() - t1
        print(f"  Forward done: {len(fwd_frames)} frames in {fwd_time:.1f}s")

        # Pin first frame to ground state
        fwd_frames[0] = ground_state.copy()
        fwd_last = fwd_frames[-1].copy()

        # Reset pass
        print(f"  Reset pass (seed={rst_seed})...")
        rst_prompt = (
            f"The same character naturally returns from acting out '{sentence}' "
            "back to the original neutral starting pose, arms lowering to sides. "
            "Smooth animation, gentle motion back to rest."
        )
        t2 = time.time()
        rst_frames = backend.generate(
            image=fwd_last,
            prompt=rst_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=target_h, width=target_w,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE,
            steps=STEPS,
            seed=rst_seed,
        )
        rst_time = time.time() - t2
        print(f"  Reset done: {len(rst_frames)} frames in {rst_time:.1f}s")

        # Pin boundary frames
        rst_frames[0] = fwd_last
        rst_frames[-1] = ground_state.copy()

        # Combine: forward + reset (skip duplicate boundary frame)
        clip_frames = fwd_frames + rst_frames[1:]

        # Save individual clip
        clip_path = os.path.join(OUTPUT_DIR, f"demo_clip_{i:02d}_{sentence.replace(' ', '_')[:30]}.mp4")
        export_to_video(clip_frames, clip_path, fps=FPS)
        clip_paths.append(clip_path)
        print(f"  Clip saved: {clip_path} ({len(clip_frames)} frames)")

        # Save sample frames for inspection
        for fi in [0, len(fwd_frames)//2, len(fwd_frames)-1]:
            fwd_frames[fi].save(os.path.join(OUTPUT_DIR, f"demo_s{i+1}_fwd_f{fi:03d}.png"))

        # Accumulate trajectory (skip frame[0] after first clip to avoid dupes)
        if i == 0:
            all_frames.extend(clip_frames)
        else:
            all_frames.extend(clip_frames[1:])

        elapsed = time.time() - total_start
        remaining_gens = (len(SENTENCES) - i - 1) * 2
        avg_per_gen = elapsed / ((i + 1) * 2)
        eta = remaining_gens * avg_per_gen
        print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    # Save full trajectory
    traj_path = os.path.join(OUTPUT_DIR, "demo_trajectory_cogvideo.mp4")
    export_to_video(all_frames, traj_path, fps=FPS)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"DONE! {len(SENTENCES)} sentences, {len(all_frames)} total frames")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Trajectory: {traj_path}")
    print(f"Individual clips: {clip_paths}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

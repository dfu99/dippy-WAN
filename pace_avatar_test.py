"""Test AI-generated avatars with WAN and LTX backends on PACE.

Runs a single charades sentence (forward + reset) for each avatar image
found in avatars/. Saves keyframes and per-avatar clips to results/.

Usage:
    python pace_avatar_test.py --backend wan14b --cache-dir /path/to/cache
    python pace_avatar_test.py --backend ltx2b --cache-dir /path/to/cache
"""
import argparse
import glob
import os
import time

import torch
import numpy as np
from PIL import Image

from backends import get_backend


TEST_SENTENCE = "He jumped"


def parse_args():
    p = argparse.ArgumentParser(description="Dippy avatar comparison test")
    p.add_argument("--backend", default="wan14b")
    p.add_argument("--cache-dir", default=".hf_cache")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--avatar-dir", default="avatars")
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--steps", type=int, default=None,
                   help="Override inference steps (default: backend default)")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="Override guidance scale (default: backend default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sentence", default=TEST_SENTENCE)
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Find avatars
    avatar_paths = sorted(glob.glob(os.path.join(args.avatar_dir, "*.png")))
    if not avatar_paths:
        print(f"No PNG avatars found in {args.avatar_dir}/")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Backend: {args.backend}")
    print(f"Test sentence: \"{args.sentence}\"")
    print(f"Avatars found: {len(avatar_paths)}")
    for p in avatar_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img = Image.open(p)
        print(f"  - {name}: {img.size} ({os.path.getsize(p) / 1024:.0f} KB)")

    # Load backend
    backend = get_backend(args.backend)
    steps = args.steps if args.steps is not None else backend.default_steps
    guidance = args.guidance_scale if args.guidance_scale is not None else backend.default_guidance

    print(f"\nLoading {backend.display_name}...")
    t0 = time.time()
    backend.load(cache_dir=args.cache_dir)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    num_frames = backend.valid_num_frames(args.num_frames)
    print(f"Frames per pass: {num_frames}, steps: {steps}, guidance: {guidance}")

    results = []

    for avatar_path in avatar_paths:
        avatar_name = os.path.splitext(os.path.basename(avatar_path))[0]
        avatar = Image.open(avatar_path).convert("RGB")
        ground_state = avatar.resize((720, 480))

        print(f"\n{'='*60}")
        print(f"Avatar: {avatar_name} (original: {avatar.size})")
        print(f"{'='*60}")

        # Save resized input for reference
        ground_state.save(os.path.join(
            args.output_dir, f"avatar_{args.backend}_{avatar_name}_input.png"))

        # Forward pass
        forward_prompt = (
            f"The character enthusiastically acts out '{args.sentence}' with big, "
            "exaggerated body movements, arms moving expressively, full-body "
            "pantomime gestures. Smooth animation, dynamic motion."
        )
        print(f"Forward pass ({num_frames} frames, {steps} steps)...")
        t1 = time.time()
        fwd_frames = backend.generate(
            image=ground_state,
            prompt=forward_prompt,
            negative_prompt="blurry, static, low quality, distorted",
            height=480, width=720,
            num_frames=num_frames,
            guidance_scale=guidance,
            steps=steps,
            seed=args.seed,
        )
        fwd_time = time.time() - t1
        print(f"  Forward: {len(fwd_frames)} frames in {fwd_time:.1f}s")

        fwd_frames[0] = ground_state.copy()
        last_forward = fwd_frames[-1].copy()

        # Save forward keyframes (first, mid, last)
        for idx, label in [(0, "f000"), (len(fwd_frames)//2, f"f{len(fwd_frames)//2:03d}"),
                           (len(fwd_frames)-1, f"f{len(fwd_frames)-1:03d}")]:
            fwd_frames[idx].save(os.path.join(
                args.output_dir,
                f"avatar_{args.backend}_{avatar_name}_fwd_{label}.png"))

        # Reset pass
        reset_prompt = (
            f"The same character naturally returns from acting out '{args.sentence}' "
            "back to the original neutral starting pose, arms lowering to sides. "
            "Smooth animation, gentle motion back to rest."
        )
        print(f"Reset pass ({num_frames} frames, {steps} steps)...")
        print(f"  Using last_image conditioning (first→last frame generation)")
        t2 = time.time()
        rst_frames = backend.generate(
            image=last_forward,
            prompt=reset_prompt,
            negative_prompt="blurry, static, low quality, distorted",
            height=480, width=720,
            num_frames=num_frames,
            guidance_scale=guidance,
            steps=steps,
            seed=args.seed + 1,
            last_image=ground_state,
        )
        rst_time = time.time() - t2
        print(f"  Reset: {len(rst_frames)} frames in {rst_time:.1f}s")

        rst_frames[0] = last_forward

        # Save the raw (model-generated) last reset frame before any override
        rst_frames[-1].save(os.path.join(
            args.output_dir,
            f"avatar_{args.backend}_{avatar_name}_rst_raw_last.png"))

        # Save reset keyframes
        for idx, label in [(0, "f000"), (len(rst_frames)//2, f"f{len(rst_frames)//2:03d}"),
                           (len(rst_frames)-1, f"f{len(rst_frames)-1:03d}")]:
            rst_frames[idx].save(os.path.join(
                args.output_dir,
                f"avatar_{args.backend}_{avatar_name}_rst_{label}.png"))

        # Chain into clip
        clip_frames = fwd_frames + rst_frames[1:]

        # Save clip video
        try:
            from diffusers.utils import export_to_video
            clip_path = os.path.join(
                args.output_dir, f"avatar_{args.backend}_{avatar_name}_clip.mp4")
            export_to_video(clip_frames, clip_path, fps=backend.fps)
            clip_size = os.path.getsize(clip_path) / (1024 * 1024)
            print(f"  Clip: {clip_path} ({len(clip_frames)} frames, {clip_size:.1f} MB)")
        except Exception as e:
            clip_path = None
            print(f"  Video export failed: {e}")

        results.append({
            "name": avatar_name,
            "fwd_time": fwd_time,
            "rst_time": rst_time,
            "total_time": fwd_time + rst_time,
            "fwd_frames": len(fwd_frames),
            "rst_frames": len(rst_frames),
            "clip_frames": len(clip_frames),
            "clip_path": clip_path,
        })

        print(f"  Total: {fwd_time + rst_time:.1f}s")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Summary
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n{'='*60}")
    print(f"=== Avatar Test Summary ===")
    print(f"Backend: {args.backend} ({backend.display_name})")
    print(f"Sentence: \"{args.sentence}\"")
    print(f"Model load: {load_time:.1f}s")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Steps: {steps}, Guidance: {guidance}, Frames/pass: {num_frames}")
    print(f"")
    total_inference = 0
    for r in results:
        print(f"  {r['name']:15s} — fwd: {r['fwd_time']:.1f}s, rst: {r['rst_time']:.1f}s, "
              f"total: {r['total_time']:.1f}s, frames: {r['clip_frames']}")
        total_inference += r["total_time"]
    print(f"\nTotal inference: {total_inference:.1f}s ({total_inference/60:.1f} min)")

    backend.unload()


if __name__ == "__main__":
    main()

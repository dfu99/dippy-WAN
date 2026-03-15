"""Generate a multi-sentence chained demo video on PACE.

Creates the flagship Dippy demo: multiple charades sentences chained
seamlessly with forward+reset passes for each.

Usage:
    python pace_demo_trajectory.py --backend wan14b --cache-dir /path/to/cache
"""
import argparse
import os
import time

import torch
import numpy as np
from PIL import Image

from backends import get_backend


DEMO_SENTENCES = [
    "He jumped",
    "She laughed",
    "She laughed at him",
    "He ran",
    "He ran because it started raining",
]


def parse_args():
    p = argparse.ArgumentParser(description="Dippy multi-sentence demo")
    p.add_argument("--backend", default="wan14b")
    p.add_argument("--cache-dir", default=".hf_cache")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--image", default=None, help="Avatar image path")
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sentences", nargs="*", default=None,
                   help="Override demo sentences")
    return p.parse_args()


def main():
    args = parse_args()
    sentences = args.sentences if args.sentences else DEMO_SENTENCES

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Backend: {args.backend}")
    print(f"Sentences: {len(sentences)}")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")

    # Load avatar
    if args.image and os.path.exists(args.image):
        avatar = Image.open(args.image).convert("RGB")
        print(f"Avatar: {args.image} ({avatar.size})")
    else:
        avatar = Image.new("RGB", (720, 480), color=(200, 220, 240))
        print("Using synthetic avatar (720x480)")

    # Load backend
    backend = get_backend(args.backend)
    print(f"\nLoading {backend.display_name}...")
    t0 = time.time()
    backend.load(cache_dir=args.cache_dir)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    num_frames = backend.valid_num_frames(args.num_frames)
    ground_state = avatar.resize((720, 480))
    all_frames = []
    clip_times = []

    for i, sentence in enumerate(sentences):
        print(f"\n{'='*60}")
        print(f"Sentence {i+1}/{len(sentences)}: \"{sentence}\"")
        print(f"{'='*60}")

        # Forward pass
        forward_prompt = (
            f"The character enthusiastically acts out '{sentence}' with big, exaggerated "
            "body movements, arms moving expressively, full-body pantomime gestures. "
            "Smooth animation, dynamic motion."
        )
        print(f"Forward pass ({num_frames} frames, {args.steps} steps)...")
        t1 = time.time()
        fwd_frames = backend.generate(
            image=ground_state,
            prompt=forward_prompt,
            negative_prompt="blurry, static, low quality, distorted",
            height=480, width=720,
            num_frames=num_frames,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=args.seed + (i * 2),
        )
        fwd_time = time.time() - t1
        print(f"  Forward: {len(fwd_frames)} frames in {fwd_time:.1f}s")

        # Force first frame to match ground state
        fwd_frames[0] = ground_state.copy()
        last_forward = fwd_frames[-1].copy()

        # Save forward keyframe
        fwd_frames[len(fwd_frames)//2].save(
            os.path.join(args.output_dir, f"demo_s{i+1}_fwd_mid.png"))

        # Reset pass
        reset_prompt = (
            f"The same character naturally returns from acting out '{sentence}' "
            "back to the original neutral starting pose, arms lowering to sides. "
            "Smooth animation, gentle motion back to rest."
        )
        print(f"Reset pass ({num_frames} frames, {args.steps} steps)...")
        t2 = time.time()
        rst_frames = backend.generate(
            image=last_forward,
            prompt=reset_prompt,
            negative_prompt="blurry, static, low quality, distorted",
            height=480, width=720,
            num_frames=num_frames,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=args.seed + (i * 2) + 1,
        )
        rst_time = time.time() - t2
        print(f"  Reset: {len(rst_frames)} frames in {rst_time:.1f}s")

        # Force boundary frames
        rst_frames[0] = last_forward
        rst_frames[-1] = ground_state.copy()

        # Chain: skip first frame of reset (duplicate of forward last)
        clip_frames = fwd_frames + rst_frames[1:]
        clip_times.append(fwd_time + rst_time)

        # Append to trajectory (skip first frame after first clip)
        if i == 0:
            all_frames.extend(clip_frames)
        else:
            all_frames.extend(clip_frames[1:])

        print(f"  Clip: {len(clip_frames)} frames, {fwd_time + rst_time:.1f}s total")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Save individual clip videos
    try:
        from diffusers.utils import export_to_video

        # Save full trajectory
        traj_path = os.path.join(args.output_dir, f"demo_trajectory_{args.backend}.mp4")
        export_to_video(all_frames, traj_path, fps=backend.fps)
        print(f"\nTrajectory: {traj_path} ({len(all_frames)} frames, "
              f"{len(all_frames)/backend.fps:.1f}s)")
    except Exception as e:
        print(f"Video export failed: {e}")

    # Summary
    total_inference = sum(clip_times)
    print(f"\n{'='*60}")
    print(f"=== Demo Summary ===")
    print(f"Backend: {args.backend}")
    print(f"Sentences: {len(sentences)}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Duration: {len(all_frames)/backend.fps:.1f}s at {backend.fps} fps")
    print(f"Model load: {load_time:.1f}s")
    print(f"Total inference: {total_inference:.1f}s ({total_inference/60:.1f} min)")
    for i, (s, t) in enumerate(zip(sentences, clip_times)):
        print(f"  {i+1}. \"{s}\" — {t:.1f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    backend.unload()


if __name__ == "__main__":
    main()

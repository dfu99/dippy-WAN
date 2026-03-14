"""Headless inference script for PACE cluster (no GUI, no Gradio).

Runs a single forward+reset generation cycle and saves frames + video.
Designed to be called from pace_inference.sbatch.

Usage:
    python pace_test_inference.py --backend wan14b --cache-dir /path/to/cache
"""
import argparse
import os
import time

import torch
from PIL import Image

from backends import get_backend


def parse_args():
    p = argparse.ArgumentParser(description="Dippy headless inference")
    p.add_argument("--backend", default="wan14b", help="Backend name (wan14b, cogvideo5b, ltx2b)")
    p.add_argument("--cache-dir", default=".hf_cache", help="HuggingFace cache directory")
    p.add_argument("--output-dir", default="results", help="Output directory for frames and video")
    p.add_argument("--image", default=None, help="Input image path (uses test image if not provided)")
    p.add_argument("--prompt", default="A cartoon character raises both arms overhead and waves with big exaggerated movements",
                   help="Forward prompt")
    p.add_argument("--reset-prompt", default="A cartoon character slowly lowers arms back to sides, returning to a neutral standing pose",
                   help="Reset prompt")
    p.add_argument("--num-frames", type=int, default=49, help="Number of frames per pass")
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--guidance-scale", type=float, default=6.0, help="Guidance scale")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--skip-reset", action="store_true", help="Skip the reset pass")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Backend: {args.backend}")

    # Load input image
    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("RGB")
        print(f"Input image: {args.image} ({img.size})")
    else:
        # Create a simple test image
        img = Image.new("RGB", (720, 480), color=(200, 220, 240))
        print("Using synthetic test image (720x480)")

    # Load backend
    backend = get_backend(args.backend)
    print(f"Loading {backend.display_name}...")
    t0 = time.time()
    backend.load(cache_dir=args.cache_dir)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Validate frame count
    num_frames = backend.valid_num_frames(args.num_frames)
    if num_frames != args.num_frames:
        print(f"Adjusted num_frames {args.num_frames} -> {num_frames} for backend constraints")

    # Forward pass
    print(f"\n--- Forward pass ({num_frames} frames, {args.steps} steps) ---")
    print(f"Prompt: {args.prompt}")
    t1 = time.time()
    fwd_frames = backend.generate(
        image=img,
        prompt=args.prompt,
        negative_prompt="blurry, static, low quality, distorted",
        height=480,
        width=720,
        num_frames=num_frames,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
    )
    fwd_time = time.time() - t1
    print(f"Forward: {len(fwd_frames)} frames in {fwd_time:.1f}s ({fwd_time/len(fwd_frames):.2f}s/frame)")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Save forward frames
    prefix = f"pace_{args.backend}"
    for i, frame in enumerate(fwd_frames):
        if i % max(1, len(fwd_frames) // 5) == 0 or i == len(fwd_frames) - 1:
            path = os.path.join(args.output_dir, f"{prefix}_fwd_f{i:03d}.png")
            frame.save(path)
            print(f"  Saved {path}")

    # Reset pass
    if not args.skip_reset:
        last_forward = fwd_frames[-1]
        print(f"\n--- Reset pass ({num_frames} frames, {args.steps} steps) ---")
        print(f"Prompt: {args.reset_prompt}")
        t2 = time.time()
        reset_frames = backend.generate(
            image=last_forward,
            prompt=args.reset_prompt,
            negative_prompt="blurry, static, low quality, distorted",
            height=480,
            width=720,
            num_frames=num_frames,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=args.seed + 1,
        )
        reset_time = time.time() - t2
        print(f"Reset: {len(reset_frames)} frames in {reset_time:.1f}s ({reset_time/len(reset_frames):.2f}s/frame)")

        # Save reset frames
        for i, frame in enumerate(reset_frames):
            if i % max(1, len(reset_frames) // 5) == 0 or i == len(reset_frames) - 1:
                path = os.path.join(args.output_dir, f"{prefix}_rst_f{i:03d}.png")
                frame.save(path)
                print(f"  Saved {path}")

        # Stitch into loop (skip first frame of reset to avoid duplicate)
        all_frames = fwd_frames + reset_frames[1:]
    else:
        all_frames = fwd_frames
        reset_time = 0

    # Save video
    try:
        from diffusers.utils import export_to_video
        video_path = os.path.join(args.output_dir, f"{prefix}_loop.mp4")
        export_to_video(all_frames, video_path, fps=8)
        print(f"\nSaved video: {video_path} ({len(all_frames)} frames)")
    except Exception as e:
        print(f"Video export failed: {e}")

    # Summary
    total_time = fwd_time + reset_time
    print(f"\n=== Summary ===")
    print(f"Backend: {args.backend}")
    print(f"Model load: {load_time:.1f}s")
    print(f"Forward: {fwd_time:.1f}s")
    if not args.skip_reset:
        print(f"Reset: {reset_time:.1f}s")
    print(f"Total inference: {total_time:.1f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Frames: {len(all_frames)}")

    backend.unload()


if __name__ == "__main__":
    main()

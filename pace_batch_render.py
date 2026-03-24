"""Render a single sentence clip for cache population.

Designed to be called by a SLURM array job. Each task renders one sentence
using the 3-segment pipeline (setup→action→reset) with last_image conditioning.

Usage:
    python pace_batch_render.py --sentence-index 0 --sentence-file sentences.json
    python pace_batch_render.py --sentence "He jumped"
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from backends import get_backend

# Use LLM-generated scene descriptions when available, else fallback to generic
try:
    from orchestrator.scene_gen import (
        build_setup_prompt, build_action_prompt, build_reset_prompt,
    )
    print("Using LLM scene description generator (orchestrator.scene_gen)")
except ImportError:
    # Fallback for PACE where orchestrator may not be installed
    AVATAR_CONTEXT = (
        "A simple vector-art anime-style robotic character, currently bald with no hair. "
        "Versatile actor that can morph, don wigs, hair accessories, hats, costumes, "
        "and sprout props to embody actions. "
        "The background can fade in, swipe, or slide like stage props — "
        "it must transition smoothly, never appear or disappear instantly."
    )

    def build_setup_prompt(sentence):
        return (
            f"{AVATAR_CONTEXT} "
            f"The character prepares to act out '{sentence}' — putting on a relevant "
            "costume or disguise, sprouting props, and letting the background fade in "
            "like a stage set. Smooth preparation, anticipation building."
        )

    def build_action_prompt(sentence):
        return (
            f"{AVATAR_CONTEXT} "
            f"The costumed character fully performs '{sentence}' with maximum energy — "
            "exaggerated full-body movement, peak of the action, dramatic performance."
        )

    def build_reset_prompt(sentence):
        return (
            f"{AVATAR_CONTEXT} "
            f"The character finishes '{sentence}' and smoothly returns to neutral. "
            "Wig and accessories dissolve away, props retract into the body, "
            "costume fades off. Background slides out or fades to empty. "
            "The bald robot returns to its default relaxed standing pose."
        )

    print("Using fallback prompt templates (orchestrator not available)")


def parse_args():
    p = argparse.ArgumentParser(description="Render one sentence for cache")
    p.add_argument("--sentence", type=str, default=None, help="Direct sentence")
    p.add_argument("--sentence-index", type=int, default=None,
                   help="Index into sentence file (for SLURM array)")
    p.add_argument("--sentence-file", default="sentences_top50.json")
    p.add_argument("--backend", default="wan14b")
    p.add_argument("--cache-dir", default=".hf_cache")
    p.add_argument("--output-dir", default="cache_clips")
    p.add_argument("--image", default="avatars/Perplexity_neutral.png")
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Determine sentence
    if args.sentence:
        sentence = args.sentence
        idx = 0
    elif args.sentence_index is not None:
        with open(args.sentence_file) as f:
            sentences = json.load(f)
        if args.sentence_index >= len(sentences):
            print(f"Index {args.sentence_index} out of range ({len(sentences)} sentences)")
            return
        sentence = sentences[args.sentence_index]
        idx = args.sentence_index
    else:
        # Check SLURM_ARRAY_TASK_ID
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is not None:
            idx = int(task_id)
            with open(args.sentence_file) as f:
                sentences = json.load(f)
            if idx >= len(sentences):
                print(f"SLURM task {idx} out of range ({len(sentences)} sentences)")
                return
            sentence = sentences[idx]
        else:
            print("Provide --sentence, --sentence-index, or run as SLURM array job")
            return

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Batch Render: sentence {idx} ===")
    print(f"Sentence: \"{sentence}\"")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load avatar
    avatar = Image.open(args.image).convert("RGB")
    ground_state = avatar.resize((720, 480))
    print(f"Avatar: {args.image}")

    # Load backend
    backend = get_backend(args.backend)
    print(f"Loading {backend.display_name}...")
    t0 = time.time()
    backend.load(cache_dir=args.cache_dir)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    num_frames = backend.valid_num_frames(args.num_frames)
    seed_base = args.seed + idx * 3

    # ── Segment 1: SETUP ──
    print(f"  Setup ({num_frames} frames)...")
    t1 = time.time()
    setup_frames = backend.generate(
        image=ground_state,
        prompt=build_setup_prompt(sentence),
        negative_prompt="blurry, static, low quality, distorted",
        height=480, width=720,
        num_frames=num_frames,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=seed_base,
    )
    setup_time = time.time() - t1
    setup_frames[0] = ground_state.copy()
    last_setup = setup_frames[-1].copy()
    print(f"    {len(setup_frames)} frames in {setup_time:.1f}s")

    # ── Segment 2: ACTION ──
    print(f"  Action ({num_frames} frames)...")
    t2 = time.time()
    action_frames = backend.generate(
        image=last_setup,
        prompt=build_action_prompt(sentence),
        negative_prompt="blurry, static, low quality, distorted",
        height=480, width=720,
        num_frames=num_frames,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=seed_base + 1,
    )
    action_time = time.time() - t2
    action_frames[0] = last_setup
    last_action = action_frames[-1].copy()
    print(f"    {len(action_frames)} frames in {action_time:.1f}s")

    # ── Segment 3: RESET ──
    print(f"  Reset ({num_frames} frames, last_image → ground state)...")
    t3 = time.time()
    reset_frames = backend.generate(
        image=last_action,
        prompt=build_reset_prompt(sentence),
        negative_prompt="blurry, static, low quality, distorted",
        height=480, width=720,
        num_frames=num_frames,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=seed_base + 2,
        last_image=ground_state,
    )
    reset_time = time.time() - t3
    reset_frames[0] = last_action
    print(f"    {len(reset_frames)} frames in {reset_time:.1f}s")

    total_time = setup_time + action_time + reset_time

    # Stitch into single clip
    clip_frames = setup_frames + action_frames[1:] + reset_frames[1:]

    # Save clip video
    from diffusers.utils import export_to_video

    # Clean sentence for filename
    safe_name = sentence.lower().replace(" ", "_")[:40]
    clip_path = os.path.join(args.output_dir, f"clip_{idx:03d}_{safe_name}.mp4")
    export_to_video(clip_frames, clip_path, fps=backend.fps)

    # Save keyframes
    for frame_idx, label in [(0, "start"), (len(clip_frames)//2, "mid"),
                              (len(clip_frames)-1, "end")]:
        clip_frames[frame_idx].save(
            os.path.join(args.output_dir, f"clip_{idx:03d}_{label}.png"))

    # Save metadata
    meta = {
        "index": idx,
        "sentence": sentence,
        "video_path": clip_path,
        "backend": args.backend,
        "num_frames": len(clip_frames),
        "duration_s": len(clip_frames) / backend.fps,
        "fps": backend.fps,
        "setup_time_s": setup_time,
        "action_time_s": action_time,
        "reset_time_s": reset_time,
        "total_inference_s": total_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "seed": seed_base,
    }
    meta_path = os.path.join(args.output_dir, f"clip_{idx:03d}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Clip: {clip_path} ({len(clip_frames)} frames, {len(clip_frames)/backend.fps:.1f}s)")
    print(f"Inference: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Peak VRAM: {meta['peak_vram_gb']:.2f} GB")

    backend.unload()


if __name__ == "__main__":
    main()

"""Generate a multi-sentence chained demo video on PACE.

3-segment pipeline per sentence:
  1. Setup: character dons disguise, background slides in
  2. Action: character performs the sentence
  3. Reset: props retract, background fades, return to neutral

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

# Avatar context — the character is a versatile, morphable robot.
AVATAR_CONTEXT = (
    "A simple vector-art anime-style robotic character, currently bald with no hair. "
    "Versatile actor that can morph, don wigs, hair accessories, hats, costumes, "
    "and sprout props to embody actions. "
    "The background can fade in, swipe, or slide like stage props — "
    "it must transition smoothly, never appear or disappear instantly."
)

# Per-sentence descriptions split into setup (costume/bg) and action (performance).
SCENE_SETUP = {
    "He jumped": (
        "The bald robot puts on a sporty headband and athletic shoes. "
        "A sky backdrop with clouds slides up from below like a stage curtain. "
        "The robot crouches into a ready position, preparing to jump."
    ),
    "She laughed": (
        "The bald robot dons a curly blonde wig and an oversized pink bow. "
        "A warm golden spotlight fades in from behind like stage lighting. "
        "The robot's expression shifts to a wide grin, starting to chuckle."
    ),
    "She laughed at him": (
        "The bald robot puts on a sassy wig with bangs and a smirk. "
        "A comic spotlight swipes in from the side, illuminating the scene. "
        "The robot turns toward the viewer with a mischievous look."
    ),
    "He ran": (
        "The bald robot puts on bright running shoes and a sweatband. "
        "A park background with trees and a path slides in from the right "
        "like a scrolling stage backdrop. The robot shifts to a running stance."
    ),
    "He ran because it started raining": (
        "Dark storm clouds slowly fade in overhead, first raindrops fall. "
        "A tiny raincoat materializes onto the robot, it pulls out an umbrella. "
        "Puddles begin forming on the ground. The sky darkens gradually."
    ),
}

SCENE_ACTION = {
    "He jumped": (
        "The robot springs upward with a huge athletic leap, legs tucked, "
        "arms thrown overhead. The sky backdrop reveals more height — clouds "
        "rush past. Peak of the jump with motion blur and hang time."
    ),
    "She laughed": (
        "The wigged robot doubles over laughing hysterically, mouth wide open, "
        "tears of joy flying off, body shaking and bouncing with each laugh. "
        "Sparkle effects and comic energy lines radiate outward."
    ),
    "She laughed at him": (
        "The wigged robot points mockingly at the viewer, leaning way back "
        "mid-laugh, one hand on its belly. Exaggerated smug expression. "
        "Sweat drops fly from the embarrassed direction. Finger wagging."
    ),
    "He ran": (
        "The robot sprints at full speed through the park, legs blurring with "
        "motion, arms pumping hard. Speed lines streak behind. Trees and scenery "
        "rush past in the background. Dynamic full-speed running."
    ),
    "He ran because it started raining": (
        "Heavy rain pours down, the robot's umbrella flips inside out in the wind. "
        "The robot runs through splashing puddles, coat flapping, with a panicked "
        "expression. Lightning flashes illuminate the background."
    ),
}


def build_setup_prompt(sentence):
    """Prompt for segment 1: costume up, set the stage."""
    setup = SCENE_SETUP.get(sentence, "")
    if setup:
        return (
            f"{AVATAR_CONTEXT} "
            f"Setting up for '{sentence}': {setup} "
            "Smooth transformation, costume materializing, background sliding in."
        )
    return (
        f"{AVATAR_CONTEXT} "
        f"The character prepares to act out '{sentence}' — putting on a relevant "
        "costume or disguise, sprouting props, and letting the background fade in "
        "like a stage set. Smooth preparation, anticipation building."
    )


def build_action_prompt(sentence):
    """Prompt for segment 2: perform the sentence."""
    action = SCENE_ACTION.get(sentence, "")
    if action:
        return (
            f"{AVATAR_CONTEXT} "
            f"The costumed character performs '{sentence}': {action} "
            "Full commitment to the action, peak performance, maximum expressiveness."
        )
    return (
        f"{AVATAR_CONTEXT} "
        f"The costumed character fully performs '{sentence}' with maximum energy — "
        "exaggerated full-body movement, peak of the action, dramatic performance."
    )


def build_reset_prompt(sentence):
    """Prompt for segment 3: tear down costume and background."""
    return (
        f"{AVATAR_CONTEXT} "
        f"The character finishes '{sentence}' and smoothly returns to neutral. "
        "Wig and accessories dissolve away, props retract into the body, "
        "costume fades off. Background slides out or fades to empty. "
        "The bald robot returns to its default relaxed standing pose."
    )


def parse_args():
    p = argparse.ArgumentParser(description="Dippy 3-segment trajectory demo")
    p.add_argument("--backend", default="wan14b")
    p.add_argument("--cache-dir", default=".hf_cache")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--image", default=None, help="Avatar image path")
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sentences", nargs="*", default=None,
                   help="Override demo sentences")
    return p.parse_args()


def save_keyframes(frames, output_dir, prefix):
    """Save 5 evenly-spaced keyframes."""
    n = len(frames)
    for kidx, klabel in [(0, "f000"),
                          (n // 4, f"f{n // 4:03d}"),
                          (n // 2, f"f{n // 2:03d}"),
                          (3 * n // 4, f"f{3 * n // 4:03d}"),
                          (n - 1, f"f{n - 1:03d}")]:
        frames[kidx].save(os.path.join(output_dir, f"{prefix}_{klabel}.png"))


def main():
    args = parse_args()
    sentences = args.sentences if args.sentences else DEMO_SENTENCES

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Backend: {args.backend}")
    print(f"Pipeline: 3-segment (setup → action → reset)")
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

        seed_base = args.seed + (i * 3)

        # ── Segment 1: SETUP (costume + background) ──
        setup_prompt = build_setup_prompt(sentence)
        print(f"  Setup ({num_frames} frames)...")
        t1 = time.time()
        setup_frames = backend.generate(
            image=ground_state,
            prompt=setup_prompt,
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
        save_keyframes(setup_frames, args.output_dir, f"demo_s{i+1}_setup")
        print(f"    {len(setup_frames)} frames in {setup_time:.1f}s")

        # ── Segment 2: ACTION (perform the sentence) ──
        action_prompt = build_action_prompt(sentence)
        print(f"  Action ({num_frames} frames)...")
        t2 = time.time()
        action_frames = backend.generate(
            image=last_setup,
            prompt=action_prompt,
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
        save_keyframes(action_frames, args.output_dir, f"demo_s{i+1}_action")
        print(f"    {len(action_frames)} frames in {action_time:.1f}s")

        # ── Segment 3: RESET (remove costume, clear background) ──
        reset_prompt = build_reset_prompt(sentence)
        print(f"  Reset ({num_frames} frames, last_image → ground state)...")
        t3 = time.time()
        reset_frames = backend.generate(
            image=last_action,
            prompt=reset_prompt,
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
        save_keyframes(reset_frames, args.output_dir, f"demo_s{i+1}_reset")
        print(f"    {len(reset_frames)} frames in {reset_time:.1f}s")

        total_time = setup_time + action_time + reset_time
        clip_times.append(total_time)

        # Chain: setup + action[1:] + reset[1:] (skip boundary dupes)
        clip_frames = setup_frames + action_frames[1:] + reset_frames[1:]

        if i == 0:
            all_frames.extend(clip_frames)
        else:
            all_frames.extend(clip_frames[1:])

        print(f"  Clip: {len(clip_frames)} frames, {total_time:.1f}s total")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Save trajectory video
    try:
        from diffusers.utils import export_to_video
        traj_path = os.path.join(args.output_dir, f"demo_trajectory_{args.backend}.mp4")
        export_to_video(all_frames, traj_path, fps=backend.fps)
        print(f"\nTrajectory: {traj_path} ({len(all_frames)} frames, "
              f"{len(all_frames)/backend.fps:.1f}s)")
    except Exception as e:
        print(f"Video export failed: {e}")

    # Summary
    total_inference = sum(clip_times)
    print(f"\n{'='*60}")
    print(f"=== Demo Summary (3-segment pipeline) ===")
    print(f"Backend: {args.backend}")
    print(f"Sentences: {len(sentences)}")
    print(f"Segments per sentence: 3 (setup → action → reset)")
    print(f"Total frames: {len(all_frames)}")
    print(f"Duration: {len(all_frames)/backend.fps:.1f}s at {backend.fps} fps")
    print(f"Model load: {load_time:.1f}s")
    print(f"Total inference: {total_inference:.1f}s ({total_inference/60:.1f} min)")
    for i, (s, t) in enumerate(zip(sentences, clip_times)):
        print(f"  {i+1}. \"{s}\" — {t:.1f}s (3 segments)")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    backend.unload()


if __name__ == "__main__":
    main()

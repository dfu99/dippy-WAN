"""Quick local inference test for CogVideoX-5B on RTX 3060 12GB."""
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.environ["HF_HOME"] = os.environ["HF_HUB_CACHE"]

import torch
from PIL import Image
from backends import get_backend

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create a simple test image (solid color with a shape)
    img = Image.new("RGB", (720, 480), color=(200, 220, 240))

    backend = get_backend("cogvideo5b")
    print("Loading model...")
    t0 = time.time()
    backend.load(cache_dir=os.environ["HF_HUB_CACHE"])
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

    print("Generating 49 frames...")
    t1 = time.time()
    frames = backend.generate(
        image=img,
        prompt="A cartoon character waves hello with big arm movements",
        negative_prompt="blurry, static, low quality",
        height=480,
        width=720,
        num_frames=49,
        guidance_scale=6.0,
        steps=10,
        seed=42,
    )
    elapsed = time.time() - t1
    print(f"Generated {len(frames)} frames in {elapsed:.1f}s ({elapsed/len(frames):.2f}s/frame)")

    # Save a few sample frames
    os.makedirs("results", exist_ok=True)
    for i in [0, 24, 48]:
        if i < len(frames):
            frames[i].save(f"results/local_test_frame_{i:03d}.png")
            print(f"Saved results/local_test_frame_{i:03d}.png")

    # Save as gif
    from diffusers.utils import export_to_video
    export_to_video(frames, "results/local_test.mp4", fps=8)
    print("Saved results/local_test.mp4")

    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    backend.unload()

if __name__ == "__main__":
    main()

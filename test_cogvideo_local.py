"""Quick local test of CogVideoX-5B backend on RTX 3060 12GB."""
import os
import sys
import time
import torch
from PIL import Image

os.environ.setdefault("DIPPY_BACKEND", "cogvideo5b")
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/hf_cache"))
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

from backends import get_backend

# Force skip quantization for local RTX 3060 test (torchao + cpu_offload incompatible)
os.environ["DIPPY_NO_QUANTIZE"] = "1"

def main():
    # Create a simple test image (solid color with a shape)
    img = Image.new("RGB", (720, 480), (200, 180, 160))
    # Draw a simple stick figure silhouette
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    cx, cy = 360, 200
    draw.ellipse([cx-30, cy-60, cx+30, cy], fill=(80, 60, 50))  # head
    draw.line([(cx, cy), (cx, cy+100)], fill=(80, 60, 50), width=5)  # body
    draw.line([(cx, cy+100), (cx-40, cy+170)], fill=(80, 60, 50), width=4)  # left leg
    draw.line([(cx, cy+100), (cx+40, cy+170)], fill=(80, 60, 50), width=4)  # right leg
    draw.line([(cx, cy+30), (cx-50, cy+80)], fill=(80, 60, 50), width=4)  # left arm
    draw.line([(cx, cy+30), (cx+50, cy+80)], fill=(80, 60, 50), width=4)  # right arm
    img.save("test_input.png")
    print(f"Test image saved: test_input.png (720x480)")

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")
    print(f"VRAM free: {torch.cuda.mem_get_info()[0]/(1024**3):.1f} GB")

    # Load backend
    print("\n--- Loading CogVideoX-5B ---")
    t0 = time.time()
    backend = get_backend("cogvideo5b")
    backend.load(cache_dir=os.environ["HF_HUB_CACHE"])
    load_time = time.time() - t0
    print(f"Load time: {load_time:.1f}s")
    print(f"VRAM used after load: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    # Generate
    print("\n--- Generating (49 frames, 480x720) ---")
    prompt = "A person waves hello with their right hand"
    t0 = time.time()
    frames = backend.generate(
        image=img,
        prompt=prompt,
        negative_prompt="blurry, distorted",
        height=480,
        width=720,
        num_frames=49,
        guidance_scale=6.0,
        steps=30,
        seed=42,
    )
    gen_time = time.time() - t0
    print(f"Generation time: {gen_time:.1f}s")
    print(f"Got {len(frames)} frames")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")

    # Save output
    os.makedirs("results", exist_ok=True)
    for i, f in enumerate(frames):
        f.save(f"results/cogvideo_test_frame_{i:03d}.png")
    print(f"Saved {len(frames)} frames to results/cogvideo_test_frame_*.png")

    # Save as GIF for quick preview
    frames[0].save(
        "results/cogvideo_test.gif",
        save_all=True,
        append_images=frames[1:],
        duration=125,  # 8fps
        loop=0,
    )
    print("Saved results/cogvideo_test.gif")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Backend: cogvideo5b (CogVideoX-5B-I2V, int8)")
    print(f"GPU: {torch.cuda.get_device_name(0)} (12GB)")
    print(f"Load time: {load_time:.1f}s")
    print(f"Generation time: {gen_time:.1f}s ({gen_time/49:.1f}s/frame)")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")
    print(f"Frames: {len(frames)} @ 8fps = {len(frames)/8:.1f}s clip")

if __name__ == "__main__":
    main()

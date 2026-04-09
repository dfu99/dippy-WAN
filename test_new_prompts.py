"""Test the updated action-style prompt template from dippy-app.py."""
import os, sys, time, torch
from PIL import Image, ImageDraw

os.environ['DIPPY_BACKEND'] = 'cogvideo5b'
os.environ['HF_HUB_CACHE'] = os.path.expanduser('~/hf_cache')
os.environ['DIPPY_NO_QUANTIZE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, '/home2/Documents/code/dippy-WAN')
from backends import get_backend

# Create cartoon avatar (same as pipeline test)
img = Image.new('RGB', (720, 480), (240, 240, 235))
draw = ImageDraw.Draw(img)
cx, cy = 360, 180
draw.ellipse([cx-40, cy-50, cx+40, cy+30], fill=(255, 220, 185), outline=(180, 140, 100), width=2)
draw.ellipse([cx-20, cy-20, cx-10, cy-10], fill=(60, 40, 30))
draw.ellipse([cx+10, cy-20, cx+20, cy-10], fill=(60, 40, 30))
draw.arc([cx-15, cy, cx+15, cy+15], 0, 180, fill=(180, 80, 80), width=2)
draw.rectangle([cx-8, cy+30, cx+8, cy+50], fill=(255, 220, 185))
draw.rectangle([cx-50, cy+50, cx+50, cy+150], fill=(70, 130, 200))
draw.rectangle([cx-80, cy+55, cx-50, cy+130], fill=(70, 130, 200))
draw.rectangle([cx+50, cy+55, cx+80, cy+130], fill=(70, 130, 200))
draw.ellipse([cx-85, cy+120, cx-55, cy+145], fill=(255, 220, 185))
draw.ellipse([cx+55, cy+120, cx+85, cy+145], fill=(255, 220, 185))
draw.rectangle([cx-50, cy+150, cx-5, cy+250], fill=(50, 50, 80))
draw.rectangle([cx+5, cy+150, cx+50, cy+250], fill=(50, 50, 80))
draw.ellipse([cx-55, cy+240, cx-5, cy+265], fill=(40, 40, 40))
draw.ellipse([cx+5, cy+240, cx+55, cy+265], fill=(40, 40, 40))
draw.ellipse([cx-42, cy-55, cx+42, cy-20], fill=(80, 50, 30))

print(f"GPU: {torch.cuda.get_device_name(0)}")
backend = get_backend('cogvideo5b')
backend.load(cache_dir=os.environ['HF_HUB_CACHE'])

sentence = "hello, nice to meet you"
ground_state = img.copy()
seed = 42

# Forward pass — NEW action-style prompt (from updated dippy-app.py)
print(f"\n=== Forward (action-style): '{sentence}' ===")
forward_prompt = (
    f"The character enthusiastically acts out '{sentence}' with big, exaggerated "
    "body movements, arms moving expressively, full-body pantomime gestures. "
    "Smooth animation, dynamic motion."
)
print(f"Prompt: {forward_prompt}")
t0 = time.time()
forward_frames = backend.generate(
    image=ground_state, prompt=forward_prompt,
    negative_prompt="blurry, distorted, deformed",
    height=480, width=720, num_frames=49,
    guidance_scale=6.0, steps=10, seed=seed,
)
fwd_time = time.time() - t0
forward_frames[0] = ground_state.copy()
forward_last = forward_frames[-1].copy()
print(f"Forward: {fwd_time:.0f}s, {len(forward_frames)} frames")

# Reset pass — NEW prompt
print(f"\n=== Reset: return to neutral ===")
reset_prompt = (
    f"The same character naturally returns from acting out '{sentence}' "
    "back to the original neutral starting pose, arms lowering to sides. "
    "Smooth animation, gentle motion back to rest."
)
print(f"Prompt: {reset_prompt}")
t0 = time.time()
reset_frames = backend.generate(
    image=forward_last, prompt=reset_prompt,
    negative_prompt="blurry, distorted, deformed",
    height=480, width=720, num_frames=49,
    guidance_scale=6.0, steps=10, seed=seed+1,
)
rst_time = time.time() - t0
reset_frames[0] = forward_last
reset_frames[-1] = ground_state.copy()
print(f"Reset: {rst_time:.0f}s, {len(reset_frames)} frames")

# Combine
all_frames = forward_frames + reset_frames[1:]
print(f"\nTotal: {len(all_frames)} frames, {len(all_frames)/8:.1f}s at 8fps")
print(f"Total time: {fwd_time+rst_time:.0f}s ({(fwd_time+rst_time)/60:.1f} min)")

# Save
os.makedirs('results', exist_ok=True)
all_frames[0].save('results/new_prompt_pipeline.gif',
    save_all=True, append_images=all_frames[1:], duration=125, loop=0)
for i in [0, 16, 24, 32, 48, 72, 96]:
    if i < len(all_frames):
        all_frames[i].save(f'results/new_prompt_f{i:03d}.png')
print(f"Saved results/new_prompt_pipeline.gif + key frames")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")

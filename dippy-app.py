import os

# ── Cache setup (must be set BEFORE importing HF libraries) ──────────────────
CACHE_DIR = "/content/drive/My Drive/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel
import gradio as gr
import tempfile
import spaces
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import random
import openai

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

MOD_VALUE = 32
DEFAULT_H_SLIDER_VALUE = 512
DEFAULT_W_SLIDER_VALUE = 896
NEW_FORMULA_MAX_AREA = 480.0 * 832.0

SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 24
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

DEFAULT_CLIP_DURATION = 2.0  # seconds → 48 frames at 24fps

default_negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards, watermark, text, signature"
)

DEFAULT_AVATAR_URL = "https://upload.wikimedia.org/wikipedia/en/d/db/Clippy-letter.PNG"

# ── Model Loading ────────────────────────────────────────────────────────────

image_encoder = CLIPVisionModel.from_pretrained(
    MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32, cache_dir=CACHE_DIR
)
vae = AutoencoderKLWan.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE_DIR
)
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID, vae=vae, image_encoder=image_encoder,
    torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=8.0
)
pipe.to("cuda")

causvid_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME, cache_dir=CACHE_DIR)
pipe.load_lora_weights(causvid_path, adapter_name="causvid_lora")
pipe.set_adapters(["causvid_lora"], adapter_weights=[0.95])
pipe.fuse_lora()

# Load default avatar image at startup
try:
    import urllib.request
    _tmp_avatar = os.path.join(tempfile.gettempdir(), "dippy_default_avatar.png")
    if not os.path.exists(_tmp_avatar):
        urllib.request.urlretrieve(DEFAULT_AVATAR_URL, _tmp_avatar)
    DEFAULT_AVATAR_IMAGE = Image.open(_tmp_avatar).convert("RGB")
except Exception:
    DEFAULT_AVATAR_IMAGE = None

# ── Helper Functions ─────────────────────────────────────────────────────────

def _calculate_new_dimensions_wan(pil_image, mod_val, calculation_max_area,
                                  min_slider_h, max_slider_h,
                                  min_slider_w, max_slider_w,
                                  default_h, default_w):
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w
    aspect_ratio = orig_h / orig_w
    calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
    calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))
    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)
    new_h = int(np.clip(calc_h, min_slider_h, (max_slider_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_slider_w, (max_slider_w // mod_val) * mod_val))
    return new_h, new_w


def handle_image_upload_for_dims_wan(uploaded_pil_image, current_h_val, current_w_val):
    if uploaded_pil_image is None:
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)
    try:
        new_h, new_w = _calculate_new_dimensions_wan(
            uploaded_pil_image, MOD_VALUE, NEW_FORMULA_MAX_AREA,
            SLIDER_MIN_H, SLIDER_MAX_H, SLIDER_MIN_W, SLIDER_MAX_W,
            DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE
        )
        return gr.update(value=new_h), gr.update(value=new_w)
    except Exception:
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)


def generate_sentences_llm(theme):
    """Call OpenAI to generate 10 simple, visually actable sentences from a theme."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise gr.Error(
            "OPENAI_API_KEY not set. Please add it to Colab secrets or environment."
        )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate simple sentences for a language learning app. "
                    "Each sentence should describe a single, visually actable action "
                    "that a cartoon character could pantomime. Keep sentences short "
                    "(5-10 words). Return exactly 10 sentences, one per line, with "
                    "no numbering or extra formatting."
                ),
            },
            {"role": "user", "content": f"Theme: {theme}"},
        ],
        temperature=0.9,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def navigate(clips, idx):
    """Return UI state for a given clip index."""
    if not clips or idx < 0 or idx >= len(clips):
        return None, "", "", gr.update(interactive=False), gr.update(interactive=False)
    clip = clips[idx]
    video_path = clip["path"]
    sentence = clip["sentence"]
    total = len(clips)
    sentence_md = f"**Sentence {idx + 1}/{total}:** {sentence}"
    indicator_md = f"Clip {idx + 1} of {total}"
    prev_interactive = idx > 0
    next_interactive = idx < total - 1
    return video_path, sentence_md, indicator_md, gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive)


def go_prev(clips, idx):
    new_idx = max(0, idx - 1)
    outputs = navigate(clips, new_idx)
    return (new_idx, *outputs)


def go_next(clips, idx):
    new_idx = min(len(clips) - 1, idx + 1) if clips else 0
    outputs = navigate(clips, new_idx)
    return (new_idx, *outputs)


def jump_to(clips, slider_val):
    idx = max(0, int(slider_val) - 1)
    if clips:
        idx = min(idx, len(clips) - 1)
    outputs = navigate(clips, idx)
    return (idx, *outputs)


# ── Core Generation ──────────────────────────────────────────────────────────

@spaces.GPU(duration=600)
def generate_trajectory(
    input_image, sentences_text, height, width,
    negative_prompt, duration_seconds, guidance_scale, steps,
    seed, randomize_seed,
    progress=gr.Progress(track_tqdm=True),
):
    """Chain WAN I2V generations with frame continuity across sentences."""
    sentences = [s.strip() for s in sentences_text.strip().splitlines() if s.strip()]
    if not sentences:
        raise gr.Error("Please enter at least one sentence.")

    # Use uploaded image or fall back to default avatar
    if input_image is None:
        if DEFAULT_AVATAR_IMAGE is not None:
            input_image = DEFAULT_AVATAR_IMAGE.copy()
        else:
            raise gr.Error("Please upload a starting avatar image.")

    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    num_frames = np.clip(
        int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL
    )
    base_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    current_image = input_image.resize((target_w, target_h))
    clips = []
    all_frames = []

    for i, sentence in enumerate(sentences):
        progress(i / len(sentences), desc=f"Generating clip {i + 1}/{len(sentences)}")

        prompt = (
            f"A character acts out: {sentence}. "
            "Smooth animation, pantomime, expressive gestures."
        )
        clip_seed = base_seed + i

        with torch.inference_mode():
            output_frames = pipe(
                image=current_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=target_h,
                width=target_w,
                num_frames=num_frames,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(clip_seed),
            ).frames[0]

        # Save individual clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            clip_path = tmp.name
        export_to_video(output_frames, clip_path, fps=FIXED_FPS)

        clips.append({"path": clip_path, "sentence": sentence})

        # Accumulate frames for full trajectory (skip frame 0 for clips 1+ to
        # avoid duplicate boundary frame)
        if i == 0:
            all_frames.extend(output_frames)
        else:
            all_frames.extend(output_frames[1:])

        # Last frame becomes input for next clip (frame continuity)
        current_image = output_frames[-1]
        if not isinstance(current_image, Image.Image):
            current_image = Image.fromarray(np.array(current_image))

    # Export full stitched trajectory
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        trajectory_path = tmp.name
    export_to_video(all_frames, trajectory_path, fps=FIXED_FPS)

    # Navigate to first clip
    first_nav = navigate(clips, 0)
    total = len(clips)

    return (
        clips,           # state: clip list
        0,               # state: current index
        first_nav[0],    # clip_video
        first_nav[1],    # sentence_md
        first_nav[2],    # indicator_md
        first_nav[3],    # prev_btn interactive
        first_nav[4],    # next_btn interactive
        trajectory_path, # full_video
        gr.update(maximum=total, value=1),  # timeline slider
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Dippy Animation Trajectory Generator") as demo:
    gr.Markdown("# Dippy Animation Trajectory Generator")
    gr.Markdown(
        "Generate chained video animations with frame continuity — "
        "the last frame of each clip becomes the first frame of the next, "
        "creating a seamless visual trajectory."
    )

    # Hidden state
    clips_state = gr.State([])
    current_idx = gr.State(0)

    with gr.Row():
        # ── Left Column: Inputs ──────────────────────────────────────────
        with gr.Column(scale=1):
            input_image_component = gr.Image(
                type="pil",
                label="Starting Avatar Image (or leave empty for default)",
            )

            theme_input = gr.Textbox(
                label="Theme",
                placeholder="e.g. animals at the zoo",
            )
            generate_sentences_btn = gr.Button("Generate Sentences (LLM)")

            sentences_input = gr.Textbox(
                label="Sentences (one per line)",
                lines=12,
                placeholder="Enter sentences here, one per line...",
            )

            duration_input = gr.Slider(
                minimum=round(MIN_FRAMES_MODEL / FIXED_FPS, 1),
                maximum=round(MAX_FRAMES_MODEL / FIXED_FPS, 1),
                step=0.1,
                value=DEFAULT_CLIP_DURATION,
                label="Clip Duration (seconds)",
                info=f"Per-clip duration. {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.",
            )
            steps_slider = gr.Slider(
                minimum=1, maximum=30, step=1, value=4,
                label="Inference Steps",
            )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    value=default_negative_prompt,
                    lines=3,
                )
                seed_input = gr.Slider(
                    label="Seed", minimum=0, maximum=MAX_SEED,
                    step=1, value=42, interactive=True,
                )
                randomize_seed_checkbox = gr.Checkbox(
                    label="Randomize seed", value=True, interactive=True,
                )
                with gr.Row():
                    height_input = gr.Slider(
                        minimum=SLIDER_MIN_H, maximum=SLIDER_MAX_H,
                        step=MOD_VALUE, value=DEFAULT_H_SLIDER_VALUE,
                        label=f"Height (multiple of {MOD_VALUE})",
                    )
                    width_input = gr.Slider(
                        minimum=SLIDER_MIN_W, maximum=SLIDER_MAX_W,
                        step=MOD_VALUE, value=DEFAULT_W_SLIDER_VALUE,
                        label=f"Width (multiple of {MOD_VALUE})",
                    )
                guidance_scale_input = gr.Slider(
                    minimum=0.0, maximum=20.0, step=0.5, value=1.0,
                    label="Guidance Scale",
                )

            generate_btn = gr.Button("Generate Full Trajectory", variant="primary")

        # ── Right Column: Outputs ────────────────────────────────────────
        with gr.Column(scale=1):
            sentence_display = gr.Markdown("*No clips generated yet.*")
            clip_video = gr.Video(
                label="Current Clip", autoplay=True, interactive=False,
            )

            with gr.Row():
                prev_btn = gr.Button("<< Prev", interactive=False)
                next_btn = gr.Button("Next >>", interactive=False)

            timeline_slider = gr.Slider(
                minimum=1, maximum=1, step=1, value=1,
                label="Timeline", interactive=True,
            )
            indicator_display = gr.Markdown("")

            full_video = gr.Video(
                label="Full Trajectory", autoplay=False, interactive=False,
            )

    # ── Event Wiring ─────────────────────────────────────────────────────

    # Auto-set dimensions on image upload / clear
    input_image_component.upload(
        fn=handle_image_upload_for_dims_wan,
        inputs=[input_image_component, height_input, width_input],
        outputs=[height_input, width_input],
    )
    input_image_component.clear(
        fn=handle_image_upload_for_dims_wan,
        inputs=[input_image_component, height_input, width_input],
        outputs=[height_input, width_input],
    )

    # Generate sentences from theme
    generate_sentences_btn.click(
        fn=generate_sentences_llm,
        inputs=[theme_input],
        outputs=[sentences_input],
    )

    # Generate full trajectory
    generate_btn.click(
        fn=generate_trajectory,
        inputs=[
            input_image_component, sentences_input, height_input, width_input,
            negative_prompt_input, duration_input,
            guidance_scale_input, steps_slider, seed_input, randomize_seed_checkbox,
        ],
        outputs=[
            clips_state, current_idx,
            clip_video, sentence_display, indicator_display,
            prev_btn, next_btn,
            full_video, timeline_slider,
        ],
    )

    # Navigation
    nav_outputs = [current_idx, clip_video, sentence_display, indicator_display, prev_btn, next_btn]

    prev_btn.click(
        fn=go_prev,
        inputs=[clips_state, current_idx],
        outputs=nav_outputs,
    )
    next_btn.click(
        fn=go_next,
        inputs=[clips_state, current_idx],
        outputs=nav_outputs,
    )
    timeline_slider.release(
        fn=jump_to,
        inputs=[clips_state, timeline_slider],
        outputs=nav_outputs,
    )

# ── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.queue().launch(share=True)

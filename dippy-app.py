import os
import platform
import re
import stat
import sys
from pathlib import Path
from urllib.request import urlretrieve
from importlib.metadata import version as pkg_version

# ── Cache setup (must be set BEFORE importing HF libraries) ──────────────────
DEFAULT_LOCAL_CACHE_DIR = "/content/hf_cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _safe_pkg_version(name):
    try:
        return pkg_version(name)
    except Exception:
        return None


def _version_tuple(version_str):
    parts = re.split(r"[.+-]", version_str)
    values = []
    for part in parts:
        if part.isdigit():
            values.append(int(part))
        else:
            break
    return tuple(values)


def _check_diffusers_transformers_compat():
    """
    Transformers 5 removed FLAX_WEIGHTS_NAME from transformers.utils.
    Diffusers < 0.35.2 may still import it.
    """
    diffusers_v = _safe_pkg_version("diffusers")
    transformers_v = _safe_pkg_version("transformers")
    if not diffusers_v or not transformers_v:
        return
    if _version_tuple(transformers_v) >= (5, 0, 0) and _version_tuple(diffusers_v) < (0, 35, 2):
        raise RuntimeError(
            "Incompatible package versions detected: "
            f"diffusers=={diffusers_v}, transformers=={transformers_v}. "
            "Use diffusers>=0.35.2 with transformers>=5, or downgrade transformers to <5."
        )


def _configure_cache_dirs():
    cache_dir = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("DIPPY_CACHE_DIR")
        or DEFAULT_LOCAL_CACHE_DIR
    )
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", cache_dir)
    os.environ.setdefault("HF_ASSETS_CACHE", os.path.join(cache_dir, "assets"))
    os.environ.setdefault("HF_XET_CACHE", os.path.join(cache_dir, "xet"))
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE"):
        os.makedirs(os.environ[key], exist_ok=True)
    return os.environ["HF_HUB_CACHE"]


CACHE_DIR = _configure_cache_dirs()

import torch
_check_diffusers_transformers_compat()

# Import diffusers utilities (export_to_video is backend-agnostic)
from diffusers.utils import export_to_video
import gradio as gr
import tempfile
try:
    import spaces
except ImportError:
    # Not on HF Spaces — provide a no-op decorator
    class _FakeSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    spaces = _FakeSpaces()
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import random
import openai

# ── Import backends ───────────────────────────────────────────────────────────

from backends import get_backend, available_backends, _frame_to_pil

# ── Constants ────────────────────────────────────────────────────────────────

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

DEFAULT_CLIP_DURATION = 2.0  # seconds

default_negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards, watermark, text, signature"
)

DEFAULT_AVATAR_URL = "https://upload.wikimedia.org/wikipedia/en/d/db/Clippy-letter.PNG"
PINNED_DIFFUSERS_VERSION = "0.36.0"
PINNED_TRANSFORMERS_VERSION = "5.1.0"
PINNED_ACCELERATE_VERSION = "1.12.0"
PINNED_HF_HUB_VERSION = "1.4.1"
PINNED_GRADIO_VERSION = "6.5.1"
COLAB_PINNED_INSTALL_CMD = (
    "!pip install -q --upgrade "
    f"diffusers=={PINNED_DIFFUSERS_VERSION} "
    f"transformers=={PINNED_TRANSFORMERS_VERSION} "
    f"accelerate=={PINNED_ACCELERATE_VERSION} "
    f"huggingface_hub=={PINNED_HF_HUB_VERSION} "
    f"gradio=={PINNED_GRADIO_VERSION} "
    "safetensors sentencepiece peft ftfy imageio-ffmpeg opencv-python"
)

# ── Backend selection ─────────────────────────────────────────────────────────

# Select backend via env var: DIPPY_BACKEND=cogvideo5b | ltx2b | wan14b
DEFAULT_BACKEND = os.environ.get("DIPPY_BACKEND", "wan14b")

# Active backend instance (loaded lazily on first generation or at startup)
_active_backend = None
_active_backend_name = None


def _get_or_load_backend(name=None):
    """Get the active backend, loading it if needed."""
    global _active_backend, _active_backend_name
    if name is None:
        name = _active_backend_name or DEFAULT_BACKEND
    if _active_backend is not None and _active_backend_name == name:
        return _active_backend
    # Unload previous backend
    if _active_backend is not None:
        print(f"Unloading {_active_backend_name}...")
        _active_backend.unload()
    print(f"Loading backend: {name}")
    _active_backend = get_backend(name)
    _active_backend.load(cache_dir=CACHE_DIR)
    _active_backend_name = name
    return _active_backend


def switch_backend(name):
    """Switch to a different backend (called from UI dropdown)."""
    backend = _get_or_load_backend(name)
    return (
        gr.update(value=backend.default_steps),
        gr.update(value=backend.default_guidance),
        f"**Active:** {backend.display_name} | VRAM: {backend.vram_gb} | "
        f"FPS: {backend.fps} | {backend.description}"
    )


# ── Utility Functions ─────────────────────────────────────────────────────────

def _print_runtime_versions():
    print("Cache directories:")
    print(f"- HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"- HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    print("Runtime package versions:")
    for pkg in ("diffusers", "transformers", "accelerate", "huggingface_hub", "gradio"):
        try:
            print(f"- {pkg}: {pkg_version(pkg)}")
        except Exception:
            print(f"- {pkg}: not installed")


def _prune_tiny_safetensors(cache_dir, repo_id, min_bytes=16 * 1024):
    repo_cache_dir = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
    if not os.path.isdir(repo_cache_dir):
        return
    removed = []
    for root, _, filenames in os.walk(repo_cache_dir):
        for filename in filenames:
            if not filename.endswith(".safetensors"):
                continue
            path = os.path.join(root, filename)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size < min_bytes:
                try:
                    os.remove(path)
                    removed.append((path, size))
                except OSError:
                    continue
    if removed:
        print(f"Removed {len(removed)} tiny safetensors stubs from {repo_cache_dir}")


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


# ── Gradio frpc tunnel management ─────────────────────────────────────────────

def _parse_version_tuple(version_str):
    return tuple(int(part) for part in version_str.split("."))


def _gradio_platform_tags():
    if sys.platform.startswith("linux"):
        os_tag = "linux"
    elif sys.platform == "darwin":
        os_tag = "darwin"
    elif sys.platform.startswith("win"):
        os_tag = "windows"
    else:
        return None, None
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch_tag = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch_tag = "aarch64"
    else:
        arch_tag = machine
    return os_tag, arch_tag


def _discover_expected_frpc_name(gradio_dir, os_tag, arch_tag):
    pattern = re.compile(rf"frpc_{os_tag}_{arch_tag}_v(\d+\.\d+)")
    matches = []
    for py_file in gradio_dir.glob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in pattern.finditer(text):
            matches.append((m.group(0), m.group(1)))
    if not matches:
        return None
    matches.sort(key=lambda x: _parse_version_tuple(x[1]), reverse=True)
    return matches[0][0]


def _ensure_gradio_frpc_binary():
    try:
        import gradio
    except Exception as exc:
        print(f"Could not import gradio for frpc preflight: {exc}")
        return
    os_tag, arch_tag = _gradio_platform_tags()
    if not os_tag or not arch_tag:
        return
    gradio_dir = Path(gradio.__file__).resolve().parent
    expected_name = _discover_expected_frpc_name(gradio_dir, os_tag, arch_tag)
    candidate_names = []
    if expected_name:
        candidate_names.append(expected_name)
    candidate_names.extend([
        f"frpc_{os_tag}_{arch_tag}_v0.3",
        f"frpc_{os_tag}_{arch_tag}_v0.2",
    ])
    candidate_names = list(dict.fromkeys(candidate_names))
    for name in candidate_names:
        path = gradio_dir / name
        if path.exists():
            try:
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                pass
            return
    for name in candidate_names:
        version = name.rsplit("_v", 1)[-1]
        url = f"https://cdn-media.huggingface.co/frpc-gradio-{version}/frpc_{os_tag}_{arch_tag}"
        dest = gradio_dir / name
        try:
            urlretrieve(url, dest)
            dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return
        except Exception:
            pass


def _launch_with_share(demo):
    in_colab = "google.colab" in sys.modules
    _ensure_gradio_frpc_binary()
    try:
        app, local_url, share_url = demo.queue().launch(
            share=True, inline=in_colab, debug=True, show_error=True,
        )
    except Exception:
        app, local_url, share_url = demo.queue().launch(
            share=False, inline=in_colab, debug=True, show_error=True,
        )
    print("Local URL:", local_url)
    if share_url:
        print("Share URL:", share_url)
    return app, local_url, share_url


# ── Startup ──────────────────────────────────────────────────────────────────

_print_runtime_versions()

# Load default avatar image
try:
    import urllib.request
    _tmp_avatar = os.path.join(tempfile.gettempdir(), "dippy_default_avatar.png")
    if not os.path.exists(_tmp_avatar):
        urllib.request.urlretrieve(DEFAULT_AVATAR_URL, _tmp_avatar)
    DEFAULT_AVATAR_IMAGE = Image.open(_tmp_avatar).convert("RGB")
except Exception:
    DEFAULT_AVATAR_IMAGE = None

# Load the selected backend at startup
_get_or_load_backend(DEFAULT_BACKEND)


# ── Core Generation ──────────────────────────────────────────────────────────

@spaces.GPU(duration=600)
def generate_trajectory(
    input_image, sentences_text, height, width,
    negative_prompt, duration_seconds, guidance_scale, steps,
    seed, randomize_seed, backend_name,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate loopable sentence clips with forward + reset passes."""
    backend = _get_or_load_backend(backend_name)

    sentences = [s.strip() for s in sentences_text.strip().splitlines() if s.strip()]
    if not sentences:
        raise gr.Error("Please enter at least one sentence.")

    if input_image is None:
        if DEFAULT_AVATAR_IMAGE is not None:
            input_image = DEFAULT_AVATAR_IMAGE.copy()
        else:
            raise gr.Error("Please upload a starting avatar image.")

    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)

    requested_frames = int(round(duration_seconds * backend.fps))
    requested_frames = int(np.clip(requested_frames, backend.min_frames, backend.max_frames))
    num_frames = backend.valid_num_frames(requested_frames)
    base_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    ground_state_image = _frame_to_pil(input_image.resize((target_w, target_h)))
    clips = []
    all_frames = []
    total_passes = max(1, len(sentences) * 2)

    for i, sentence in enumerate(sentences):
        forward_seed = base_seed + (i * 2)
        backward_seed = base_seed + (i * 2) + 1

        progress((i * 2) / total_passes, desc=f"Forward pass {i + 1}/{len(sentences)}")
        forward_prompt = (
            f"The character enthusiastically acts out '{sentence}' with big, exaggerated "
            "body movements, arms moving expressively, full-body pantomime gestures. "
            "Smooth animation, dynamic motion."
        )
        forward_frames = backend.generate(
            image=ground_state_image,
            prompt=forward_prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=forward_seed,
        )
        if len(forward_frames) == 0:
            raise gr.Error("Forward generation produced no frames.")
        forward_frames[0] = ground_state_image.copy()
        forward_last_frame = forward_frames[-1].copy()

        progress((i * 2 + 1) / total_passes, desc=f"Reset pass {i + 1}/{len(sentences)}")
        backward_prompt = (
            f"The same character naturally returns from acting out '{sentence}' "
            "back to the original neutral starting pose, arms lowering to sides. "
            "Smooth animation, gentle motion back to rest."
        )
        backward_frames = backend.generate(
            image=forward_last_frame,
            prompt=backward_prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=backward_seed,
        )
        if len(backward_frames) == 0:
            raise gr.Error("Reset generation produced no frames.")
        backward_frames[0] = forward_last_frame
        backward_frames[-1] = ground_state_image.copy()

        output_frames = forward_frames + backward_frames[1:]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            clip_path = tmp.name
        export_to_video(output_frames, clip_path, fps=backend.fps)
        clips.append({"path": clip_path, "sentence": sentence})

        if i == 0:
            all_frames.extend(output_frames)
        else:
            all_frames.extend(output_frames[1:])

    progress(1.0, desc="Finished")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        trajectory_path = tmp.name
    export_to_video(all_frames, trajectory_path, fps=backend.fps)

    first_nav = navigate(clips, 0)
    total = len(clips)

    return (
        clips,
        0,
        first_nav[0],
        first_nav[1],
        first_nav[2],
        first_nav[3],
        first_nav[4],
        trajectory_path,
        gr.update(maximum=total, value=1),
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────

# Build backend choices for dropdown
_backend_choices = []
for _bname in available_backends():
    _b = get_backend(_bname)
    _backend_choices.append((_b.display_name, _bname))

with gr.Blocks(title="Dippy Animation Trajectory Generator") as demo:
    gr.Markdown("# Dippy Animation Trajectory Generator")
    gr.Markdown(
        "Generate sentence clips with a forward action pass plus a reset pass, "
        "so each clip starts and ends in the same ground-state pose for easy reordering."
    )

    # Hidden state
    clips_state = gr.State([])
    current_idx = gr.State(0)

    with gr.Row():
        # ── Left Column: Inputs ──────────────────────────────────────────
        with gr.Column(scale=1):
            # Backend selector
            backend_dropdown = gr.Dropdown(
                choices=_backend_choices,
                value=DEFAULT_BACKEND,
                label="Model Backend",
                info="Select model based on your GPU. CogVideoX/LTX run on T4 (free Colab).",
            )
            backend_info = gr.Markdown(
                f"**Active:** {_active_backend.display_name} | "
                f"VRAM: {_active_backend.vram_gb} | "
                f"FPS: {_active_backend.fps} | "
                f"{_active_backend.description}"
                if _active_backend else ""
            )

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
                minimum=0.3,
                maximum=round(MAX_FRAMES_MODEL / FIXED_FPS, 1),
                step=0.1,
                value=DEFAULT_CLIP_DURATION,
                label="Clip Duration (seconds)",
                info="Per-pass duration. Each sentence runs forward + reset (~2x this length).",
            )
            steps_slider = gr.Slider(
                minimum=1, maximum=50, step=1, value=4,
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

    # Backend switching
    backend_dropdown.change(
        fn=switch_backend,
        inputs=[backend_dropdown],
        outputs=[steps_slider, guidance_scale_input, backend_info],
    )

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
            backend_dropdown,
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
    _launch_with_share(demo)

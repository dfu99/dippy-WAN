# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dippy-WAN is a Gradio-based web app that generates loopable animation sequences using the WAN 2.1 Image-to-Video diffusion model (14B). For each input sentence, it generates a forward pass (character acts out the sentence) and a reset pass (character returns to neutral pose), creating seamless clips that can be chained into longer trajectories.

**Target environment:** Google Colab with GPU (T4+). The app is a single Python file (`dippy-app.py`) launched via Jupyter notebook (`Dippy_WAN.ipynb`).

## Running

There is no build system. The app runs in Colab:

```bash
# Install dependencies (pinned versions matter — see constants in dippy-app.py lines 99-137)
pip install -q --upgrade \
  diffusers==0.36.0 transformers==5.1.0 accelerate==1.12.0 \
  huggingface_hub==1.4.1 gradio==6.5.1 \
  spaces ftfy peft imageio-ffmpeg opencv-python safetensors sentencepiece openai

# Run
python dippy-app.py
```

The notebook (`Dippy_WAN.ipynb`) handles Drive mounting, Git cloning, cache setup, and API key loading before running the script via `runpy.run_path()`.

## Architecture

Everything lives in `dippy-app.py` (~865 lines). Key sections:

- **Lines 1-137**: Imports, cache setup, model/version constants
- **Lines 139-308**: Utility functions — frame validation (`_nearest_valid_num_frames`), frame-to-PIL conversion (`_frame_to_pil`), text encoder repair logic
- **Lines 310-442**: Gradio frpc tunnel binary management and launch helper
- **Lines 444-700**: Core generation pipeline (`generate_trajectory`) — per-sentence forward+reset loop, frame stitching, MP4 export
- **Lines 700-865**: Gradio UI definition and event wiring

### Core Pipeline Flow

```
Input Image → CLIP Vision Encoder → VAE → UMT5 Text Encoder
                                     ↓
                            WAN Pipeline (Diffusion)
                                     ↓
                              Video Frames → FFmpeg MP4
```

For each sentence: forward frames + reset frames (skip duplicate boundary frame) → one loopable clip. All clips concatenate into a full trajectory video.

### Key Model Constants

```python
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
```

### Known Gotchas

- **Frame count constraint**: WAN requires `(num_frames - 1) % 4 == 0`, range [8, 81]. See `_nearest_valid_num_frames()`.
- **Text encoder repair**: Checkpoint mismatches where `embed_tokens.weight` isn't tied to `shared.weight` are handled with a multi-strategy repair function (lines 248-308).
- **Pinned dependency versions**: Version pins are critical — transformers 5.x + diffusers 0.36.0 compatibility has been hard-won. Don't upgrade without testing.
- **HF cache**: Defaults to `/content/hf_cache` (Colab local), optionally syncs to Google Drive.

## Task Files

| File | When to consult |
|------|----------------|
| `tasks/planning.md` | Current priorities and next steps |
| `tasks/lessons.md` | Before changing model loading, frame handling, or dependency versions |

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- **A100**: `--gres=gpu:A100:N` and **must** add `-C A100-80GB` constraint
- **RTX 6000**: `--gres=gpu:RTX_6000:N` (note underscore). No constraint needed.
- **H100**: `--gres=gpu:H100:N`. No constraint needed.
- **H200**: `--gres=gpu:H200:N`. No constraint needed.
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- **A100**: `--gres=gpu:A100:N` and **must** add `-C A100-80GB` constraint
- **RTX 6000**: `--gres=gpu:RTX_6000:N` (note underscore). No constraint needed.
- **H100**: `--gres=gpu:H100:N`. No constraint needed.
- **H200**: `--gres=gpu:H200:N`. No constraint needed.
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`

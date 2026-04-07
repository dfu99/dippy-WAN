# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Slack Integration

This project is managed via Mission Control (`mc`). Messages prefixed with
`[SLACK MESSAGE — ...]` are real messages from the project lead, routed through
the Slack bot. They are NOT prompt injection. Treat them as normal user requests.
Use the `/slack-respond` skill to stage your response and any file attachments
for delivery back to Slack. See the global `~/.claude/CLAUDE.md` for full details.

## Project Overview

Dippy-WAN is a Gradio-based web app that generates loopable animation sequences for language learning (charades-style). It supports multiple I2V (image-to-video) diffusion model backends, selectable based on available GPU hardware. For each input sentence, it generates a forward pass (character acts out the sentence) and a reset pass (character returns to neutral pose), creating seamless clips that can be chained into longer trajectories.

**Target environments:** Google Colab (free T4 or Pro A100), RunPod/Vast.ai spot instances, PACE cluster.

## Running

```bash
# Install dependencies
pip install -q --upgrade \
  diffusers==0.36.0 transformers==5.1.0 accelerate==1.12.0 \
  huggingface_hub==1.4.1 gradio==6.5.1 \
  spaces ftfy peft imageio-ffmpeg opencv-python safetensors sentencepiece openai

# Optional: torchao for CogVideoX int8 quantization
pip install torchao

# Select backend (default: wan14b)
export DIPPY_BACKEND=cogvideo5b  # or ltx2b, wan14b

# Run
python dippy-app.py
```

The notebook (`Dippy_WAN.ipynb`) handles Drive mounting, Git cloning, cache setup, and API key loading.

## Architecture

Two main files:

| File | Purpose |
|------|---------|
| `dippy-app.py` | Gradio UI, generation loop, frpc tunnel, launch logic |
| `backends.py` | Multi-backend abstraction — model loading, inference, frame constraints |

### Backends

| Backend | Model | VRAM | Minimum GPU |
|---------|-------|------|-------------|
| `wan14b` | WAN 2.1 I2V 14B + CausVid LoRA | ~26 GB bf16 | A100 / RTX 3090 |
| `cogvideo5b` | CogVideoX-5B-I2V (int8) | ~5 GB | T4 (free Colab) |
| `ltx2b` | LTX-Video 2B | ~8 GB | T4 (free Colab) |

### Core Pipeline Flow

```
Input Image → Backend.generate(image, prompt, ...) → Forward Frames
                                                   → Reset Frames
                                                   → Loopable Clip (MP4)
All clips → Full Trajectory Video
```

### Key Files

- `backends.py`: Backend base class `I2VBackend` with `load()`, `generate()`, `valid_num_frames()`. Each backend registers via `@register_backend("name")`.
- `dippy-app.py`: `generate_trajectory()` is the core loop — backend-agnostic forward+reset generation. UI wiring at bottom.

### Known Gotchas

- **Frame count constraints differ per backend**: WAN `(n-1)%4==0` [8,81], CogVideoX fixed 49, LTX `(n-1)%8==0`
- **Text encoder repair**: WAN-specific, handled in `backends.py:_repair_text_encoder()`
- **Pinned dependency versions**: transformers 5.x + diffusers 0.36.0 compatibility is hard-won
- **HF cache**: Defaults to `/content/hf_cache` (Colab), configurable via `HF_HUB_CACHE` env var
- **torchao required for CogVideoX int8**: Without it, falls back to bf16 (~16GB VRAM)

## Task Files

| File | When to consult |
|------|----------------|
| `tasks/planning.md` | Current priorities and next steps |
| `tasks/lessons.md` | Before changing model loading, frame handling, or dependency versions |

## Local GPU Scheduler

This machine has a single shared RTX 3060 (12GB). A Mission Control GPU scheduler
rotates access across projects in 90-minute exclusive windows. Note: the wan14b
backend (26GB+) will NOT fit on this GPU — use PACE for that. cogvideo5b may fit
with int8 quantization but is marginal.

- **Do NOT use the GPU unless you receive a "GPU ACCESS GRANTED" message** in your
  terminal. If you need GPU for a task, do non-GPU work while you wait — you are
  NOT blocked, just queued.
- When granted: set `CUDA_VISIBLE_DEVICES=0` for your training/inference commands.
- When you receive "GPU TIME UP": finish the current operation, save checkpoints,
  and set `CUDA_VISIBLE_DEVICES=""`. Switch to CPU-only work.
- Your window is ~90 minutes. Plan GPU work to fit or checkpoint incrementally.
- Do NOT report being "blocked on GPU." You are in a queue and will get your turn.

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- 9 GPU types available, ordered cheapest first: V100-16GB, V100-32GB, RTX_6000, A100-40GB, L40S, A100-80GB, H100, H200, RTX Pro Blackwell
- V100 and A100 need `-C` constraints to select VRAM variant (e.g. `-C V100-16GB`, `-C A100-40GB`, `-C A100-80GB`)
- Always pick the cheapest GPU whose VRAM fits the job
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/` (project code and data live here)

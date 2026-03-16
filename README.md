# Dippy-WAN: Animated Charades for Language Learning

Dippy-WAN generates loopable animation sequences where a character acts out sentences — think **charades for language learners**. For each sentence, it produces a forward pass (character performs the action) and a reset pass (character returns to neutral), creating seamless clips that chain into longer trajectories.

Part of the [alinakai](https://github.com/search?q=alinakai) language learning ecosystem.

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
│ Input Image  │────▶│ Forward Pass  │────▶│  Reset Pass  │────▶│ Loopable Clip│
│ (avatar)     │     │ "waves hello" │     │ return→neutral│     │ (MP4)        │
└─────────────┘     └───────────────┘     └──────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────────────────────────┐
                    │   Full Trajectory Video (MP4)    │
                    │  clip1 + clip2 + ... + clipN     │
                    └──────────────────────────────────┘
```

## Current Status

**Phase: Multi-Backend Development** (March 2026)

| Feature | Status |
|---------|--------|
| WAN 2.1 14B I2V pipeline | Done |
| Forward + reset looping | Done |
| Gradio UI with clip navigation | Done |
| LLM sentence generation (GPT-4o-mini) | Done |
| Multi-backend support (WAN / CogVideoX / LTX) | Done |
| CogVideoX-5B backend (T4-friendly) | Done |
| LTX-Video 2B backend (T4-friendly) | Done |
| Colab notebook integration | Done |
| Quality comparison across backends | Pending |
| alinakai integration | Planned |

## Model Backends

The app supports multiple I2V model backends, selectable via dropdown or env var. This lets you develop cheaply on a T4 and deploy with WAN 14B quality when needed.

| Backend | Model | VRAM | GPU | Speed | Quality |
|---------|-------|------|-----|-------|---------|
| `wan14b` | WAN 2.1 I2V 14B + CausVid LoRA | ~26 GB bf16 | A100 / RTX 3090+ | ~3-5 min/clip | Best |
| `cogvideo5b` | CogVideoX-5B-I2V (int8) | ~5 GB int8 | T4 (free Colab) | ~6-10 min/clip | Good |
| `ltx2b` | LTX-Video 2B | ~8 GB fp16 | T4 (free Colab) | ~1-3 min/clip | Fair |

### Cost Comparison

| Platform | GPU | Cost | WAN 14B? | Best For |
|----------|-----|------|----------|----------|
| Colab Free | T4 16GB | $0 | No (use CogVideoX/LTX) | Development iteration |
| Colab Pro+ | A100 40GB | $49.99/mo | Yes | Occasional production runs |
| RunPod Spot | RTX 3090 24GB | ~$0.22/hr | Yes | Cheapest WAN 14B runs |
| Vast.ai | RTX 3090 24GB | ~$0.15/hr | Yes | Batch generation |
| PACE Cluster | A100 80GB | Free (academic) | Yes | Research runs |

## Quick Start

### Colab (recommended for development)

1. Open `Dippy_WAN.ipynb` in Google Colab
2. Run all cells — the notebook handles Drive mounting, Git clone, dependencies, and launch
3. Select a backend in the UI dropdown based on your GPU

### Local / Cloud GPU

```bash
# Install dependencies
pip install -q --upgrade \
  diffusers==0.36.0 transformers==5.1.0 accelerate==1.12.0 \
  huggingface_hub==1.4.1 gradio==6.5.1 \
  spaces ftfy peft imageio-ffmpeg opencv-python safetensors sentencepiece openai

# Optional: install torchao for CogVideoX int8 quantization on T4
pip install torchao

# Select backend (default: wan14b)
export DIPPY_BACKEND=cogvideo5b  # or ltx2b, wan14b

# Run
python dippy-app.py
```

### PACE Cluster (SLURM)

```bash
#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH -t 02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

module load cuda
cd ~/scratch/dippy-WAN
export DIPPY_BACKEND=wan14b
python dippy-app.py
```

## Architecture

```
dippy-app.py      — Gradio UI, generation loop, frpc tunnel management
backends.py       — Multi-backend abstraction (WAN 14B, CogVideoX 5B, LTX-Video 2B)
Dippy_WAN.ipynb   — Colab launcher (Drive mount, deps, env setup)
```

### Pipeline Flow

```
Input Image → Backend.load() → Backend.generate(image, prompt, ...)
                                        │
           ┌────────────────────────────┘
           ▼
   Forward frames ──┐
                    ├──▶ Loopable clip (forward + reset, skip boundary dupe)
   Reset frames ────┘
           │
           ▼
   Concatenate all clips → Full trajectory MP4
```

Each backend encapsulates its own model loading, frame constraints, quantization, and inference logic. The generation loop in `dippy-app.py` is backend-agnostic.

## Milestones

### M1: Core Pipeline (Complete)
- WAN 2.1 14B I2V with CausVid LoRA for faster inference
- Forward + reset loop architecture for seamless clips
- Gradio UI with clip navigation, timeline slider, video export

### M2: Cost Reduction (Complete)
- Multi-backend architecture (`backends.py`)
- CogVideoX-5B-I2V with int8 quantization for T4
- LTX-Video 2B for rapid iteration
- Backend selection via UI dropdown and env var

### M3: Quality & Evaluation (Next)
- Side-by-side quality comparison across backends
- Quantitative metrics (FID, motion consistency)
- Optimal settings per backend (steps, guidance, resolution)
- Demo gallery with representative outputs

### M4: alinakai Integration (Future)
- API endpoint for programmatic clip generation
- Sentence difficulty scoring integration
- User progress tracking with generated clips
- Batch generation pipeline for curriculum content

## Development Tips

- **Use CogVideoX or LTX for iteration** — save WAN 14B for final quality checks
- **Pin dependency versions** — the combination of diffusers 0.36.0 + transformers 5.1.0 was hard-won
- **Frame count constraints differ per model** — WAN needs `(n-1)%4==0`, CogVideoX is fixed at 49, LTX needs `(n-1)%8==0`
- **Text encoder repair** — WAN has a known issue where `embed_tokens.weight` isn't tied to `shared.weight`; the backend handles this automatically

## License

Research use. See individual model licenses:
- WAN 2.1: [Apache 2.0](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)
- CogVideoX: [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
- LTX-Video: [LTX License](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev)

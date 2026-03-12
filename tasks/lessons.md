# Lessons — dippy-WAN

_Hard-won lessons, gotchas, and things that broke before._
_This file is append-mostly. Only remove entries proven wrong._

## Dependencies

- **Pin versions strictly**: diffusers==0.36.0 + transformers==5.1.0 + accelerate==1.12.0 is the tested combo. Transformers 5.x removed `FLAX_WEIGHTS_NAME` which breaks diffusers < 0.35.2.
- **torchao needed for CogVideoX int8**: Without it, CogVideoX-5B needs ~16GB VRAM (won't fit T4 with overhead). With int8_weight_only(), drops to ~5GB.

## Model Loading

- **WAN text encoder repair**: Checkpoint mismatches cause `embed_tokens.weight` to not be tied to `shared.weight`. The `_repair_text_encoder()` function in `backends.py` handles this with a multi-strategy approach (tie_weights → copy values).
- **WAN CausVid LoRA**: Fused at weight 0.95 for faster inference. Must be loaded after pipeline is on CUDA.
- **Prune tiny safetensors**: HF cache can have 79-byte link stubs that prevent proper model loading. `_prune_tiny_safetensors()` cleans these up.

## Frame Constraints

- **WAN**: `(num_frames - 1) % 4 == 0`, range [8, 81]
- **CogVideoX**: Fixed at 49 frames, 8 fps (~6 seconds)
- **LTX-Video**: `(num_frames - 1) % 8 == 0`, range varies by version
- **Boundary frame dedup**: When stitching forward + reset, skip frame[0] of reset pass to avoid duplicate boundary frame.

## Local GPU (RTX 3060 12GB)

- **CogVideoX-5B bf16 OOMs with `enable_model_cpu_offload()`** on 12GB GPUs. Peak usage is ~11GB during inference, overflows with desktop overhead.
- **`enable_sequential_cpu_offload()` works** but is ~10x slower (45s/step vs ~4s/step) — moves individual layers rather than whole submodels.
- **CogVideoX 10 steps ≈ 30 steps quality**: On RTX 3060, 10 steps takes 6 min vs 23 min for 30 steps, with visually comparable output. Use 10 steps for local dev, 30 for final renders.
- **torchao int8 + cpu_offload incompatible**: torchao 0.11 quantized tensors (`AffineQuantizedTensor`) can't be moved between CPU/GPU by accelerate hooks. torchao 0.16 has a logger bug in diffusers 0.36.0. Set `DIPPY_NO_QUANTIZE=1` as workaround.
- **Missing deps for CogVideoX tokenizer**: needs `tiktoken` and `protobuf` beyond the base pip install list.
- **Conda env setup**: Created `dippy` env cloned from `torch-py312` (torch 2.10+cu128). diffusers 0.36.0 + transformers 5.1.0 work.
- **LTX-Video download crashes shell**: Downloading `Lightricks/LTX-Video-0.9.7-dev` (22 files) OOM-killed the process and broke the shell session entirely (exit code 120, all subsequent commands fail). Set HF_TOKEN to avoid rate-limited retries, and don't run other heavy processes concurrently.

## Cost

- **Colab Pro+ A100 burned through credits in 1 day** with WAN 14B — each trajectory (multiple sentences × 2 passes) uses 5-10+ minutes of A100 time
- **RunPod RTX 3090 spot at $0.22/hr** is ~10x cheaper per GPU-hour than Colab Pro+
- **Free Colab T4** works with CogVideoX-5B (int8) and LTX-Video 2B for development iteration

## Gradio

- **frpc binary**: Colab share links need the frpc tunnel binary. The `_ensure_gradio_frpc_binary()` function auto-downloads it if missing.
- **`spaces` import**: Outside HF Spaces, `import spaces` fails. The app provides a no-op fallback decorator.

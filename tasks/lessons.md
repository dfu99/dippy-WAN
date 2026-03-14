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
- **LTX-Video download crashes shell**: Downloading `Lightricks/LTX-Video-0.9.7-dev` (22 files, 35GB) OOM-killed the process and broke the shell session entirely (exit code 120). Set HF_TOKEN and don't run other heavy processes concurrently.
- **LTX-Video 2B barely animates**: On RTX 3060 with sequential offload, LTX generates near-static frames from a stick figure input. CogVideoX produces dramatically better I2V results. LTX may work better with different inputs or as T2V, but for Dippy's I2V charades use case, CogVideoX-5B is the clear local choice.
- **LTX also needs sequential offload on 12GB**: `enable_model_cpu_offload()` OOMs at 11.25GB. Same fix as CogVideoX — auto-detect VRAM < 14GB.

## Prompt Engineering (CogVideoX-5B)

- **Action-style prompts produce arm/body gestures**: Generic prompts ("waves hello") only produce facial changes. Explicit body descriptions ("raises both arms overhead and waves, big exaggerated waving motion, arms moving up and down") produce dramatic full-body gestures. Always include explicit physical movement descriptions in forward prompts.
- **guidance_scale=6.0 is optimal**: Higher values (9.0, 12.0) reduce motion and add blur/artifacts. Don't increase guidance to try to get more motion — use better prompts instead.

## Cost

- **Colab Pro+ A100 burned through credits in 1 day** with WAN 14B — each trajectory (multiple sentences × 2 passes) uses 5-10+ minutes of A100 time
- **RunPod RTX 3090 spot at $0.22/hr** is ~10x cheaper per GPU-hour than Colab Pro+
- **Free Colab T4** works with CogVideoX-5B (int8) and LTX-Video 2B for development iteration

## PACE Cluster

- **System Python has no torch**: PACE login/compute nodes have Python 3.9 system-wide but no ML packages. Must create a venv with `module load anaconda3` then install PyTorch from CUDA wheels (`--index-url https://download.pytorch.org/whl/cu128`). Bare `pip install torch` in a SLURM script will fail with `ModuleNotFoundError`.
- **Persist venv on scratch**: Put venvs at `~/scratch/dippy_venv` so they survive across jobs. Check `if [ ! -d "$VENV_DIR" ]` to skip recreation on repeat runs.
- **LoRA fusion must happen on CPU**: `peft`'s `fuse_lora()` calls `weight_B @ weight_A` via CUBLAS. On A100 with bf16, this can fail with `CUBLAS_STATUS_INVALID_VALUE`. Fix: call `fuse_lora()` before `.to("cuda")`, then move the fused model to GPU.
- **PyTorch 2.10+cu128 has CUBLAS bugs on A100**: Both bf16 AND float32 matmuls (`cublasGemmEx`, `cublasSgemmStridedBatched`) fail with `CUBLAS_STATUS_INVALID_VALUE` in UMT5 text encoder attention. Not a dtype issue — it's a PyTorch/CUDA compatibility bug. Fix: use PyTorch 2.6+cu124 instead, which is stable on A100.

## Gradio

- **frpc binary**: Colab share links need the frpc tunnel binary. The `_ensure_gradio_frpc_binary()` function auto-downloads it if missing.
- **`spaces` import**: Outside HF Spaces, `import spaces` fails. The app provides a no-op fallback decorator.

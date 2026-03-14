"""
Multi-backend I2V pipeline abstraction for Dippy.

Supports:
  - wan14b:     WAN 2.1 I2V 14B (original, requires A100/3090 24GB+)
  - cogvideo5b: CogVideoX-5B-I2V (runs on T4 16GB with int8 quantization)
  - ltx2b:      LTX-Video 2B (runs on T4 16GB with fp16/fp8)

Usage:
    backend = get_backend("cogvideo5b")
    backend.load(cache_dir="/content/hf_cache")
    frames = backend.generate(image=pil_img, prompt="...", ...)
"""

import os
import abc
import numpy as np
import torch
from PIL import Image

# ── Registry ──────────────────────────────────────────────────────────────────

_BACKENDS = {}


def register_backend(name):
    def decorator(cls):
        _BACKENDS[name] = cls
        return cls
    return decorator


def get_backend(name):
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[name]()


def available_backends():
    return list(_BACKENDS.keys())


# ── Base class ────────────────────────────────────────────────────────────────

class I2VBackend(abc.ABC):
    """Abstract base for image-to-video generation backends."""

    display_name: str = "Unknown"
    vram_gb: str = "?"
    description: str = ""
    default_steps: int = 4
    default_guidance: float = 1.0
    fps: int = 24
    min_frames: int = 8
    max_frames: int = 81

    def __init__(self):
        self.pipe = None
        self._loaded = False

    @abc.abstractmethod
    def load(self, cache_dir=None):
        """Load model weights to GPU. Called once."""

    @abc.abstractmethod
    def generate(self, image, prompt, negative_prompt, height, width,
                 num_frames, guidance_scale, steps, seed):
        """Generate frames from an image+prompt. Returns list[PIL.Image]."""

    def valid_num_frames(self, requested):
        """Snap requested frame count to nearest model-valid value."""
        return int(np.clip(requested, self.min_frames, self.max_frames))

    def unload(self):
        """Free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self):
        return self._loaded


# ── Frame helpers (shared) ────────────────────────────────────────────────────

def _frame_to_pil(frame):
    """Convert model frame outputs (PIL/np/tensor) into a valid RGB PIL image."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype.kind == "f":
        min_v, max_v = float(np.nanmin(arr)), float(np.nanmax(arr))
        if min_v >= -1.01 and max_v <= 1.01:
            arr = (arr + 1.0) * 127.5 if min_v < 0.0 else arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return Image.fromarray(arr[:, :, :3], mode="RGB")
    raise TypeError(f"Unsupported frame shape/dtype: {arr.shape}, {arr.dtype}")


def _frames_to_list(frames):
    if frames is None:
        return []
    if isinstance(frames, np.ndarray):
        return [frames] if frames.ndim == 3 else [frames[i] for i in range(frames.shape[0])]
    try:
        return list(frames)
    except TypeError:
        return [frames]


# ── WAN 2.1 14B Backend ──────────────────────────────────────────────────────

@register_backend("wan14b")
class Wan14BBackend(I2VBackend):
    display_name = "WAN 2.1 14B (A100/3090+)"
    vram_gb = "~26 GB bf16"
    description = "Highest quality. Requires A100 or RTX 3090+ (24GB+ VRAM)."
    default_steps = 4
    default_guidance = 1.0
    fps = 24
    min_frames = 8
    max_frames = 81

    MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    LORA_REPO_ID = "Kijai/WanVideo_comfy"
    LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

    def valid_num_frames(self, requested):
        """WAN requires (num_frames - 1) % 4 == 0."""
        valid = [
            n for n in range(self.min_frames, self.max_frames + 1)
            if (n - 1) % 4 == 0
        ]
        clamped = int(np.clip(requested, self.min_frames, self.max_frames))
        return min(valid, key=lambda n: (abs(n - clamped), n))

    def load(self, cache_dir=None):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
        from transformers import CLIPVisionModel
        from huggingface_hub import hf_hub_download

        print(f"Loading {self.display_name}...")
        image_encoder = CLIPVisionModel.from_pretrained(
            self.MODEL_ID, subfolder="image_encoder",
            torch_dtype=torch.float32, cache_dir=cache_dir,
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae",
            torch_dtype=torch.float32, cache_dir=cache_dir,
        )
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.MODEL_ID, vae=vae, image_encoder=image_encoder,
            torch_dtype=torch.bfloat16, cache_dir=cache_dir,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=8.0,
        )

        # Repair text encoder embeddings if needed
        _repair_text_encoder(self.pipe)

        # Load and fuse CausVid LoRA on CPU first to avoid CUBLAS errors,
        # then move the fused model to CUDA
        causvid_path = hf_hub_download(
            repo_id=self.LORA_REPO_ID,
            filename=self.LORA_FILENAME,
            cache_dir=cache_dir,
        )
        self.pipe.load_lora_weights(causvid_path, adapter_name="causvid_lora")
        self.pipe.set_adapters(["causvid_lora"], adapter_weights=[0.95])
        self.pipe.fuse_lora()
        self.pipe.unload_lora_weights()

        self.pipe.to("cuda")
        self._loaded = True
        print(f"{self.display_name} loaded.")

    def generate(self, image, prompt, negative_prompt, height, width,
                 num_frames, guidance_scale, steps, seed):
        with torch.inference_mode():
            output = self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).frames[0]
        frames = _frames_to_list(output)
        return [_frame_to_pil(f) for f in frames]


# ── CogVideoX 5B I2V Backend ─────────────────────────────────────────────────

@register_backend("cogvideo5b")
class CogVideo5BBackend(I2VBackend):
    display_name = "CogVideoX 5B I2V (T4-friendly)"
    vram_gb = "~5 GB int8 / ~16 GB bf16"
    description = "Good quality, runs on T4 with int8 quantization. 49 frames at 8fps."
    default_steps = 10  # 10 steps ≈ 30 steps quality (obj-003), 4x faster
    default_guidance = 6.0
    fps = 8
    min_frames = 49
    max_frames = 49  # CogVideoX generates exactly 49 frames

    MODEL_ID = "THUDM/CogVideoX-5b-I2V"

    def valid_num_frames(self, requested):
        return 49  # CogVideoX is fixed at 49 frames (~6s at 8fps)

    def load(self, cache_dir=None):
        from diffusers import CogVideoXImageToVideoPipeline

        print(f"Loading {self.display_name}...")

        # Try int8 quantization for T4 compatibility
        # Skip if DIPPY_NO_QUANTIZE is set (workaround: torchao + cpu_offload incompatible on some versions)
        if os.environ.get("DIPPY_NO_QUANTIZE"):
            use_quantization = False
            print("Quantization disabled (DIPPY_NO_QUANTIZE). Using bf16 with cpu_offload.")
        else:
            try:
                import torchao
                from torchao.quantization import quantize_, int8_weight_only
                use_quantization = True
            except ImportError:
                use_quantization = False
                print("torchao not available; loading in bf16 (needs ~16GB VRAM)")

        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )

        if use_quantization:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(self.pipe.transformer, int8_weight_only())
            print("Applied int8 quantization to transformer (~5GB VRAM)")

        # Use sequential cpu offload for tighter VRAM (e.g. RTX 3060 12GB)
        # Falls back to model-level offload if sequential isn't needed
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 14:
                self.pipe.enable_sequential_cpu_offload()
                print(f"Using sequential CPU offload (VRAM: {vram_gb:.0f}GB)")
            else:
                self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self._loaded = True
        print(f"{self.display_name} loaded.")

    def generate(self, image, prompt, negative_prompt, height, width,
                 num_frames, guidance_scale, steps, seed):
        # CogVideoX works best at 480x720, snap to nearest supported
        cog_h = max(480, (int(height) // 16) * 16)
        cog_w = max(720, (int(width) // 16) * 16)
        # Cap to avoid OOM on T4
        if cog_h * cog_w > 480 * 720:
            cog_h, cog_w = 480, 720

        with torch.inference_mode():
            output = self.pipe(
                image=image.resize((cog_w, cog_h)),
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=cog_h,
                width=cog_w,
                num_frames=49,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).frames[0]
        frames = _frames_to_list(output)
        return [_frame_to_pil(f) for f in frames]


# ── LTX-Video 2B Backend ─────────────────────────────────────────────────────

@register_backend("ltx2b")
class LTXVideo2BBackend(I2VBackend):
    display_name = "LTX-Video 2B (T4-friendly)"
    vram_gb = "~8 GB fp16"
    description = "Fast generation, lower quality. Runs easily on T4."
    default_steps = 30
    default_guidance = 3.0
    fps = 24
    min_frames = 9
    max_frames = 97

    MODEL_ID = "Lightricks/LTX-Video-0.9.7-dev"
    COND_MODEL_ID = "Lightricks/LTX-Video-0.9.7-dev"

    def valid_num_frames(self, requested):
        # LTX requires (num_frames - 1) % 8 == 0
        valid = [
            n for n in range(self.min_frames, self.max_frames + 1)
            if (n - 1) % 8 == 0
        ]
        clamped = int(np.clip(requested, self.min_frames, self.max_frames))
        return min(valid, key=lambda n: (abs(n - clamped), n))

    def load(self, cache_dir=None):
        try:
            from diffusers import LTXConditionPipeline
            self._use_cond_pipeline = True
        except ImportError:
            from diffusers import LTXImageToVideoPipeline
            self._use_cond_pipeline = False

        print(f"Loading {self.display_name}...")

        if self._use_cond_pipeline:
            self.pipe = LTXConditionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
            )
        else:
            self.pipe = LTXImageToVideoPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
            )

        # Use sequential cpu offload for tighter VRAM (e.g. RTX 3060 12GB)
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 14:
                self.pipe.enable_sequential_cpu_offload()
                print(f"Using sequential CPU offload (VRAM: {vram_gb:.0f}GB)")
            else:
                self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.enable_model_cpu_offload()
        self._loaded = True
        print(f"{self.display_name} loaded.")

    def generate(self, image, prompt, negative_prompt, height, width,
                 num_frames, guidance_scale, steps, seed):
        # LTX works at multiples of 32 for dimensions
        ltx_h = max(256, (int(height) // 32) * 32)
        ltx_w = max(256, (int(width) // 32) * 32)
        # Cap for low-VRAM GPUs
        if ltx_h * ltx_w > 512 * 768:
            ltx_h, ltx_w = 512, 768

        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=ltx_h,
            width=ltx_w,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )

        resized_img = image.resize((ltx_w, ltx_h))

        if self._use_cond_pipeline:
            from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
            condition = LTXVideoCondition(
                image=resized_img,
                frame_index=0,
            )
            gen_kwargs["conditions"] = [condition]
        else:
            gen_kwargs["image"] = resized_img

        with torch.inference_mode():
            output = self.pipe(**gen_kwargs).frames[0]
        frames = _frames_to_list(output)
        return [_frame_to_pil(f) for f in frames]


# ── Text encoder repair (WAN-specific) ───────────────────────────────────────

def _repair_text_encoder(pipe):
    """Ensure WAN text encoder embeddings are tied to shared embeddings."""
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        return
    shared = getattr(text_encoder, "shared", None)
    encoder = getattr(text_encoder, "encoder", None)
    embed_tokens = getattr(encoder, "embed_tokens", None) if encoder else None
    if shared is None or embed_tokens is None:
        return
    try:
        is_tied = (embed_tokens.weight is shared.weight) or (
            embed_tokens.weight.data_ptr() == shared.weight.data_ptr()
        )
    except Exception:
        is_tied = False
    if is_tied:
        print("Text encoder embeddings are tied (OK).")
        return
    print("Text encoder embeddings not tied. Repairing...")
    try:
        text_encoder.tie_weights()
    except Exception:
        pass
    try:
        is_tied = (embed_tokens.weight is shared.weight) or (
            embed_tokens.weight.data_ptr() == shared.weight.data_ptr()
        )
    except Exception:
        is_tied = False
    if is_tied:
        print("Repaired by tie_weights().")
        return
    embed_shape = tuple(embed_tokens.weight.shape)
    shared_shape = tuple(shared.weight.shape)
    if embed_shape != shared_shape:
        raise RuntimeError(
            f"Text encoder embedding shapes incompatible: "
            f"embed_tokens={embed_shape}, shared={shared_shape}"
        )
    try:
        with torch.no_grad():
            src = shared.weight.detach()
            if src.dtype != embed_tokens.weight.dtype or src.device != embed_tokens.weight.device:
                src = src.to(dtype=embed_tokens.weight.dtype, device=embed_tokens.weight.device)
            embed_tokens.weight.copy_(src)
        print("Applied embedding repair (shared -> encoder.embed_tokens).")
    except Exception as exc:
        print(f"Warning: could not copy shared embeddings ({exc}). Continuing.")

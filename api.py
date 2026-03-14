"""
Dippy API — REST endpoint for alinakai integration.

Provides a sentence → clip pipeline accessible via HTTP.
Designed to run alongside or instead of the Gradio UI.

Usage:
    # Start API server
    DIPPY_BACKEND=cogvideo5b python api.py

    # Or import and mount on existing ASGI app
    from api import app
"""

import base64
import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backends import get_backend, available_backends

app = FastAPI(
    title="Dippy API",
    description="Image-to-video generation for language learning animations",
    version="0.1.0",
)

# ── State ────────────────────────────────────────────────────────────────────

_backend = None
_output_dir = Path(os.environ.get("DIPPY_OUTPUT_DIR", "/tmp/dippy-output"))
_output_dir.mkdir(parents=True, exist_ok=True)


def _get_backend():
    global _backend
    if _backend is None or not _backend.is_loaded:
        backend_name = os.environ.get("DIPPY_BACKEND", "cogvideo5b")
        _backend = get_backend(backend_name)
        cache_dir = os.environ.get("HF_HUB_CACHE", "/content/hf_cache")
        _backend.load(cache_dir=cache_dir)
    return _backend


# ── Models ───────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for /generate endpoint."""
    sentence: str = Field(..., description="Sentence the character will act out")
    image_base64: Optional[str] = Field(None, description="Base64-encoded avatar image (PNG/JPEG)")
    image_url: Optional[str] = Field(None, description="URL to avatar image")
    height: int = Field(480, ge=128, le=1024)
    width: int = Field(720, ge=128, le=1024)
    duration_seconds: float = Field(2.0, ge=0.5, le=6.0)
    guidance_scale: float = Field(6.0, ge=1.0, le=20.0)
    steps: int = Field(10, ge=1, le=50)
    seed: Optional[int] = Field(None, description="Random seed (null for random)")
    include_reset: bool = Field(True, description="Generate reset pass back to neutral pose")


class GenerateResponse(BaseModel):
    """Response from /generate endpoint."""
    job_id: str
    clip_url: str
    forward_url: str
    reset_url: Optional[str] = None
    num_frames: int
    duration_seconds: float
    backend: str


class BackendInfo(BaseModel):
    name: str
    display_name: str
    vram_gb: str
    fps: int
    min_frames: int
    max_frames: int


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "backends": available_backends()}


@app.get("/backends", response_model=list[BackendInfo])
async def list_backends():
    result = []
    for name in available_backends():
        b = get_backend(name)
        result.append(BackendInfo(
            name=name,
            display_name=b.display_name,
            vram_gb=b.vram_gb,
            fps=b.fps,
            min_frames=b.min_frames,
            max_frames=b.max_frames,
        ))
    return result


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate a loopable animation clip for a sentence."""
    import random
    import numpy as np
    from PIL import Image
    from diffusers.utils import export_to_video

    backend = _get_backend()
    job_id = uuid.uuid4().hex[:12]
    job_dir = _output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Load avatar image
    if req.image_base64:
        img_bytes = base64.b64decode(req.image_base64)
        avatar = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif req.image_url:
        import urllib.request
        with urllib.request.urlopen(req.image_url) as resp:
            avatar = Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
        raise HTTPException(400, "Provide image_base64 or image_url")

    # Snap dimensions
    MOD = 16
    target_h = max(MOD, (req.height // MOD) * MOD)
    target_w = max(MOD, (req.width // MOD) * MOD)
    avatar = avatar.resize((target_w, target_h))

    # Frame count
    requested_frames = int(round(req.duration_seconds * backend.fps))
    num_frames = backend.valid_num_frames(
        int(np.clip(requested_frames, backend.min_frames, backend.max_frames))
    )
    seed = req.seed if req.seed is not None else random.randint(0, 2**31)

    # Forward pass
    forward_prompt = (
        f"The character enthusiastically acts out '{req.sentence}' with big, exaggerated "
        "body movements, arms moving expressively, full-body pantomime gestures. "
        "Smooth animation, dynamic motion."
    )
    forward_frames = backend.generate(
        image=avatar,
        prompt=forward_prompt,
        negative_prompt="static, blurry, distorted",
        height=target_h,
        width=target_w,
        num_frames=num_frames,
        guidance_scale=req.guidance_scale,
        steps=req.steps,
        seed=seed,
    )
    if not forward_frames:
        raise HTTPException(500, "Forward generation produced no frames")
    forward_frames[0] = avatar.copy()

    forward_path = str(job_dir / "forward.mp4")
    export_to_video(forward_frames, forward_path, fps=backend.fps)

    # Reset pass
    reset_path = None
    if req.include_reset:
        reset_prompt = (
            f"The same character naturally returns from acting out '{req.sentence}' "
            "back to the original neutral starting pose, arms lowering to sides. "
            "Smooth animation, gentle motion back to rest."
        )
        reset_frames = backend.generate(
            image=forward_frames[-1].copy(),
            prompt=reset_prompt,
            negative_prompt="static, blurry, distorted",
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=req.guidance_scale,
            steps=req.steps,
            seed=seed + 1,
        )
        if reset_frames:
            reset_frames[0] = forward_frames[-1].copy()
            reset_frames[-1] = avatar.copy()
            reset_path = str(job_dir / "reset.mp4")
            export_to_video(reset_frames, reset_path, fps=backend.fps)

    # Combined clip
    all_frames = list(forward_frames)
    if req.include_reset and reset_frames:
        all_frames.extend(reset_frames[1:])
    clip_path = str(job_dir / "clip.mp4")
    export_to_video(all_frames, clip_path, fps=backend.fps)

    base_url = f"/clips/{job_id}"
    return GenerateResponse(
        job_id=job_id,
        clip_url=f"{base_url}/clip.mp4",
        forward_url=f"{base_url}/forward.mp4",
        reset_url=f"{base_url}/reset.mp4" if reset_path else None,
        num_frames=len(all_frames),
        duration_seconds=len(all_frames) / backend.fps,
        backend=os.environ.get("DIPPY_BACKEND", "cogvideo5b"),
    )


@app.get("/clips/{job_id}/{filename}")
async def get_clip(job_id: str, filename: str):
    """Download a generated clip."""
    path = _output_dir / job_id / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Clip not found")
    return FileResponse(str(path), media_type="video/mp4")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DIPPY_API_PORT", "8080"))
    print(f"Starting Dippy API on port {port}")
    print(f"Backend: {os.environ.get('DIPPY_BACKEND', 'cogvideo5b')}")
    print(f"Output dir: {_output_dir}")
    uvicorn.run(app, host="0.0.0.0", port=port)

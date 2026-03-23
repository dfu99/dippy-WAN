"""
Dippy Orchestrator API — trajectory selection, segment management, video stitching.

Standalone FastAPI service designed for integration with alinakai.

Usage:
    uvicorn orchestrator.app:app --host 0.0.0.0 --port 8100
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .segment_db import SegmentDB
from .trajectory_engine import TrajectoryEngine
from .regen_scheduler import RegenScheduler
from .stitcher import stitch_segments

# ── Configuration ────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DIPPY_DB_PATH", "data/segments.db")
STITCH_OUTPUT_DIR = os.environ.get(
    "DIPPY_STITCH_DIR", "data/stitched"
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dippy Orchestrator",
    description="Trajectory alignment, segment caching, and video stitching for language learning animations",
    version="1.0.0",
)

_db: Optional[SegmentDB] = None
_engine: Optional[TrajectoryEngine] = None
_scheduler: Optional[RegenScheduler] = None


def get_db() -> SegmentDB:
    global _db
    if _db is None:
        _db = SegmentDB(DB_PATH)
    return _db


def get_engine() -> TrajectoryEngine:
    global _engine
    if _engine is None:
        _engine = TrajectoryEngine(get_db())
    return _engine


def get_scheduler() -> RegenScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = RegenScheduler()
    return _scheduler


# ── Request/Response Models ──────────────────────────────────────────────────


class TrajectoryRequest(BaseModel):
    branches: list[list[str]] = Field(
        ..., description="List of trajectory branches, each a list of sentences"
    )
    user_proficiency: float = Field(
        0.5, ge=0.0, le=1.0, description="User proficiency (0=beginner, 1=fluent)"
    )
    similarity_threshold: float = Field(
        0.75, ge=0.0, le=1.0, description="Minimum similarity for cache hit"
    )


class AddSegmentRequest(BaseModel):
    sentence: str
    video_path: str
    backend: str = "wan14b"
    avatar_hash: str = ""
    duration_s: float = 0.0
    num_frames: int = 0
    fps: int = 24
    prompt_setup: str = ""
    prompt_action: str = ""
    prompt_reset: str = ""


class StitchRequest(BaseModel):
    segment_ids: list[str] = Field(
        ..., description="Ordered list of segment IDs to stitch"
    )
    crossfade_s: float = Field(0.3, ge=0.0, le=2.0)
    fps: int = Field(24, ge=1, le=60)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    db = get_db()
    return {"status": "ok", "segments": db.count(), "db_path": DB_PATH}


@app.post("/select-trajectory")
def select_trajectory(req: TrajectoryRequest):
    """Evaluate trajectory branches and select the best one.

    Returns the chosen branch with per-sentence alignment, confidence score,
    and a prioritized regeneration plan for poorly-aligned segments.
    """
    if not req.branches:
        raise HTTPException(400, "At least one branch is required")

    engine = get_engine()
    engine.similarity_threshold = req.similarity_threshold
    selection = engine.select_trajectory(req.branches)

    # Build regeneration plan for the chosen branch
    scheduler = get_scheduler()
    regen_plan = scheduler.build_plan(
        gaps=selection.chosen.gaps,
        user_proficiency=req.user_proficiency,
        current_confidence=selection.chosen.confidence,
        total_sentences=len(selection.chosen.sentences),
    )

    return {
        "selection": selection.to_dict(),
        "regeneration_plan": regen_plan.to_dict(),
    }


@app.get("/segments")
def list_segments():
    """List all cached segments."""
    db = get_db()
    return {"segments": db.get_all_segments(), "count": db.count()}


@app.get("/segments/{segment_id}")
def get_segment(segment_id: str):
    """Get a specific segment by ID."""
    db = get_db()
    seg = db.get_segment(segment_id)
    if seg is None:
        raise HTTPException(404, f"Segment {segment_id} not found")
    return seg


@app.post("/segments")
def add_segment(req: AddSegmentRequest):
    """Add a video segment to the cache."""
    db = get_db()
    segment_id = db.add_segment(
        sentence=req.sentence,
        video_path=req.video_path,
        backend=req.backend,
        avatar_hash=req.avatar_hash,
        duration_s=req.duration_s,
        num_frames=req.num_frames,
        fps=req.fps,
        prompt_setup=req.prompt_setup,
        prompt_action=req.prompt_action,
        prompt_reset=req.prompt_reset,
    )
    return {"segment_id": segment_id, "sentence": req.sentence}


@app.delete("/segments/{segment_id}")
def delete_segment(segment_id: str):
    """Delete a segment from the cache."""
    db = get_db()
    if db.delete_segment(segment_id):
        return {"deleted": segment_id}
    raise HTTPException(404, f"Segment {segment_id} not found")


@app.post("/stitch")
def stitch(req: StitchRequest):
    """Stitch cached segments into a single trajectory video."""
    db = get_db()
    video_paths = []
    for sid in req.segment_ids:
        seg = db.get_segment(sid)
        if seg is None:
            raise HTTPException(404, f"Segment {sid} not found")
        if not Path(seg["video_path"]).exists():
            raise HTTPException(
                404, f"Video file missing for segment {sid}: {seg['video_path']}"
            )
        video_paths.append(seg["video_path"])

    output_dir = Path(STITCH_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(
        output_dir / f"trajectory_{'_'.join(req.segment_ids[:3])}.mp4"
    )

    result = stitch_segments(
        video_paths=video_paths,
        output_path=output_path,
        crossfade_duration=req.crossfade_s,
        fps=req.fps,
    )
    return result


@app.get("/stitch/{filename}")
def serve_stitched(filename: str):
    """Serve a previously stitched video file."""
    path = Path(STITCH_OUTPUT_DIR) / filename
    if not path.exists():
        raise HTTPException(404, "Stitched video not found")
    return FileResponse(str(path), media_type="video/mp4")


@app.post("/search")
def search_segments(sentence: str = Form(...), top_k: int = Form(5)):
    """Search for cached segments similar to a sentence."""
    db = get_db()
    results = db.search_nearest(sentence, top_k=top_k)
    return {"query": sentence, "results": results}

"""
Pydantic models for the Dippy Orchestrator API.

These define the contract between alinakai and the orchestrator.
All request/response types are here for a single source of truth.
"""

from pydantic import BaseModel, Field


# ── Segment Models ───────────────────────────────────────────────────────────


class SegmentInfo(BaseModel):
    """A cached video segment in the database."""
    id: str
    sentence: str
    sentence_normalized: str = ""
    video_path: str
    backend: str = "wan14b"
    avatar_hash: str = ""
    duration_s: float = 0.0
    num_frames: int = 0
    fps: int = 24
    created_at: float = 0.0


class AddSegmentRequest(BaseModel):
    """Request to add a new segment to the cache."""
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


class AddSegmentResponse(BaseModel):
    segment_id: str
    sentence: str


class SegmentListResponse(BaseModel):
    segments: list[dict]
    count: int


class SearchRequest(BaseModel):
    sentence: str
    top_k: int = Field(5, ge=1, le=100)


class SearchMatch(BaseModel):
    segment: dict
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchMatch]


# ── Trajectory Models ────────────────────────────────────────────────────────


class TrajectoryRequest(BaseModel):
    """Request to evaluate and select the best trajectory branch.

    alinakai sends multiple candidate branches. The orchestrator evaluates
    each against its segment cache and returns the best one.
    """
    branches: list[list[str]] = Field(
        ...,
        description="List of trajectory branches, each a list of sentences",
        min_length=1,
    )
    user_proficiency: float = Field(
        0.5, ge=0.0, le=1.0,
        description="User proficiency level (0=beginner, 1=fluent)",
    )
    similarity_threshold: float = Field(
        0.75, ge=0.0, le=1.0,
        description="Minimum cosine similarity for a cache hit",
    )


class SegmentMatch(BaseModel):
    """Alignment of one sentence to a cached segment."""
    sentence: str
    position: int
    segment_id: str | None
    segment_sentence: str | None
    similarity: float
    video_path: str | None
    needs_regeneration: bool


class BranchResult(BaseModel):
    """Evaluation result for one trajectory branch."""
    branch_index: int
    sentences: list[str]
    confidence: float = Field(description="Mean cosine similarity across sentences")
    coverage: float = Field(description="Fraction of sentences with acceptable cache hits")
    matches: list[SegmentMatch]
    gaps: list[SegmentMatch]


class TrajectorySelection(BaseModel):
    """Which branch was chosen and why."""
    chosen_branch: BranchResult
    all_branches: list[BranchResult]
    cache_size: int


# ── Regeneration Models ──────────────────────────────────────────────────────


class RegenItem(BaseModel):
    """A single segment queued for regeneration."""
    sentence: str
    position: int
    similarity: float
    misalignment: float
    time_to_reach_s: float
    time_to_regen_s: float
    urgency: float
    is_critical: bool = Field(description="True if user arrives before regen completes")
    priority_score: float


class RegenPlan(BaseModel):
    """Prioritized regeneration plan for missing/poor segments."""
    items: list[RegenItem]
    total_regen_time_s: float
    critical_count: int
    estimated_cache_improvement: float


class TrajectoryResponse(BaseModel):
    """Full response to a trajectory selection request."""
    selection: TrajectorySelection
    regeneration_plan: RegenPlan


# ── Stitch Models ────────────────────────────────────────────────────────────


class StitchRequest(BaseModel):
    """Request to concatenate segments into a single video."""
    segment_ids: list[str] = Field(
        ...,
        description="Ordered list of segment IDs to stitch",
        min_length=1,
    )
    crossfade_s: float = Field(0.3, ge=0.0, le=2.0)
    fps: int = Field(24, ge=1, le=60)


class StitchResult(BaseModel):
    """Result of a video stitch operation."""
    output_path: str
    duration_s: float
    segments_used: int
    segments_missing: int = 0
    crossfade: bool = False
    crossfade_duration_s: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# ── Health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    segments: int
    db_path: str

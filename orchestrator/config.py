"""
Centralized configuration for the Dippy Orchestrator.

All environment variables and defaults in one place. Import `settings`
anywhere in the orchestrator package.

Usage:
    from orchestrator.config import settings

    db = SegmentDB(settings.db_path)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Orchestrator configuration. Override via environment variables."""

    # Database
    db_path: str = ""
    stitch_dir: str = ""
    scene_cache_dir: str = ""
    clips_dir: str = ""

    # Embedding model
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Trajectory alignment
    similarity_threshold: float = 0.75

    # Regeneration scheduling
    segment_gen_time_s: float = 70.0  # ~23s × 3 segments on A100
    seconds_per_sentence: float = 30.0  # base time per sentence for TTR

    # Video stitching
    default_crossfade_s: float = 0.3
    default_fps: int = 24

    # WAN generation defaults
    wan_steps: int = 4
    wan_guidance_scale: float = 0.7
    wan_num_frames: int = 49

    # LLM scene generation
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8100

    # Frontend
    frontend_port: int = 7861

    def __post_init__(self):
        """Load overrides from environment variables."""
        self.db_path = os.environ.get("DIPPY_DB_PATH", self.db_path or "data/segments.db")
        self.stitch_dir = os.environ.get("DIPPY_STITCH_DIR", self.stitch_dir or "data/stitched")
        self.scene_cache_dir = os.environ.get("DIPPY_SCENE_CACHE", self.scene_cache_dir or "data/scene_cache")
        self.clips_dir = os.environ.get("DIPPY_CLIPS_DIR", self.clips_dir or "cache_clips")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", self.openai_api_key)
        self.similarity_threshold = float(os.environ.get("DIPPY_SIM_THRESHOLD", self.similarity_threshold))
        self.segment_gen_time_s = float(os.environ.get("DIPPY_GEN_TIME_S", self.segment_gen_time_s))
        self.api_port = int(os.environ.get("DIPPY_API_PORT", self.api_port))
        self.frontend_port = int(os.environ.get("DIPPY_FRONTEND_PORT", self.frontend_port))

        # Ensure directories exist
        for d in [self.db_path, self.stitch_dir, self.scene_cache_dir]:
            Path(d).parent.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()

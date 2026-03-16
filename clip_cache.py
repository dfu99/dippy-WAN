"""
Clip cache for Dippy — stores generated forward+reset clips by sentence hash.

When a sentence has been generated before (same sentence, backend, avatar hash),
the cached MP4 is returned instantly instead of re-running inference.

Cache structure:
    {cache_dir}/
        index.json          — maps cache keys to clip metadata
        clips/
            {key}_fwd.mp4   — forward pass clip
            {key}_rst.mp4   — reset pass clip
            {key}_full.mp4  — combined forward+reset clip

Usage:
    from clip_cache import ClipCache

    cache = ClipCache("/path/to/cache")
    hit = cache.get("He jumped", backend="wan14b", avatar_hash="abc123")
    if hit:
        # hit["full_path"], hit["fwd_path"], hit["rst_path"]
    else:
        # generate clips...
        cache.put("He jumped", backend="wan14b", avatar_hash="abc123",
                  fwd_frames=fwd_frames, rst_frames=rst_frames,
                  fps=24, metadata={...})
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from PIL import Image


def _sentence_normalize(sentence: str) -> str:
    """Normalize sentence for cache key: lowercase, strip, collapse whitespace."""
    return " ".join(sentence.lower().strip().split())


def _avatar_hash(image: Image.Image, size: int = 64) -> str:
    """Hash a PIL image by downscaling and hashing pixel data.

    Uses a small thumbnail to be invariant to minor resizing differences.
    """
    thumb = image.copy()
    thumb.thumbnail((size, size))
    thumb = thumb.convert("RGB")
    pixel_bytes = thumb.tobytes()
    return hashlib.sha256(pixel_bytes).hexdigest()[:16]


def _cache_key(sentence: str, backend: str, avatar_hash: str) -> str:
    """Generate a unique cache key from sentence + backend + avatar."""
    normalized = _sentence_normalize(sentence)
    raw = f"{normalized}|{backend}|{avatar_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class ClipCache:
    """Persistent file-based cache for generated video clips."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.clips_dir = self.cache_dir / "clips"
        self.index_path = self.cache_dir / "index.json"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def get(self, sentence: str, backend: str, avatar_hash: str) -> Optional[dict]:
        """Look up a cached clip. Returns dict with paths if hit, None if miss."""
        key = _cache_key(sentence, backend, avatar_hash)
        entry = self._index.get(key)
        if entry is None:
            return None

        # Verify files still exist
        full_path = self.clips_dir / f"{key}_full.mp4"
        if not full_path.exists():
            del self._index[key]
            self._save_index()
            return None

        entry["full_path"] = str(full_path)
        entry["fwd_path"] = str(self.clips_dir / f"{key}_fwd.mp4")
        entry["rst_path"] = str(self.clips_dir / f"{key}_rst.mp4")
        entry["cache_hit"] = True
        return entry

    def put(self, sentence: str, backend: str, avatar_hash: str,
            fwd_frames: list, rst_frames: list, fps: int,
            metadata: Optional[dict] = None) -> dict:
        """Store a generated clip in the cache. Returns paths dict."""
        from diffusers.utils import export_to_video

        key = _cache_key(sentence, backend, avatar_hash)

        # Export forward clip
        fwd_path = self.clips_dir / f"{key}_fwd.mp4"
        export_to_video(fwd_frames, str(fwd_path), fps=fps)

        # Export reset clip
        rst_path = self.clips_dir / f"{key}_rst.mp4"
        export_to_video(rst_frames, str(rst_path), fps=fps)

        # Export combined clip (skip duplicate boundary frame)
        combined = fwd_frames + rst_frames[1:]
        full_path = self.clips_dir / f"{key}_full.mp4"
        export_to_video(combined, str(full_path), fps=fps)

        # Store index entry
        entry = {
            "sentence": sentence,
            "sentence_normalized": _sentence_normalize(sentence),
            "backend": backend,
            "avatar_hash": avatar_hash,
            "fps": fps,
            "fwd_frames": len(fwd_frames),
            "rst_frames": len(rst_frames),
            "total_frames": len(combined),
            "created_at": time.time(),
        }
        if metadata:
            entry["metadata"] = metadata

        self._index[key] = entry
        self._save_index()

        return {
            "full_path": str(full_path),
            "fwd_path": str(fwd_path),
            "rst_path": str(rst_path),
            "cache_hit": False,
            **entry,
        }

    def has(self, sentence: str, backend: str, avatar_hash: str) -> bool:
        """Check if a clip is cached without returning full data."""
        return self.get(sentence, backend, avatar_hash) is not None

    def stats(self) -> dict:
        """Return cache statistics."""
        total_size = sum(
            f.stat().st_size for f in self.clips_dir.iterdir() if f.is_file()
        )
        return {
            "entries": len(self._index),
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "cache_dir": str(self.cache_dir),
        }

    def clear(self):
        """Remove all cached clips."""
        shutil.rmtree(self.clips_dir, ignore_errors=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}
        self._save_index()

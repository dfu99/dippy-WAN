"""
Batch ingest rendered clips into the orchestrator segment DB.

Scans a directory for clip_NNN_meta.json files (produced by pace_batch_render.py)
and inserts each as a segment with proper embeddings. Idempotent — skips
sentences already in the DB with high similarity.

Usage:
    python -m orchestrator.ingest --clips-dir cache_clips/
    python -m orchestrator.ingest --clips-dir /path/to/remote/cache_clips --db data/segments.db
"""

import argparse
import json
import os
from pathlib import Path

from .segment_db import SegmentDB


def ingest_clips(
    clips_dir: str,
    db_path: str = "data/segments.db",
    skip_threshold: float = 0.99,
) -> dict:
    """Ingest all clip metadata files from a directory into the segment DB.

    Args:
        clips_dir: directory containing clip_NNN_meta.json files
        db_path: path to SQLite database
        skip_threshold: skip if a segment with this similarity already exists

    Returns:
        dict with counts: added, skipped, errors
    """
    db = SegmentDB(db_path)
    clips_path = Path(clips_dir)
    meta_files = sorted(clips_path.glob("clip_*_meta.json"))

    added = 0
    skipped = 0
    errors = []

    print(f"Found {len(meta_files)} metadata files in {clips_dir}")
    print(f"DB currently has {db.count()} segments")

    for meta_file in meta_files:
        try:
            with open(meta_file) as f:
                meta = json.load(f)

            sentence = meta["sentence"]
            video_path = meta["video_path"]

            # Check if video exists
            if not Path(video_path).exists():
                # Try relative to clips_dir
                alt_path = clips_path / Path(video_path).name
                if alt_path.exists():
                    video_path = str(alt_path)
                else:
                    errors.append(f"{meta_file.name}: video not found at {video_path}")
                    continue

            # Check for duplicates
            existing = db.search_nearest(sentence, top_k=1)
            if existing and existing[0]["similarity"] >= skip_threshold:
                skipped += 1
                continue

            # Insert
            segment_id = db.add_segment(
                sentence=sentence,
                video_path=str(Path(video_path).resolve()),
                backend=meta.get("backend", "wan14b"),
                avatar_hash=meta.get("avatar_hash", "perplexity_neutral"),
                duration_s=meta.get("duration_s", 0.0),
                num_frames=meta.get("num_frames", 0),
                fps=meta.get("fps", 24),
                prompt_setup=meta.get("prompt_setup", ""),
                prompt_action=meta.get("prompt_action", ""),
                prompt_reset=meta.get("prompt_reset", ""),
                metadata={
                    "source": "batch_render",
                    "inference_time_s": meta.get("total_inference_s", 0),
                    "seed": meta.get("seed", 0),
                    "guidance_scale": meta.get("guidance_scale", 0.7),
                },
            )
            added += 1
            print(f"  + [{segment_id}] \"{sentence}\"")

        except Exception as e:
            errors.append(f"{meta_file.name}: {e}")

    db.close()
    result = {
        "added": added,
        "skipped": skipped,
        "errors": len(errors),
        "error_details": errors,
        "total_in_db": SegmentDB(db_path).count(),
    }
    print(f"\nIngested: {added} added, {skipped} skipped, {len(errors)} errors")
    print(f"DB now has {result['total_in_db']} segments")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest rendered clips into segment DB")
    p.add_argument("--clips-dir", default="cache_clips", help="Directory with clip_NNN_meta.json files")
    p.add_argument("--db", default="data/segments.db", help="Segment database path")
    p.add_argument("--skip-threshold", type=float, default=0.99, help="Skip if similarity above this")
    args = p.parse_args()
    ingest_clips(args.clips_dir, args.db, args.skip_threshold)

"""
Seed the segment database with existing demo clips.

Scans results/ for known demo video files and adds them as segments.
"""

import os
from pathlib import Path

from .segment_db import SegmentDB


# Sentences from our validated demo runs
DEMO_SENTENCES = [
    "He jumped",
    "She laughed",
    "She laughed at him",
    "He ran",
    "He ran because it started raining",
]

# Additional vocabulary sentences for broader cache coverage
EXTRA_SENTENCES = [
    "She waved",
    "He pointed at the sky",
    "They danced together",
    "She cried",
    "He ate the food",
    "She opened the door",
    "He read a book",
    "She sang a song",
    "He climbed the stairs",
    "They walked to school",
]


def seed_demo_clips(
    db_path: str = "data/segments.db",
    results_dir: str = "results",
) -> int:
    """Seed the database with demo clips from results/.

    Looks for demo_trajectory_wan14b.mp4 and per-sentence clips.
    Returns number of segments added.
    """
    db = SegmentDB(db_path)
    results = Path(results_dir)
    count = 0

    # Add per-sentence clips if they exist (from avatar tests)
    for i, sentence in enumerate(DEMO_SENTENCES, 1):
        # Check for individual sentence clips
        clip_path = results / f"avatar_wan14b_Perplexity_clip.mp4"
        if not clip_path.exists():
            clip_path = results / f"demo_trajectory_wan14b.mp4"

        if clip_path.exists():
            # Check if already in DB
            existing = db.search_nearest(sentence, top_k=1)
            if existing and existing[0]["similarity"] > 0.99:
                continue

            db.add_segment(
                sentence=sentence,
                video_path=str(clip_path.resolve()),
                backend="wan14b",
                avatar_hash="perplexity_neutral",
                duration_s=4.0,  # ~49 frames at 24fps per segment
                num_frames=49 * 3,  # 3 segments
                fps=24,
            )
            count += 1

    # Add extra vocabulary sentences with placeholder paths
    # (these create embeddings for alignment even without videos yet)
    for sentence in EXTRA_SENTENCES:
        existing = db.search_nearest(sentence, top_k=1)
        if existing and existing[0]["similarity"] > 0.99:
            continue
        # Mark as needing generation with a placeholder path
        db.add_segment(
            sentence=sentence,
            video_path="PENDING_GENERATION",
            backend="wan14b",
            avatar_hash="perplexity_neutral",
        )
        count += 1

    db.close()
    return count


if __name__ == "__main__":
    count = seed_demo_clips()
    print(f"Seeded {count} segments.")

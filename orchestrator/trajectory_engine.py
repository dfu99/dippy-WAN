"""
Trajectory Engine — selects the best trajectory branch and aligns it to cached segments.

Given multiple candidate trajectory branches (each a list of sentences),
evaluates each against the segment cache and returns the branch with the
highest aggregate alignment score, along with per-sentence match details.
"""

from dataclasses import dataclass, field

import numpy as np

from .segment_db import SegmentDB, encode_sentences


@dataclass
class SegmentMatch:
    """A single sentence's alignment to a cached segment."""
    sentence: str
    position: int
    segment_id: str | None
    segment_sentence: str | None
    similarity: float
    video_path: str | None
    needs_regeneration: bool

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "position": self.position,
            "segment_id": self.segment_id,
            "segment_sentence": self.segment_sentence,
            "similarity": round(self.similarity, 4),
            "video_path": self.video_path,
            "needs_regeneration": self.needs_regeneration,
        }


@dataclass
class BranchResult:
    """Evaluation result for one trajectory branch."""
    branch_index: int
    sentences: list[str]
    matches: list[SegmentMatch]
    confidence: float
    coverage: float  # fraction of sentences with acceptable alignment
    gaps: list[SegmentMatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "branch_index": self.branch_index,
            "sentences": self.sentences,
            "confidence": round(self.confidence, 4),
            "coverage": round(self.coverage, 4),
            "matches": [m.to_dict() for m in self.matches],
            "gaps": [g.to_dict() for g in self.gaps],
        }


@dataclass
class TrajectorySelection:
    """Result of trajectory branch selection."""
    chosen: BranchResult
    all_branches: list[BranchResult]
    cache_size: int

    def to_dict(self) -> dict:
        return {
            "chosen_branch": self.chosen.to_dict(),
            "all_branches": [b.to_dict() for b in self.all_branches],
            "cache_size": self.cache_size,
        }


class TrajectoryEngine:
    """Evaluates and selects trajectory branches against the segment cache."""

    def __init__(self, db: SegmentDB, similarity_threshold: float = 0.75):
        self.db = db
        self.similarity_threshold = similarity_threshold

    def evaluate_branch(
        self, sentences: list[str], branch_index: int = 0
    ) -> BranchResult:
        """Evaluate a single branch against the segment cache."""
        if self.db.count() == 0:
            matches = [
                SegmentMatch(
                    sentence=s,
                    position=i,
                    segment_id=None,
                    segment_sentence=None,
                    similarity=0.0,
                    video_path=None,
                    needs_regeneration=True,
                )
                for i, s in enumerate(sentences)
            ]
            return BranchResult(
                branch_index=branch_index,
                sentences=sentences,
                matches=matches,
                confidence=0.0,
                coverage=0.0,
                gaps=matches.copy(),
            )

        search_results = self.db.search_batch(sentences, top_k=1)
        matches = []
        gaps = []

        for i, (sentence, results) in enumerate(zip(sentences, search_results)):
            if results and results[0]["similarity"] > 0:
                best = results[0]
                seg = best["segment"]
                sim = best["similarity"]
                match = SegmentMatch(
                    sentence=sentence,
                    position=i,
                    segment_id=seg["id"],
                    segment_sentence=seg["sentence"],
                    similarity=sim,
                    video_path=seg["video_path"],
                    needs_regeneration=sim < self.similarity_threshold,
                )
            else:
                match = SegmentMatch(
                    sentence=sentence,
                    position=i,
                    segment_id=None,
                    segment_sentence=None,
                    similarity=0.0,
                    video_path=None,
                    needs_regeneration=True,
                )
            matches.append(match)
            if match.needs_regeneration:
                gaps.append(match)

        similarities = [m.similarity for m in matches]
        confidence = float(np.mean(similarities)) if similarities else 0.0
        coverage = sum(1 for m in matches if not m.needs_regeneration) / max(
            len(matches), 1
        )

        return BranchResult(
            branch_index=branch_index,
            sentences=sentences,
            matches=matches,
            confidence=confidence,
            coverage=coverage,
            gaps=gaps,
        )

    def select_trajectory(
        self, branches: list[list[str]]
    ) -> TrajectorySelection:
        """Evaluate all branches and select the best one.

        Selection criteria (in order):
        1. Highest coverage (fraction of usable cached segments)
        2. Highest confidence (mean cosine similarity)
        """
        results = [
            self.evaluate_branch(branch, i) for i, branch in enumerate(branches)
        ]
        # Sort by (coverage desc, confidence desc)
        results_sorted = sorted(
            results, key=lambda r: (r.coverage, r.confidence), reverse=True
        )
        return TrajectorySelection(
            chosen=results_sorted[0],
            all_branches=results,
            cache_size=self.db.count(),
        )

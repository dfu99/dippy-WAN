"""
Regeneration Scheduler — prioritizes which segments to regenerate.

Given a list of poorly-aligned segments (gaps), estimates the urgency of
regeneration based on:
  - Misalignment severity (1 - cosine_similarity)
  - Time to regenerate (TTR) based on known inference speed
  - Time until user reaches the segment (from proficiency + position)

Segments where the user would arrive before regeneration completes
are flagged as critical.
"""

from dataclasses import dataclass

from .trajectory_engine import SegmentMatch


# Average time to generate one 3-segment clip on A100 (setup+action+reset)
DEFAULT_SEGMENT_GEN_TIME_S = 70.0  # ~23s per segment × 3 segments


@dataclass
class RegenItem:
    """A segment queued for regeneration with priority metadata."""
    sentence: str
    position: int
    similarity: float
    misalignment: float
    time_to_reach_s: float
    time_to_regen_s: float
    urgency: float  # higher = more urgent
    is_critical: bool  # user arrives before regen completes
    priority_score: float

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "position": self.position,
            "similarity": round(self.similarity, 4),
            "misalignment": round(self.misalignment, 4),
            "time_to_reach_s": round(self.time_to_reach_s, 1),
            "time_to_regen_s": round(self.time_to_regen_s, 1),
            "urgency": round(self.urgency, 4),
            "is_critical": self.is_critical,
            "priority_score": round(self.priority_score, 4),
        }


@dataclass
class RegenPlan:
    """Complete regeneration plan with ordered items."""
    items: list[RegenItem]
    total_regen_time_s: float
    critical_count: int
    estimated_cache_improvement: float  # projected confidence after regen

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "total_regen_time_s": round(self.total_regen_time_s, 1),
            "critical_count": self.critical_count,
            "estimated_cache_improvement": round(
                self.estimated_cache_improvement, 4
            ),
        }


class RegenScheduler:
    """Schedules segment regeneration by priority."""

    def __init__(
        self,
        segment_gen_time_s: float = DEFAULT_SEGMENT_GEN_TIME_S,
    ):
        self.segment_gen_time_s = segment_gen_time_s

    def estimate_time_to_reach(
        self,
        position: int,
        user_proficiency: float,
        seconds_per_sentence: float = 30.0,
    ) -> float:
        """Estimate seconds until user reaches a given sentence position.

        Args:
            position: 0-indexed position in the trajectory
            user_proficiency: 0.0 (beginner) to 1.0 (fluent)
                Higher proficiency → faster progression → less time
            seconds_per_sentence: base time per sentence at proficiency=0
        """
        # Beginners take longer per sentence, fluent users are faster
        # Scale: proficiency 0.0 → 1.0x speed, proficiency 1.0 → 3.0x speed
        speed_multiplier = 1.0 + 2.0 * user_proficiency
        time_per_sentence = seconds_per_sentence / speed_multiplier
        return position * time_per_sentence

    def build_plan(
        self,
        gaps: list[SegmentMatch],
        user_proficiency: float = 0.5,
        current_confidence: float = 0.0,
        total_sentences: int = 0,
    ) -> RegenPlan:
        """Build a prioritized regeneration plan from gap analysis.

        Args:
            gaps: segments that need regeneration (from TrajectoryEngine)
            user_proficiency: 0.0-1.0 user skill level
            current_confidence: current trajectory confidence score
            total_sentences: total number of sentences in trajectory
        """
        items = []
        cumulative_regen_time = 0.0

        for gap in gaps:
            misalignment = 1.0 - gap.similarity
            time_to_reach = self.estimate_time_to_reach(
                gap.position, user_proficiency
            )
            # Account for queue: this segment won't start regenerating until
            # all higher-priority items ahead of it complete
            effective_regen_time = cumulative_regen_time + self.segment_gen_time_s
            is_critical = time_to_reach < effective_regen_time

            # Urgency = misalignment × time pressure
            # Higher misalignment → more urgent
            # Earlier position → more urgent (user hits it sooner)
            time_pressure = max(0.0, 1.0 - (time_to_reach / max(effective_regen_time, 1.0)))
            urgency = misalignment * (1.0 + time_pressure)

            # Priority combines urgency with position (earlier = higher)
            position_weight = 1.0 / (1.0 + gap.position)
            priority_score = urgency * (1.0 + position_weight)

            items.append(
                RegenItem(
                    sentence=gap.sentence,
                    position=gap.position,
                    similarity=gap.similarity,
                    misalignment=misalignment,
                    time_to_reach_s=time_to_reach,
                    time_to_regen_s=self.segment_gen_time_s,
                    urgency=urgency,
                    is_critical=is_critical,
                    priority_score=priority_score,
                )
            )
            cumulative_regen_time += self.segment_gen_time_s

        # Sort by priority (highest first)
        items.sort(key=lambda x: x.priority_score, reverse=True)

        # Estimate improvement: if we regenerate all gaps, confidence goes up
        if total_sentences > 0 and gaps:
            gap_count = len(gaps)
            improvement = (gap_count / total_sentences) * (
                1.0 - current_confidence
            )
        else:
            improvement = 0.0

        return RegenPlan(
            items=items,
            total_regen_time_s=len(items) * self.segment_gen_time_s,
            critical_count=sum(1 for i in items if i.is_critical),
            estimated_cache_improvement=improvement,
        )

"""
Video Stitcher — concatenates cached video segments into a continuous trajectory.

Takes an ordered list of video segment paths and produces a single MP4
with optional crossfade transitions at segment boundaries.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _get_video_duration(path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def stitch_segments(
    video_paths: list[str],
    output_path: str,
    crossfade_duration: float = 0.3,
    fps: int = 24,
) -> dict:
    """Concatenate video segments into a single trajectory video.

    Args:
        video_paths: ordered list of MP4 file paths
        output_path: path for the output stitched video
        crossfade_duration: seconds of crossfade between segments (0 = hard cut)
        fps: output frame rate

    Returns:
        dict with output path, duration, segment count, and any warnings
    """
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found. Install with: apt install ffmpeg")

    valid_paths = [p for p in video_paths if Path(p).exists()]
    missing = [p for p in video_paths if not Path(p).exists()]
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} segments missing: {missing}")

    if not valid_paths:
        raise ValueError("No valid video segments to stitch")

    if len(valid_paths) == 1:
        # Single segment — just copy
        subprocess.run(
            ["cp", valid_paths[0], output_path], check=True
        )
        duration = _get_video_duration(output_path)
        return {
            "output_path": output_path,
            "duration_s": duration,
            "segments_used": 1,
            "segments_missing": len(missing),
            "crossfade": False,
            "warnings": warnings,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if crossfade_duration > 0 and len(valid_paths) > 1:
        return _stitch_with_crossfade(
            valid_paths, output_path, crossfade_duration, fps, warnings
        )
    else:
        return _stitch_concat(valid_paths, output_path, fps, warnings)


def _stitch_concat(
    paths: list[str], output_path: str, fps: int, warnings: list[str]
) -> dict:
    """Simple concatenation without crossfade."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        for p in paths:
            f.write(f"file '{p}'\n")
        list_path = f.name

    try:
        # Use stream copy first (fast), fall back to re-encode if it fails
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            # Re-encode if stream copy fails (mixed codecs/formats)
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_path,
                    "-vf", f"fps={fps},format=yuv420p",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-movflags", "+faststart",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
    finally:
        Path(list_path).unlink(missing_ok=True)

    duration = _get_video_duration(output_path)
    return {
        "output_path": output_path,
        "duration_s": duration,
        "segments_used": len(paths),
        "segments_missing": 0,
        "crossfade": False,
        "warnings": warnings,
    }


def _stitch_with_crossfade(
    paths: list[str],
    output_path: str,
    crossfade_s: float,
    fps: int,
    warnings: list[str],
) -> dict:
    """Concatenation with crossfade transitions between segments.

    Uses ffmpeg's xfade filter for smooth blending at boundaries.
    """
    if len(paths) == 2:
        # Simple case: one crossfade
        dur0 = _get_video_duration(paths[0])
        offset = max(0.0, dur0 - crossfade_s)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", paths[0],
                "-i", paths[1],
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_s}:offset={offset},format=yuv420p[v]",
                "-map", "[v]",
                "-c:v", "libx264",
                "-r", str(fps),
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            check=True,
        )
    else:
        # Multi-segment: chain xfade filters
        inputs = []
        for p in paths:
            inputs.extend(["-i", p])

        durations = [_get_video_duration(p) for p in paths]
        filter_parts = []
        cumulative_duration = durations[0]

        # First xfade
        offset = max(0.0, cumulative_duration - crossfade_s)
        filter_parts.append(
            f"[0:v][1:v]xfade=transition=fade:duration={crossfade_s}:offset={offset}[v1]"
        )
        cumulative_duration = offset + durations[1]

        # Subsequent xfades
        for i in range(2, len(paths)):
            prev_label = f"v{i-1}"
            curr_label = f"v{i}"
            offset = max(0.0, cumulative_duration - crossfade_s)
            filter_parts.append(
                f"[{prev_label}][{i}:v]xfade=transition=fade:duration={crossfade_s}:offset={offset}[{curr_label}]"
            )
            cumulative_duration = offset + durations[i]

        last_label = f"v{len(paths)-1}"
        filter_parts.append(f"[{last_label}]format=yuv420p[vout]")
        filter_complex = ";".join(filter_parts)

        subprocess.run(
            [
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-c:v", "libx264",
                "-r", str(fps),
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            check=True,
        )

    duration = _get_video_duration(output_path)
    return {
        "output_path": output_path,
        "duration_s": duration,
        "segments_used": len(paths),
        "segments_missing": 0,
        "crossfade": True,
        "crossfade_duration_s": crossfade_s,
        "warnings": warnings,
    }

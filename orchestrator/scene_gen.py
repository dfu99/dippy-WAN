"""
LLM Scene Description Generator — auto-generates setup/action prompts for any sentence.

Given an arbitrary sentence, calls GPT-4o-mini to produce:
  1. A setup description (costume, props, background staging)
  2. An action description (performing the sentence)

Results are cached to disk to avoid repeated API calls for the same sentence.

Usage:
    from orchestrator.scene_gen import generate_scene, SceneDescription

    scene = generate_scene("He jumped")
    print(scene.setup)   # "The bald robot puts on a sporty headband..."
    print(scene.action)  # "The robot springs upward with a huge leap..."

Requires OPENAI_API_KEY environment variable.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


AVATAR_CONTEXT = (
    "A simple vector-art anime-style robotic character, currently bald with no hair. "
    "Versatile actor that can morph, don wigs, hair accessories, hats, costumes, "
    "and sprout props to embody actions. "
    "The background can fade in, swipe, or slide like stage props — "
    "it must transition smoothly, never appear or disappear instantly."
)

SYSTEM_PROMPT = f"""You are a creative director for animated charades clips. The avatar is:

{AVATAR_CONTEXT}

Given a sentence that the avatar must act out, generate two descriptions:

1. **setup**: How the character prepares — what costume/disguise it puts on, what props appear, how the background transitions in. Focus on the transformation FROM the bald neutral state TO the costumed character. Describe backgrounds sliding, fading, or swiping in like stage props.

2. **action**: How the character performs the sentence at peak energy — the climax of the action with maximum expressiveness. The character is already in costume from the setup phase.

Rules:
- Keep each description to 2-3 sentences
- The character starts bald — use wigs, hats, headbands, accessories as disguises
- Backgrounds must transition smoothly (fade, slide, swipe) — never instant
- Be vivid and specific about body movements, expressions, and props
- The action should be the dramatic peak, not a repeat of setup

Respond in JSON format:
{{"setup": "...", "action": "..."}}"""


@dataclass
class SceneDescription:
    """Generated setup and action descriptions for a sentence."""
    sentence: str
    setup: str
    action: str
    source: str  # "llm", "cache", or "fallback"


# Simple disk cache — use config if available, else env var fallback
try:
    from .config import settings as _cfg
    _CACHE_DIR = Path(_cfg.scene_cache_dir)
except ImportError:
    _CACHE_DIR = Path(os.environ.get("DIPPY_SCENE_CACHE", "data/scene_cache"))


def _cache_key(sentence: str) -> str:
    normalized = " ".join(sentence.lower().strip().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _load_from_cache(sentence: str) -> Optional[SceneDescription]:
    key = _cache_key(sentence)
    cache_file = _CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            return SceneDescription(
                sentence=sentence,
                setup=data["setup"],
                action=data["action"],
                source="cache",
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_to_cache(sentence: str, setup: str, action: str):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(sentence)
    cache_file = _CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump({"sentence": sentence, "setup": setup, "action": action}, f, indent=2)


def _fallback_description(sentence: str) -> SceneDescription:
    """Generate a generic description without LLM when API is unavailable."""
    return SceneDescription(
        sentence=sentence,
        setup=(
            f"The bald robot prepares to act out '{sentence}' — putting on a relevant "
            "costume or disguise, sprouting props. The background fades in like a stage set. "
            "The robot shifts into a ready stance with anticipation."
        ),
        action=(
            f"The costumed character fully performs '{sentence}' with maximum energy — "
            "exaggerated full-body movement, peak of the action, dramatic performance."
        ),
        source="fallback",
    )


def generate_scene(
    sentence: str,
    use_cache: bool = True,
    model: str = "gpt-4o-mini",
) -> SceneDescription:
    """Generate setup and action descriptions for a sentence.

    Checks cache first, then calls GPT-4o-mini, falls back to generic if no API key.

    Args:
        sentence: the sentence to generate descriptions for
        use_cache: whether to check/save disk cache
        model: OpenAI model to use

    Returns:
        SceneDescription with setup and action text
    """
    # Check cache
    if use_cache:
        cached = _load_from_cache(sentence)
        if cached:
            return cached

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        result = _fallback_description(sentence)
        if use_cache:
            _save_to_cache(sentence, result.setup, result.action)
        return result

    # Call OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sentence: \"{sentence}\""},
            ],
            temperature=0.7,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        result = SceneDescription(
            sentence=sentence,
            setup=data["setup"],
            action=data["action"],
            source="llm",
        )

        if use_cache:
            _save_to_cache(sentence, result.setup, result.action)

        return result

    except Exception as e:
        print(f"LLM scene generation failed for '{sentence}': {e}")
        result = _fallback_description(sentence)
        if use_cache:
            _save_to_cache(sentence, result.setup, result.action)
        return result


def generate_scenes_batch(
    sentences: list[str],
    use_cache: bool = True,
    model: str = "gpt-4o-mini",
) -> list[SceneDescription]:
    """Generate descriptions for multiple sentences. Cache-aware."""
    return [generate_scene(s, use_cache=use_cache, model=model) for s in sentences]


def build_setup_prompt(sentence: str) -> str:
    """Build a full setup prompt with avatar context + LLM-generated scene."""
    scene = generate_scene(sentence)
    return f"{AVATAR_CONTEXT} Setting up for '{sentence}': {scene.setup} Smooth transformation."


def build_action_prompt(sentence: str) -> str:
    """Build a full action prompt with avatar context + LLM-generated scene."""
    scene = generate_scene(sentence)
    return f"{AVATAR_CONTEXT} The costumed character performs '{sentence}': {scene.action} Maximum expressiveness."


def build_reset_prompt(sentence: str) -> str:
    """Build a reset prompt (same for all sentences, no LLM needed)."""
    return (
        f"{AVATAR_CONTEXT} "
        f"The character finishes '{sentence}' and smoothly returns to neutral. "
        "Wig and accessories dissolve away, props retract into the body, "
        "costume fades off. Background slides out or fades to empty. "
        "The bald robot returns to its default relaxed standing pose."
    )

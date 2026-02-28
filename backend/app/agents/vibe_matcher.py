"""
Node 3 — The VIBE MATCHER (Qualitative Analysis)
Aesthetic reasoning engine using Gemini 1.5 Pro multimodal.
Scores each venue's vibe against the user's desired aesthetic.
"""

import asyncio
import json
import logging

from app.models.state import PathfinderState
from app.services.gemini import generate_content

logger = logging.getLogger(__name__)

_VIBE_PROMPT = """You are a venue aesthetic analyst. Analyze this venue and score its "vibe".

Venue: {name}
Address: {address}
Category: {category}

User's desired vibe: {vibe_preference}

Based on the venue photos and information, respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "score": <float 0.0 to 1.0, how well this venue matches the desired vibe>,
  "style": "<one-word style label, e.g. cozy, minimalist, industrial, vibrant>",
  "descriptors": ["<descriptor 1>", "<descriptor 2>", "<descriptor 3>"],
  "confidence": <float 0.0 to 1.0, how confident you are in this assessment>
}}
"""

_NEUTRAL_VIBE = "a welcoming, enjoyable atmosphere suitable for groups"


async def _score_venue(venue: dict, vibe_preference: str) -> dict | None:
    """Score a single venue's vibe using Gemini 1.5 Pro multimodal."""
    prompt = _VIBE_PROMPT.format(
        name=venue.get("name", "Unknown"),
        address=venue.get("address", ""),
        category=venue.get("category", ""),
        vibe_preference=vibe_preference,
    )

    photos = venue.get("photos", [])

    try:
        raw = await generate_content(
            prompt=prompt,
            model="gemini-2.5-flash",
            image_urls=photos if photos else None,
        )
        if not raw:
            return None

        # Strip markdown fences if Gemini wraps the JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        return json.loads(cleaned)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Vibe scoring failed for %s: %s", venue.get("name"), exc)
        return None


def vibe_matcher_node(state: PathfinderState) -> PathfinderState:
    """
    Score each candidate venue on subjective vibe / aesthetic match.

    Steps
    -----
    1. Get vibe preference from parsed_intent (or use neutral default).
    2. For each candidate, call Gemini 1.5 Pro with photos + prompt.
    3. Parse JSON response into vibe scores.
    4. Write vibe_scores dict to state.
    """
    intent = state.get("parsed_intent", {})
    vibe_pref = intent.get("vibe") or _NEUTRAL_VIBE
    candidates = state.get("candidate_venues", [])

    if not candidates:
        logger.info("Vibe Matcher: no candidates to score")
        return {"vibe_scores": {}}

    # Score all venues concurrently
    async def _score_all():
        return await asyncio.gather(*[_score_venue(v, vibe_pref) for v in candidates])

    try:
        results = asyncio.run(_score_all())
    except Exception as exc:
        logger.error("Vibe Matcher failed: %s", exc)
        results = [None] * len(candidates)

    # Build vibe_scores dict keyed by venue_id
    vibe_scores = {}
    for venue, result in zip(candidates, results):
        vid = venue.get("venue_id", "")
        if result:
            vibe_scores[vid] = result
        else:
            # Graceful fallback — don't crash, just mark as unscored
            vibe_scores[vid] = {
                "score": None,
                "style": "unknown",
                "descriptors": [],
                "confidence": 0.0,
            }

    logger.info("Vibe Matcher scored %d/%d venues",
                sum(1 for v in vibe_scores.values() if v.get("score") is not None),
                len(candidates))

    return {"vibe_scores": vibe_scores}

"""
Node 6 — The CRITIC (Adversarial)
Actively tries to break the plan with real-world risk checks.
Model: Gemini (Adversarial Reasoning)
Tools: OpenWeather API, PredictHQ
"""

import json
import logging
import asyncio

from app.models.state import PathfinderState
from app.services.openweather import get_weather
from app.services.predicthq import get_events
from app.services.gemini import generate_content

logger = logging.getLogger(__name__)


async def critic_node(state: PathfinderState) -> PathfinderState:
    """
    Cross-reference top venues with real-world risks.

    Steps
    -----
    1. Fetch weather forecast via OpenWeather API.
    2. Fetch upcoming events / road closures via PredictHQ.
    3. Identify dealbreakers (rain-prone parks, marathon routes, …).
    4. If critical: set fast_fail = True → triggers Commander retry.
    5. Return updated state with risk_flags, fast_fail, fast_fail_reason.
    """
    candidates = state.get("candidate_venues", [])
    if not candidates:
        logger.info("Critic: no candidates to evaluate.")
        return {
            "risk_flags": {},
            "veto": False,
            "veto_reason": None,
            "fast_fail": False,
            "fast_fail_reason": None,
        }

    # Evaluate the top 3 candidates (to save time/tokens)
    top_candidates = candidates[:3]

    async def _analyze_venue(venue):
        lat = venue.get("lat")
        lng = venue.get("lng")
        venue_id = venue.get("venue_id", venue.get("name", "unknown"))

        # Parallel fetch of weather + events with a short timeout each
        weather, events = await asyncio.gather(
            get_weather(lat, lng),
            get_events(lat, lng),
        )

        prompt = f"""
        You are the PATHFINDER Critic Agent. Your job is to find reasons why this plan is TERRIBLE.
        Look for dealbreakers that would ruin the experience.

        Context:
        User Intent: {json.dumps(state.get("parsed_intent", {}))}
        Venue: {venue.get("name")} ({venue.get("category")})
        Weather Profile: {json.dumps(weather)}
        Upcoming Events Nearby: {json.dumps(events)}

        Evaluate the venue against Fast-Fail Conditions:
        - Condition A: Are there fewer than 3 viable venues after risk filtering? (Assume no for a
          single venue unless the user intent is extremely strict and this venue wildly misses it
          alongside weather/event risks).
        - Condition B: Is there a Top Candidate Veto? (e.g., outdoor activity + heavy rain,
          traffic jam due to a marathon blocking access).

        If either condition is met, trigger a fast-fail.

        Output exact JSON:
        {{
            "risks": [
                {{"type": "weather", "severity": "high/medium/low", "detail": "explanation"}}
            ],
            "fast_fail": true/false,
            "fast_fail_reason": "if true, short reason for early termination"
        }}
        """

        try:
            resp = await generate_content(prompt)
            if not resp:
                return venue_id, {"risks": [], "fast_fail": False, "fast_fail_reason": None}
            cleaned = resp.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            analysis = json.loads(cleaned.strip())
            return venue_id, analysis
        except Exception as exc:
            logger.error("Critic Gemini call failed for %s: %s", venue_id, exc)
            return venue_id, {"risks": [], "fast_fail": False, "fast_fail_reason": None}

    # Run all venues concurrently — 25s timeout per venue so a slow Gemini/weather
    # call never freezes the whole pipeline.
    raw = await asyncio.gather(
        *[asyncio.wait_for(_analyze_venue(v), timeout=25.0) for v in top_candidates],
        return_exceptions=True,
    )

    risk_flags: dict = {}
    overall_fast_fail = False
    fast_fail_reason = None

    for i, res in enumerate(raw):
        venue = top_candidates[i]
        vid = venue.get("venue_id", venue.get("name", "unknown"))

        if isinstance(res, Exception):
            logger.warning("Critic skipping %s due to error: %s", vid, res)
            risk_flags[vid] = []
            continue

        venue_id, analysis = res
        risk_flags[venue_id] = analysis.get("risks", [])

        # Only fast-fail if the #1 candidate triggers it
        if (
            analysis.get("fast_fail")
            and venue_id == top_candidates[0].get("venue_id", top_candidates[0].get("name"))
        ):
            overall_fast_fail = True
            fast_fail_reason = analysis.get("fast_fail_reason")

    return {
        "risk_flags": risk_flags,
        "fast_fail": overall_fast_fail,
        "fast_fail_reason": fast_fail_reason,
        "veto": overall_fast_fail,
        "veto_reason": fast_fail_reason,
    }

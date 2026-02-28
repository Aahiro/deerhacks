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


def critic_node(state: PathfinderState) -> PathfinderState:
    """
    Cross-reference top venues with real-world risks.

    Steps
    -----
    1. Fetch weather forecast via OpenWeather API.
    2. Fetch upcoming events / road closures via PredictHQ.
    3. Identify dealbreakers (rain-prone parks, marathon routes, …).
    4. If critical: set veto = True → triggers Commander retry.
    5. Return updated state with risk_flags, veto, veto_reason.
    """
    candidates = state.get("candidate_venues", [])
    if not candidates:
        logger.info("Critic: no candidates to evaluate.")
        state["risk_flags"] = {}
        state["veto"] = False
        state["veto_reason"] = None
        return state

    # Evaluate the top 3 candidates (to save time/tokens)
    top_candidates = candidates[:3]
    risk_flags = {}

    async def _analyze_venue(venue):
        lat = venue.get("lat")
        lng = venue.get("lng")
        venue_id = venue.get("venue_id", venue.get("name", "unknown"))
        
        # Parallel fetch
        weather, events = await asyncio.gather(
            get_weather(lat, lng),
            get_events(lat, lng)
        )
        
        # Prepare context for Gemini
        context = {
            "name": venue.get("name"),
            "category": venue.get("category"),
            "intent": state.get("parsed_intent", {}),
            "weather": weather,
            "events": events
        }
        
        prompt = f"""
        You are the PATHFINDER Critic Agent. Your job is to find reasons why this plan is TERRIBLE.
        Look for dealbreakers that would ruin the experience.
        
        Context:
        User Intent: {json.dumps(context['intent'])}
        Venue: {context['name']} ({context['category']})
        Weather Profile: {json.dumps(context['weather'])}
        Upcoming Events Nearby: {json.dumps(context['events'])}
        
        If there's a serious risk (e.g. outdoor activity + heavy rain, traffic jam due to a marathon), output a veto.
        
        Output exact JSON:
        {{
            "risks": [
                {{"type": "weather", "severity": "high/medium/low", "detail": "explanation"}}
            ],
            "veto": true/false,
            "veto_reason": "if true, short reason"
        }}
        """
        
        try:
            resp = await generate_content(prompt)
            if resp.startswith("```json"):
                resp = resp[7:]
            if resp.endswith("```"):
                resp = resp[:-3]
            
            analysis = json.loads(resp.strip())
            return venue_id, analysis
        except Exception as e:
            logger.error(f"Critic Gemini call failed for {venue_id}: {e}")
            return venue_id, {"risks": [], "veto": False, "veto_reason": None}

    async def _run_all():
        return await asyncio.gather(*[_analyze_venue(v) for v in top_candidates])

    # Run analysis for all top candidates
    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        # If loop is already running in environment
        import nest_asyncio
        nest_asyncio.apply()
        results = asyncio.run(_run_all())
    
    overall_veto = False
    veto_reason = None
    
    for venue_id, analysis in results:
        risk_flags[venue_id] = analysis.get("risks", [])
        # If any of the top 3 has a critical veto for the whole plan, we could trigger graph retry.
        # But realistically we only veto if ALL of them are terrible, otherwise we just flag.
        # For simplicity, if the #1 candidate gets a veto, we flag the system.
        if analysis.get("veto") and venue_id == top_candidates[0].get("venue_id", top_candidates[0].get("name")):
            overall_veto = True
            veto_reason = analysis.get("veto_reason")

    state["risk_flags"] = risk_flags
    state["veto"] = overall_veto
    state["veto_reason"] = veto_reason

    return state

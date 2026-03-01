"""
Final Synthesis Node — Ranks venues and generates human-readable explanations.

After all analysts and the Critic have run, the Synthesiser:
  1. Collects all agent outputs (vibe, access, cost, risk).
  2. Applies the Commander's dynamic weights to compute a composite score.
  3. Uses Gemini to generate "Why this venue" and "Watch out" text.
  4. Produces the ranked_results list matching the VenueResult schema.
"""

import asyncio
import json
import logging

from app.models.state import PathfinderState
from app.services.gemini import generate_content

logger = logging.getLogger(__name__)


_SYNTHESIS_PROMPT = """You are the PATHFINDER Synthesiser. Given the analysis data for a venue, produce a concise summary.

Venue: {name}
Address: {address}
Category: {category}

Vibe Analysis: {vibe_data}
Cost Analysis: {cost_data}
Risk Flags: {risk_data}

User's Original Query: {raw_prompt}

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "why": "<1-2 sentence explanation of why this venue is a good fit for the user's query>",
  "watch_out": "<1 sentence warning about potential issues, or empty string if none>"
}}
"""

_GLOBAL_CONSENSUS_PROMPT = """You are the PATHFINDER Synthesiser. Given the top ranked venues for a user's query, produce a single comparative summary highlighting the best option based on price consensus, star rating, weather/risks, and vibe.

User's Query: {raw_prompt}

Top Venues Data:
{venues_data}

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "global_consensus": "<2-3 sentence comparative summary>",
  "email_draft": "<Draft a polite email to the top recommended venue inquiring about group availability, pricing confirmation, and mentioning the group size. Keep it professional.>"
}}
"""


def _compute_composite_score(
    venue_id: str,
    vibe_scores: dict,
    cost_profiles: dict,
    risk_flags: dict,
    agent_weights: dict,
) -> float:
    """
    Compute a weighted composite score (0.0–1.0) from all agent outputs.

    Default weights if not specified by Commander:
      vibe: 0.33, cost: 0.40, risk_penalty: 0.27
    """
    # Get individual scores
    vibe = vibe_scores.get(venue_id, {}).get("vibe_score")
    vibe_score = vibe if vibe is not None else 0.5

    cost_profile = cost_profiles.get(venue_id, {})
    cost_score = cost_profile.get("value_score", 0.5)

    # Risk penalty: high-severity risks reduce the score
    risks = risk_flags.get(venue_id, [])
    risk_penalty = 0.0
    for r in risks:
        severity = r.get("severity", "low") if isinstance(r, dict) else "low"
        if severity == "high":
            risk_penalty += 0.3
        elif severity == "medium":
            risk_penalty += 0.15
        else:
            risk_penalty += 0.05
    risk_penalty = min(risk_penalty, 0.5)  # Cap at 0.5
    risk_score = 1.0 - risk_penalty

    # Apply Commander weights
    w_vibe = agent_weights.get("vibe_matcher", 0.33)
    w_cost = agent_weights.get("cost_analyst", 0.40)
    w_risk = agent_weights.get("critic", 0.27)

    # Normalise weights
    total_w = w_vibe + w_cost + w_risk
    if total_w > 0:
        w_vibe /= total_w
        w_cost /= total_w
        w_risk /= total_w

    composite = (
        w_vibe * vibe_score +
        w_cost * cost_score +
        w_risk * risk_score
    )

    return round(max(0.0, min(1.0, composite)), 3)


async def _generate_explanation(
    venue: dict,
    vibe_data: dict,
    cost_data: dict,
    risk_data: list,
    raw_prompt: str,
) -> dict:
    """Use Gemini to generate Why/Watch Out text for a single venue."""
    prompt = _SYNTHESIS_PROMPT.format(
        name=venue.get("name", "Unknown"),
        address=venue.get("address", ""),
        category=venue.get("category", ""),
        vibe_data=json.dumps(vibe_data) if vibe_data else "N/A",
        cost_data=json.dumps(cost_data) if cost_data else "N/A",
        risk_data=json.dumps(risk_data) if risk_data else "None",
        raw_prompt=raw_prompt,
    )

    try:
        raw = await generate_content(prompt=prompt, model="gemini-2.5-flash")
        if not raw:
            return {"why": "", "watch_out": ""}

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]

        return json.loads(cleaned.strip())
    except Exception as exc:
        logger.warning("Synthesis explanation failed for %s: %s", venue.get("name"), exc)
        return {"why": "", "watch_out": ""}

async def _generate_global_consensus(top_venues: list, raw_prompt: str) -> tuple[str, str]:
    """Use Gemini to generate a global consensus and an email draft comparing top choices."""
    simplified_venues = []
    for rank, (composite, venue, vid) in enumerate(top_venues, 1):
        simplified_venues.append({
            "rank": rank,
            "name": venue.get("name"),
            "score": composite,
            "traits": venue
        })

    prompt = _GLOBAL_CONSENSUS_PROMPT.format(
        raw_prompt=raw_prompt,
        venues_data=json.dumps(simplified_venues, default=str)
    )

    try:
        raw = await generate_content(prompt=prompt, model="gemini-2.5-flash")
        if not raw:
            return "Consensus unavailable.", ""

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]

        data = json.loads(cleaned.strip())
        return data.get("global_consensus", ""), data.get("email_draft", "")
    except Exception as exc:
        logger.warning("Global consensus generation failed: %s", exc)
        return "Based on the options, these venues are the strongest matches.", ""


def synthesiser_node(state: PathfinderState) -> PathfinderState:
    """
    Final synthesis: rank venues and produce human-readable results.

    Steps
    -----
    1. Compute composite scores using all agent outputs + Commander weights.
    2. Sort venues by composite score (descending).
    3. Generate Gemini explanations for top 3.
    4. Build ranked_results matching VenueResult schema.
    """
    candidates = state.get("candidate_venues", [])
    if not candidates:
        logger.info("Synthesiser: no candidates to rank")
        return {"ranked_results": []}

    vibe_scores = state.get("vibe_scores", {})
    cost_profiles = state.get("cost_profiles", {})
    risk_flags = state.get("risk_flags", {})
    agent_weights = state.get("agent_weights", {})
    raw_prompt = state.get("raw_prompt", "")
    
    requires_oauth = state.get("requires_oauth", False)
    allowed_actions = state.get("allowed_actions", [])
    oauth_scopes = state.get("oauth_scopes", [])

    # Step 1: Score all venues
    scored = []
    for venue in candidates:
        vid = venue.get("venue_id", venue.get("name", "unknown"))
        composite = _compute_composite_score(
            vid, vibe_scores, cost_profiles, risk_flags, agent_weights
        )
        scored.append((composite, venue, vid))

    # Step 2: Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Step 3: Generate explanations for top 3
    top_venues = scored[:3]

    async def _explain_all():
        return await asyncio.gather(*[
            _generate_explanation(
                venue=venue,
                vibe_data=vibe_scores.get(vid, {}),
                cost_data=cost_profiles.get(vid, {}),
                risk_data=risk_flags.get(vid, []),
                raw_prompt=raw_prompt,
            )
            for _, venue, vid in top_venues
        ])

    try:
        explanations = asyncio.run(_explain_all())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        explanations = asyncio.run(_explain_all())
    except Exception as exc:
        logger.error("Synthesis explanations failed: %s", exc)
        explanations = [{"why": "", "watch_out": ""} for _ in top_venues]

    # Step 4: Build ranked_results
    ranked_results = []
    for rank, ((composite, venue, vid), explanation) in enumerate(zip(top_venues, explanations), 1):
        vibe_entry = vibe_scores.get(vid, {})
        cost_entry = cost_profiles.get(vid, {})

        ranked_results.append({
            "rank": rank,
            "name": venue.get("name", "Unknown"),
            "address": venue.get("address", ""),
            "lat": venue.get("lat", 0.0),
            "lng": venue.get("lng", 0.0),
            "vibe_score": vibe_entry.get("vibe_score"),
            "price_range": cost_entry.get("price_range"),
            "price_confidence": cost_entry.get("confidence", "none"),
            "why": explanation.get("why", ""),
            "watch_out": explanation.get("watch_out", ""),
        })

    logger.info("Synthesiser ranked %d venues (top 3 explained)", len(scored))

    # Step 5: Generate Global Consensus
    try:
        consensus_text, email_draft = asyncio.run(_generate_global_consensus(top_venues, raw_prompt))
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        consensus_text, email_draft = asyncio.run(_generate_global_consensus(top_venues, raw_prompt))

    action_request = None
    if requires_oauth and "send_email" in allowed_actions:
        action_request = {
            "type": "oauth_consent",
            "reason": f"To automatically email {top_venues[0][1].get('name', 'the top venue')} for availability.",
            "scopes": ["email.send"],
            "draft": email_draft
        }

    return {
        "ranked_results": ranked_results,
        "global_consensus": consensus_text,
        "action_request": action_request
    }

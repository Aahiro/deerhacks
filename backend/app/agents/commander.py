"""
Node 1 — The COMMANDER (Orchestrator)
Central brain: intent parsing, complexity tiering, dynamic agent weighting.
Model: Gemini 1.5 Flash
Fallback: Keyword-based heuristic when Gemini is unavailable.
"""

import json
import logging
import re
import asyncio

from app.models.state import PathfinderState
from app.services.gemini import generate_content

logger = logging.getLogger(__name__)


# ── Keyword dictionaries for fallback agent activation ────────────────

_COST_KEYWORDS = {
    "budget", "cheap", "affordable", "cost", "price", "pricing", "money",
    "expensive", "inexpensive", "free", "deal", "discount", "value",
    "economical", "low-cost", "under", "frugal", "save", "saving",
    "per person", "per head", "rates", "fees", "rental", "rent",
    "spend", "spending", "splurge", "worth", "overpriced",
    "$", "dollar", "bucks",
}

_VIBE_KEYWORDS = {
    "vibe", "vibes", "aesthetic", "cozy", "chill", "trendy", "hipster",
    "romantic", "classy", "upscale", "fancy", "elegant", "modern",
    "rustic", "bohemian", "artsy", "quirky", "retro", "vintage",
    "minimalist", "industrial", "dark academia", "cottagecore",
    "cyberpunk", "neon", "instagrammable", "photogenic", "cute",
    "charming", "intimate", "lively", "energetic", "fun", "exciting",
    "relaxing", "peaceful", "calm", "serene", "warm", "inviting",
    "atmosphere", "ambiance", "mood", "theme", "decor", "design",
    "beautiful", "pretty", "gorgeous", "stunning", "loves", "hates",
    "likes", "dislikes", "prefers", "preference",
}

_ACCESS_KEYWORDS = {
    "near", "nearby", "close", "closest", "walking distance", "transit",
    "subway", "bus", "ttc", "streetcar", "drive", "driving", "commute",
    "travel", "distance", "far", "location", "directions", "accessible",
    "accessibility", "parking", "bike", "cycling", "walkable",
    "downtown", "midtown", "uptown", "east end", "west end", "north",
    "south", "neighbourhood", "neighborhood", "area", "district",
    "central", "convenient", "easy to get to", "reachable",
}

_CRITIC_KEYWORDS = {
    "weather", "rain", "snow", "cold", "hot", "outdoor", "outside",
    "patio", "rooftop", "garden", "park", "beach", "pool",
    "risk", "risky", "safe", "safety", "reliable", "dependable",
    "crowded", "busy", "packed", "noisy", "quiet", "event",
    "festival", "concert", "marathon", "parade", "construction",
    "closure", "closed", "cancelled", "available", "availability",
    "weekend", "saturday", "sunday", "tonight", "today", "tomorrow",
    "evening", "afternoon", "morning", "night",
}

_BUDGET_VALUES = {
    "free": "free", "cheap": "low", "budget": "low", "affordable": "low",
    "inexpensive": "low", "economical": "low", "frugal": "low",
    "low-cost": "low", "moderate": "medium", "mid-range": "medium",
    "reasonable": "medium", "expensive": "high", "upscale": "high",
    "luxury": "high", "premium": "high", "splurge": "high",
    "fancy": "high", "high-end": "high",
}


def _keyword_fallback(raw_prompt: str) -> dict:
    """
    Heuristic fallback when Gemini is unavailable.
    Parses intent and activates agents using keyword matching.
    """
    prompt_lower = raw_prompt.lower()
    words = set(re.findall(r"[a-z$]+(?:'[a-z]+)?", prompt_lower))

    # ── Extract intent fields ──
    # Activity: take the core noun phrase (best-effort)
    activity = raw_prompt.split(" in ")[0].strip() if " in " in raw_prompt else raw_prompt

    # Location: look for "in <location>" pattern
    loc_match = re.search(r"\bin\s+([\w\s]+?)(?:\.|,|$)", raw_prompt, re.IGNORECASE)
    location = loc_match.group(1).strip() if loc_match else "Toronto"

    # Group size: look for numbers near "people", "kids", "guests", "friends", "group"
    size_match = re.search(
        r"(\d+)\s*(?:people|person|kids|children|guests|friends|of us|group)",
        raw_prompt, re.IGNORECASE,
    )
    group_size = int(size_match.group(1)) if size_match else 1

    # Budget: keyword match
    budget = "medium"
    for kw, val in _BUDGET_VALUES.items():
        if kw in prompt_lower:
            budget = val
            break

    # Budget from dollar amounts: "$400-600", "under $200"
    dollar_match = re.search(r"\$\s*(\d+)", raw_prompt)
    if dollar_match:
        amount = int(dollar_match.group(1))
        if amount < 100:
            budget = "low"
        elif amount < 500:
            budget = "medium"
        else:
            budget = "high"

    # Vibe: pick first matching vibe keyword
    vibe = None
    for kw in _VIBE_KEYWORDS:
        if kw in prompt_lower:
            vibe = kw
            break

    parsed_intent = {
        "activity": activity,
        "group_size": group_size,
        "budget": budget,
        "location": location,
        "vibe": vibe,
    }

    # ── Determine which agents to activate ──
    active_agents = ["scout"]  # always on
    agent_weights = {"scout": 1.0}

    # Check each agent's keywords against the prompt
    cost_hits = sum(1 for kw in _COST_KEYWORDS if kw in prompt_lower)
    vibe_hits = sum(1 for kw in _VIBE_KEYWORDS if kw in prompt_lower)
    access_hits = sum(1 for kw in _ACCESS_KEYWORDS if kw in prompt_lower)
    critic_hits = sum(1 for kw in _CRITIC_KEYWORDS if kw in prompt_lower)

    if cost_hits > 0:
        active_agents.append("cost_analyst")
        agent_weights["cost_analyst"] = min(0.5 + cost_hits * 0.1, 1.0)

    if vibe_hits > 0:
        active_agents.append("vibe_matcher")
        agent_weights["vibe_matcher"] = min(0.4 + vibe_hits * 0.1, 1.0)

    if access_hits > 0:
        active_agents.append("access_analyst")
        agent_weights["access_analyst"] = min(0.4 + access_hits * 0.1, 1.0)

    if critic_hits > 0:
        active_agents.append("critic")
        agent_weights["critic"] = min(0.4 + critic_hits * 0.1, 1.0)

    # If group activity with multiple people, default-activate cost + access + critic
    if group_size > 1:
        for agent in ["cost_analyst", "access_analyst", "critic"]:
            if agent not in active_agents:
                active_agents.append(agent)
                agent_weights[agent] = 0.5

    # If nothing beyond scout matched, activate all with moderate weight
    if len(active_agents) == 1:
        active_agents = ["scout", "vibe_matcher", "cost_analyst", "access_analyst", "critic"]
        agent_weights = {a: 0.6 for a in active_agents}
        agent_weights["scout"] = 1.0

    # ── Determine complexity tier ──
    n_agents = len(active_agents)
    if n_agents <= 2:
        tier = "tier_1"
    elif n_agents <= 4:
        tier = "tier_2"
    else:
        tier = "tier_3"

    logger.info(
        "Commander FALLBACK: tier=%s, agents=%s, budget=%s, group=%d",
        tier, active_agents, budget, group_size,
    )

    return {
        "parsed_intent": parsed_intent,
        "complexity_tier": tier,
        "active_agents": active_agents,
        "agent_weights": agent_weights,
    }


# ── User profile weight adjustment ───────────────────────────────────

def _apply_user_profile_weights(agent_weights: dict, user_profile: dict) -> dict:
    """
    Nudge agent weights based on the authenticated user's saved preferences.

    Preferences come from the Auth0 JWT's app_metadata.preferences field:
      budget_sensitive    → boost cost_analyst weight
      accessibility_priority → boost access_analyst weight
      vibe_first          → boost vibe_matcher weight
      risk_averse         → boost critic weight
    """
    prefs = user_profile.get("app_metadata", {}).get("preferences", {})
    if not prefs:
        return agent_weights

    weights = dict(agent_weights)  # copy so we don't mutate

    if prefs.get("budget_sensitive"):
        weights["cost_analyst"] = min(1.0, weights.get("cost_analyst", 0.5) + 0.2)
        logger.info("Commander: boosting cost_analyst (budget_sensitive user)")

    if prefs.get("accessibility_priority"):
        weights["access_analyst"] = min(1.0, weights.get("access_analyst", 0.5) + 0.2)
        logger.info("Commander: boosting access_analyst (accessibility_priority user)")

    if prefs.get("vibe_first"):
        weights["vibe_matcher"] = min(1.0, weights.get("vibe_matcher", 0.5) + 0.2)
        logger.info("Commander: boosting vibe_matcher (vibe_first user)")

    if prefs.get("risk_averse"):
        weights["critic"] = min(1.0, weights.get("critic", 0.5) + 0.2)
        logger.info("Commander: boosting critic (risk_averse user)")

    return weights


# ── Main node entry point ─────────────────────────────────────────────

def commander_node(state: PathfinderState) -> PathfinderState:
    """
    Parse the raw user prompt into a structured execution plan.

    Steps
    -----
    1. Call Gemini 1.5 Flash to classify intent & extract parameters.
    2. Determine complexity tier (quick / full / adversarial).
    3. Compute dynamic agent weights based on keywords.
    4. Return updated state with parsed_intent, complexity_tier, agent_weights.

    Fallback: If Gemini is unavailable, use keyword-based heuristics.
    """
    raw_prompt = state.get("raw_prompt", "")

    prompt = f"""
    You are the PATHFINDER Commander Agent. Analyze the user's query and output a JSON execution plan.
    Query: "{raw_prompt}"
    
    Determine:
    1. Intent parameters (activity, group_size, budget, location, vibe).
    2. Complexity Tier:
       - 'tier_1': Simple lookup (Scout only or light analysis)
       - 'tier_2': Multi-factor personal (Group activity, constraints -> Scout, Cost, Access, Critic, maybe Vibe)
       - 'tier_3': Strategic/Business (Deep research -> all 5 agents)
    3. Active Agents: List the agents to activate from: ["scout", "vibe_matcher", "access_analyst", "cost_analyst", "critic"]. Scout is always mandatory.
    4. Agent Weights: Assign a float (0.0 to 1.0) to each activated agent indicating its importance.
    
    Output exactly in this JSON format:
    {{
      "parsed_intent": {{
        "activity": "...",
        "group_size": 10,
        "budget": "low",
        "location": "west end",
        "vibe": "..."
      }},
      "complexity_tier": "tier_2",
      "active_agents": ["scout", "cost_analyst", "access_analyst", "critic"],
      "agent_weights": {{
        "scout": 1.0,
        ...
      }}
    }}
    Do not output markdown code blocks. Only the raw JSON string.
    """

    plan = None
    context = []

    try:
        response_text = asyncio.run(generate_content(prompt))

        # Clean up possible markdown artifacts
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        plan = json.loads(response_text.strip())

    except Exception as e:
        logger.warning("Commander Gemini call failed: %s — using keyword fallback", e)
        plan = _keyword_fallback(raw_prompt)

    agent_weights = plan.get("agent_weights", {"scout": 1.0})

    user_profile = state.get("user_profile", {})
    if user_profile:
        agent_weights = _apply_user_profile_weights(agent_weights, user_profile)

    return {
        "parsed_intent": plan.get("parsed_intent", {}),
        "complexity_tier": plan.get("complexity_tier", "tier_2"),
        "active_agents": plan.get("active_agents", ["scout"]),
        "agent_weights": agent_weights,
        "veto": False,  # clear previous veto so Critic evaluates fresh on retry
        # only increment when we're actually retrying (incoming veto was True)
        "retry_count": state.get("retry_count", 0) + (1 if state.get("veto") else 0),
    }

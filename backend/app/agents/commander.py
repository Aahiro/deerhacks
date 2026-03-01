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
    critic_hits = sum(1 for kw in _CRITIC_KEYWORDS if kw in prompt_lower)

    if cost_hits > 0:
        active_agents.append("cost_analyst")
        agent_weights["cost_analyst"] = min(0.5 + cost_hits * 0.1, 1.0)

    if vibe_hits > 0:
        active_agents.append("vibe_matcher")
        agent_weights["vibe_matcher"] = min(0.4 + vibe_hits * 0.1, 1.0)

    if critic_hits > 0:
        active_agents.append("critic")
        agent_weights["critic"] = min(0.4 + critic_hits * 0.1, 1.0)

    # If group activity with multiple people, default-activate cost + critic
    if group_size > 1:
        for agent in ["cost_analyst", "critic"]:
            if agent not in active_agents:
                active_agents.append(agent)
                agent_weights[agent] = 0.5

    # If nothing beyond scout matched, activate all with moderate weight
    if len(active_agents) == 1:
        active_agents = ["scout", "vibe_matcher", "cost_analyst", "critic"]
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


def _apply_user_profile_weights(
    agent_weights: dict, user_profile: dict
) -> dict:
    """
    Adjust agent weights based on the authenticated user's profile metadata.
    Reads app_metadata set by Auth0 rules/actions.
    """
    meta = user_profile.get("app_metadata", {})
    preferences = meta.get("preferences", {})

    # Budget sensitivity
    if preferences.get("budget_sensitive"):
        agent_weights["cost_analyst"] = min(
            agent_weights.get("cost_analyst", 0.5) + 0.2, 1.0
        )

    # Vibe-first user
    if preferences.get("vibe_first"):
        agent_weights["vibe_matcher"] = min(
            agent_weights.get("vibe_matcher", 0.4) + 0.2, 1.0
        )

    # Risk-averse user
    if preferences.get("risk_averse"):
        agent_weights["critic"] = min(
            agent_weights.get("critic", 0.4) + 0.2, 1.0
        )

    return agent_weights


# ── Main node entry point ─────────────────────────────────────────────

def commander_node(state: PathfinderState) -> PathfinderState:
    """
    Parse the raw user prompt into a structured execution plan.

    Steps
    -----
    1. Check Auth0 identity and load user profile metadata.
    2. Call Gemini 1.5 Flash to classify intent & extract parameters.
    3. Determine complexity tier (quick / full / adversarial).
    4. Compute dynamic agent weights based on keywords and user profile.
    5. Return updated state with parsed_intent, complexity_tier, agent_weights, user_profile.

    Fallback: If Gemini is unavailable, use keyword-based heuristics.
    """
    raw_prompt = state.get("raw_prompt", "")
    auth_user_id = state.get("auth_user_id")
    
    # ── Fetch Auth0 Profile if available ──
    user_profile = state.get("user_profile")
    if auth_user_id and not user_profile:
        if auth_user_id == "auth0|local_test" or not auth_user_id.startswith("auth0|"):
            logger.info("Skipping Auth0 Management API lookup for simulated or non-standard user_id.")
            user_profile = {"app_metadata": {"preferences": {"budget_sensitive": False, "vibe_first": True}}} # standard profile
        else:
            from app.services.auth0 import auth0_service
            try:
                user_profile = asyncio.run(auth0_service.get_user_profile(auth_user_id))
            except RuntimeError:
                import nest_asyncio
                nest_asyncio.apply()
                user_profile = asyncio.run(auth0_service.get_user_profile(auth_user_id))
            except Exception as e:
                logger.warning(f"Failed to fetch user profile in Commander: {e}")
                user_profile = {}
    
    profile_context = ""
    if user_profile:
        prefs = user_profile.get("app_metadata", {}).get("preferences", {})
        if prefs:
            profile_context = f"\nUser Preferences Context: {json.dumps(prefs)}\nAdjust agent weights to favor these preferences."

    prompt = f"""You are the PATHFINDER Commander. Your first task is to establish the Execution Context.

OAUTH & IDENTITY LOGIC:
Check the user_id. If it is auth0|local_test or looks simulated, set identity_context to "standard_profile".
Do NOT request a Management API lookup if the user_id does not follow the auth0|{{id}} format.
Annotate the plan with requires_auth: false if the user is just looking for public cafes.

COMPLEXITY TIERING:
For "Cyberpunk Cafe", activate: SCOUT, VIBE, ACCESS, CRITIC.
Note: Skip COST if the intent is purely aesthetic and no booking is requested to save time.

    Query: "{raw_prompt}"{profile_context}
    
    Determine:
    1. Intent parameters (activity, group_size, budget, location, vibe).
    2. Complexity Tier:
       - 'tier_1': Simple lookup (Scout only or light analysis)
       - 'tier_2': Multi-factor personal (Group activity, constraints -> Scout, Cost, Access, Critic, maybe Vibe)
       - 'tier_3': Strategic/Business (Deep research -> all 5 agents)
    3. Active Agents: List the agents to activate from: ["scout", "vibe_matcher", "cost_analyst", "critic"]. Scout is always mandatory.
       IMPORTANT: DO NOT activate "vibe_matcher" unless the user's query specifically mentions aesthetics, vibes, beauty, theme, or atmosphere. For all other queries, omit it.
       IMPORTANT: Skip "cost_analyst" if the intent is purely aesthetic and no booking is requested.
    4. Agent Weights: Assign a float (0.0 to 1.0) to each activated agent indicating its importance.
    5. OAuth Requirements: Detect if this request requires acting on behalf of the user (e.g., booking, sending an email, checking a calendar).
       - If yes, set "requires_oauth": true, and list the "oauth_scopes" (e.g., "email.send", "calendar.read") and "allowed_actions" (e.g., "send_email", "check_availability").
       - If no, set "requires_oauth": false, and leave arrays empty.
    
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
      "active_agents": ["scout", "cost_analyst", "critic"],
      "agent_weights": {{
        "scout": 1.0,
        ...
      }},
      "requires_oauth": false,
      "oauth_scopes": [],
      "allowed_actions": [],
      "identity_context": "standard_profile"
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

    # If we're on a retry (veto or fast_fail was set), clear the flags and
    # increment retry_count so the cap in _should_retry works correctly.
    was_retry = bool(state.get("veto") or state.get("fast_fail"))

    return {
        "parsed_intent": plan.get("parsed_intent", {}),
        "complexity_tier": plan.get("complexity_tier", "tier_2"),
        "active_agents": plan.get("active_agents", ["scout"]),
        "agent_weights": plan.get("agent_weights", {"scout": 1.0}),
        "requires_oauth": plan.get("requires_oauth", False),
        "oauth_scopes": plan.get("oauth_scopes", []),
        "allowed_actions": plan.get("allowed_actions", []),
        "user_profile": user_profile,  # Pass profile down to other agents
        # Clear retry flags so the Critic starts fresh on the second pass
        "veto": False,
        "fast_fail": False,
        "veto_reason": None,
        "fast_fail_reason": None,
        "retry_count": state.get("retry_count", 0) + (1 if was_retry else 0),
    }

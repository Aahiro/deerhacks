"""
Shared state definition for the LangGraph workflow.
"""

from typing import TypedDict, List, Optional, Any


class PathfinderState(TypedDict, total=False):
    """Shared state passed between all LangGraph nodes."""

    # ── Input & Identity ──
    raw_prompt: str
    parsed_intent: dict
    auth_user_id: Optional[str]
    user_profile: Optional[dict]
    # ── Commander outputs ──
    complexity_tier: str          # "tier_1" | "tier_2" | "tier_3"
    active_agents: List[str]      # which agents to run for this query
    agent_weights: dict           # e.g. {"vibe": 0.3, "cost": 0.5, ...}
    
    # ── OAuth Detection (Node 1) ──
    requires_oauth: bool
    oauth_scopes: List[str]
    allowed_actions: List[str]
    
    # ── Scout outputs ──
    candidate_venues: List[dict]  # list of raw node dictionaries
    
    # ── Vibe Matcher outputs ──
    vibe_scores: dict             # venue_id → score

    # ── Cost ──
    cost_profiles: dict             # venue_id → {"price_range": "$$", "confidence": "high", "value_score": 0.5}

    # ── Critic outputs ──
    risk_flags: dict              # venue_id → [risk strings]
    veto: bool                    # True if the Critic forced a retry
    veto_reason: Optional[str]
    fast_fail: bool               # True if Critical early termination triggered (Condition A or B)
    fast_fail_reason: Optional[str]

    # ── Final ranked output ──
    ranked_results: List[dict]
    action_request: Optional[dict] # For OAuth UI presentation in Synthesiser

    # ── User-provided context ──
    member_locations: List[dict]  # [{lat, lng}] for each group member

    # ── Snowflake memory ──
    snowflake_context: Optional[dict]  # Historical risk data from Snowflake

    # ── Chat reprompting ──
    chat_history: Optional[List[dict]]  # [{role, content}] for conversational loop



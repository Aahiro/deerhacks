"""
PATHFINDER API routes.
"""

from fastapi import APIRouter

from app.schemas import PlanRequest, PlanResponse

router = APIRouter()


@router.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    """
    Accept a natural-language activity request and return ranked venues.

    Flow: prompt → Commander → Scout → [Vibe, Access, Cost] → Critic → Synthesiser → results
    """
    from app.graph import pathfinder_graph

    initial_state = {
        "raw_prompt": request.prompt,
        "parsed_intent": {},
        "complexity_tier": "tier_2",
        "active_agents": [],
        "agent_weights": {},
        "candidate_venues": [],
        "vibe_scores": {},
        "accessibility_scores": {},
        "isochrones": {},
        "cost_profiles": {},
        "risk_flags": {},
        "veto": False,
        "veto_reason": None,
        "ranked_results": [],
        # Forward request params for agents to use
        "member_locations": request.member_locations or [],
    }

    # Inject explicit fields into parsed_intent if provided
    if request.group_size > 1 or request.budget or request.location or request.vibe:
        initial_state["parsed_intent"] = {
            "group_size": request.group_size,
            "budget": request.budget,
            "location": request.location,
            "vibe": request.vibe,
        }

    # Run the full LangGraph workflow
    result = await pathfinder_graph.ainvoke(initial_state)

    return PlanResponse(
        venues=result.get("ranked_results", []),
        execution_summary="Pipeline complete.",
    )


@router.get("/health")
async def api_health():
    return {"status": "ok"}

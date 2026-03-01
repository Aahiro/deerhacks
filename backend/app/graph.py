"""
LangGraph workflow — assembles all agent nodes into the PATHFINDER graph.
"""

from langgraph.graph import StateGraph, END
import asyncio

from app.models.state import PathfinderState
from app.agents.commander import commander_node
from app.agents.scout import scout_node
from app.agents.vibe_matcher import vibe_matcher_node
from app.agents.cost_analyst import cost_analyst_node
from app.agents.critic import critic_node
from app.agents.synthesiser import synthesiser_node


def _should_retry(state: PathfinderState) -> str:
    """Conditional edge: retry once if the Critic vetoed or fast_failed. Cap at 1 to prevent infinite loops."""
    if (state.get("fast_fail") or state.get("veto")) and state.get("retry_count", 0) < 1:
        return "commander"
    return "synthesiser"


async def parallel_analysts_node(state: PathfinderState) -> PathfinderState:
    """
    Runs the Vibe Matcher, Cost Analyst, and Critic concurrently.
    Merges their returned states.
    If the Critic returns early with fast_fail, it overrides remaining long-running tasks.
    """
    active = state.get("active_agents", [])
    
    # Define mapping of agent names to their node functions
    agent_map = {
        "vibe_matcher": vibe_matcher_node,
        "cost_analyst": cost_analyst_node,
        "critic": critic_node
    }

    # Determine which analysts to run
    tasks = []
    # Always spawn in thread so we don't block the async event loop with sync code
    for name, func in agent_map.items():
        if not active or name in active:
            # Pass a shallow copy so mutations don't corrupt the loop state if they modify `state` directly
            tasks.append(asyncio.to_thread(func, state.copy()))

    if not tasks:
        return {}

    # Run them all concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge returned state updates
    merged_state = {}
    for res in results:
        if isinstance(res, Exception):
            import logging
            logging.getLogger(__name__).error("Parallel analyst failed: %s", res)
            continue
        if isinstance(res, dict):
            merged_state.update(res)

    return merged_state


def build_graph() -> StateGraph:
    """Construct and compile the PATHFINDER LangGraph."""

    graph = StateGraph(PathfinderState)

    # ── Register nodes ──
    graph.add_node("commander", commander_node)
    graph.add_node("scout", scout_node)
    
    # ── Instead of 4 sequential nodes, we use the unified parallel runner ──
    graph.add_node("parallel_analysts", parallel_analysts_node)
    
    graph.add_node("synthesiser", synthesiser_node)

    # ── Define edges ──
    graph.set_entry_point("commander")
    graph.add_edge("commander", "scout")
    
    # Fan out to analysts running concurrently
    graph.add_edge("scout", "parallel_analysts")

    # ── Conditional retry or synthesis ──
    graph.add_conditional_edges("parallel_analysts", _should_retry, {
        "commander": "commander",
        "synthesiser": "synthesiser",
    })

    # Synthesiser → END
    graph.add_edge("synthesiser", END)

    return graph.compile()


# Singleton compiled graph
pathfinder_graph = build_graph()

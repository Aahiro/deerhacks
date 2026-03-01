"""
LangGraph workflow — assembles all agent nodes into the PATHFINDER graph.
"""

import asyncio
import logging

from langgraph.graph import StateGraph, END

from app.models.state import PathfinderState
from app.agents.commander import commander_node
from app.agents.scout import scout_node
from app.agents.vibe_matcher import vibe_matcher_node
from app.agents.cost_analyst import cost_analyst_node
from app.agents.critic import critic_node
from app.agents.synthesiser import synthesiser_node

logger = logging.getLogger(__name__)


def _should_retry(state: PathfinderState) -> str:
    """Conditional edge: retry once if the Critic vetoed or fast_failed. Cap at 1 to prevent infinite loops."""
    if (state.get("fast_fail") or state.get("veto")) and state.get("retry_count", 0) < 1:
        return "commander"
    return "synthesiser"


async def parallel_analysts_node(state: PathfinderState) -> PathfinderState:
    """
    Runs Vibe Matcher, Cost Analyst, and Critic concurrently with per-agent timeouts.

    - vibe_matcher and critic are async — called directly (no thread wrapping needed).
    - cost_analyst is pure-sync — wrapped in asyncio.to_thread().
    - Each agent gets 45 s; any agent that times out or throws returns a safe
      empty fallback so the pipeline always reaches the Synthesiser.
    """
    active = state.get("active_agents", [])

    # ── Build tasks ──────────────────────────────────────────────────────────
    tasks: list = []
    task_names: list = []

    if not active or "vibe_matcher" in active:
        tasks.append(asyncio.wait_for(vibe_matcher_node(state.copy()), timeout=45.0))
        task_names.append("vibe_matcher")

    if not active or "critic" in active:
        tasks.append(asyncio.wait_for(critic_node(state.copy()), timeout=45.0))
        task_names.append("critic")

    # cost_analyst is synchronous — run in a thread so it never blocks the event loop
    if not active or "cost_analyst" in active:
        tasks.append(
            asyncio.wait_for(asyncio.to_thread(cost_analyst_node, state.copy()), timeout=10.0)
        )
        task_names.append("cost_analyst")

    if not tasks:
        return {}

    # ── Run concurrently ─────────────────────────────────────────────────────
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Merge state; provide safe fallbacks for any failed agent ─────────────
    merged_state: dict = {}
    for name, res in zip(task_names, results):
        if isinstance(res, Exception):
            logger.error(
                "Analyst '%s' failed (%s: %s) — using fallback state",
                name,
                type(res).__name__,
                res,
            )
            # Graceful fallback so Synthesiser always has keys to read
            if name == "vibe_matcher":
                merged_state.setdefault("vibe_scores", {})
            elif name == "cost_analyst":
                merged_state.setdefault("cost_profiles", {})
            elif name == "critic":
                merged_state.setdefault("risk_flags", {})
                merged_state.setdefault("fast_fail", False)
                merged_state.setdefault("fast_fail_reason", None)
                merged_state.setdefault("veto", False)
                merged_state.setdefault("veto_reason", None)
        elif isinstance(res, dict):
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

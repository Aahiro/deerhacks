"""
LangGraph workflow — assembles all agent nodes into the PATHFINDER graph.
"""

from langgraph.graph import StateGraph, END

from app.models.state import PathfinderState
from app.agents.commander import commander_node
from app.agents.scout import scout_node
from app.agents.vibe_matcher import vibe_matcher_node
from app.agents.access_analyst import access_analyst_node
from app.agents.cost_analyst import cost_analyst_node
from app.agents.critic import critic_node
from app.agents.synthesiser import synthesiser_node


def _should_retry(state: PathfinderState) -> str:
    """Conditional edge: retry once if the Critic vetoed. After that, synthesise anyway."""
    if state.get("veto") and state.get("retry_count", 0) < 1:
        return "commander"
    return "synthesiser"


def _make_conditional_agent(agent_name: str, agent_fn):
    """Wrap an agent so it no-ops when not in active_agents list."""
    def wrapper(state: PathfinderState) -> PathfinderState:
        active = state.get("active_agents", [])
        # If Commander specified active agents and this one isn't listed, skip
        if active and agent_name not in active:
            return state
        return agent_fn(state)
    wrapper.__name__ = agent_fn.__name__
    return wrapper


def build_graph() -> StateGraph:
    """Construct and compile the PATHFINDER LangGraph."""

    graph = StateGraph(PathfinderState)

    # ── Register nodes ──
    graph.add_node("commander", commander_node)
    graph.add_node("scout", scout_node)
    graph.add_node("vibe_matcher", _make_conditional_agent("vibe_matcher", vibe_matcher_node))
    graph.add_node("access_analyst", _make_conditional_agent("access_analyst", access_analyst_node))
    graph.add_node("cost_analyst", _make_conditional_agent("cost_analyst", cost_analyst_node))
    graph.add_node("critic", _make_conditional_agent("critic", critic_node))
    graph.add_node("synthesiser", synthesiser_node)

    # ── Define edges ──
    # Sequential pipeline: Commander → Scout → analysts in series → Critic → Synthesiser
    # (Running analysts sequentially avoids LangGraph's parallel write conflicts)
    graph.set_entry_point("commander")
    graph.add_edge("commander", "scout")
    graph.add_edge("scout", "vibe_matcher")
    graph.add_edge("vibe_matcher", "access_analyst")
    graph.add_edge("access_analyst", "cost_analyst")
    graph.add_edge("cost_analyst", "critic")

    # ── Conditional retry or synthesis ──
    graph.add_conditional_edges("critic", _should_retry, {
        "commander": "commander",
        "synthesiser": "synthesiser",
    })

    # Synthesiser → END
    graph.add_edge("synthesiser", END)

    return graph.compile()


# Singleton compiled graph
pathfinder_graph = build_graph()

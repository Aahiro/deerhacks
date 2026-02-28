"""
Node 1 — The COMMANDER (Orchestrator)
Central brain: intent parsing, complexity tiering, dynamic agent weighting.
Model: Gemini 1.5 Flash
"""

import json
import logging
import asyncio

from app.models.state import PathfinderState
from app.services.gemini import generate_content
from app.services.snowflake import snowflake_service

logger = logging.getLogger(__name__)


def commander_node(state: PathfinderState) -> PathfinderState:
    """
    Parse the raw user prompt into a structured execution plan.

    Steps
    -----
    1. Call Gemini 1.5 Flash to classify intent & extract parameters.
    2. Determine complexity tier (quick / full / adversarial).
    3. Compute dynamic agent weights based on keywords.
    4. Query Snowflake for historical risk pre-check.
    5. Return updated state with parsed_intent, complexity_tier, agent_weights.
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
    
    try:
        response_text = asyncio.run(generate_content(prompt))
        
        # Clean up possible markdown artifacts
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        plan = json.loads(response_text.strip())
        
        state["parsed_intent"] = plan.get("parsed_intent", {})
        state["complexity_tier"] = plan.get("complexity_tier", "tier_2")
        state["active_agents"] = plan.get("active_agents", ["scout"])
        state["agent_weights"] = plan.get("agent_weights", {"scout": 1.0})
        
        # Snowflake historical risk pre-check
        # We query Snowflake Cortex for context based on the raw prompt
        context = snowflake_service.cortex_search(raw_prompt, top_k=2)
        state["snowflake_context"] = context
        
    except Exception as e:
        logger.error(f"Commander failed to parse intent: {e}")
        return {
            "parsed_intent": {},
            "complexity_tier": "tier_1",
            "active_agents": ["scout"],
            "agent_weights": {"scout": 1.0},
            "snowflake_context": []
        }
        
    return {
        "parsed_intent": plan.get("parsed_intent", {}),
        "complexity_tier": plan.get("complexity_tier", "tier_2"),
        "active_agents": plan.get("active_agents", ["scout"]),
        "agent_weights": plan.get("agent_weights", {"scout": 1.0}),
        "snowflake_context": context
    }

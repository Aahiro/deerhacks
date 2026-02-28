import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import nest_asyncio
import asyncio
nest_asyncio.apply()

from app.graph import pathfinder_graph
from app.models.state import PathfinderState

def run_test():
    prompt = "I need a budget-friendly escape room for 4 people in downtown Toronto this Saturday at 2 PM."
    print("Testing Graph with prompt:", prompt)
    
    state = PathfinderState(raw_prompt=prompt)
    
    print("\n[Executing Graph...]")
    final_state = pathfinder_graph.invoke(state)
    
    print("\n--- FINAL STATE ---")
    intent = final_state.get("parsed_intent", {})
    print("Intent:", intent)
    print("Complexity Tier:", final_state.get("complexity_tier"))
    
    active = final_state.get("active_agents", [])
    print("Active Agents originally proposed:", active)
    
    candidates = final_state.get("candidate_venues", [])
    print(f"\nFound {len(candidates)} candidates.")
    for v in candidates[:3]:
        print(f" - {v.get('name')} (Rating: {v.get('rating')}) [ID: {v.get('venue_id')}]")
        
    print("\nCost Profiles computed for:", list(final_state.get("cost_profiles", {}).keys()))
    print("Vibe Scores computed for:", list(final_state.get("vibe_scores", {}).keys()))
    
    print("\nCritic Veto:", final_state.get("veto"))
    if final_state.get("veto"):
        print("Critic Veto Reason:", final_state.get("veto_reason"))
        
    print("\nRisk Flags:")
    flags = final_state.get("risk_flags", {})
    for vid, risks in flags.items():
        if risks:
            print(f" - {vid}: {risks}")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        import traceback
        traceback.print_exc()

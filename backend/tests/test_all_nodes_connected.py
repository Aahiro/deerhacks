import asyncio
import logging
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from app.graph import pathfinder_graph
from app.models.state import PathfinderState

async def run_integration_test():
    print("=" * 60)
    print("PATHFINDER Full Graph Integration Test (Nodes 1-7)")
    print("=" * 60)

    # We use a query that will trigger all analysts (group activity, budget, vibe, risk)
    query = "Find a cheap, aesthetic indoor basketball court for 10 people in downtown Toronto for tomorrow afternoon"
    
    initial_state = PathfinderState(
        raw_prompt=query,
        auth_user_id="auth0|test_user"
    )

    print(f"\n[1] Invoking LangGraph with query: '{query}'")
    
    final_state = None
    try:
        # Actually run the LangGraph
        final_state = await pathfinder_graph.ainvoke(initial_state)
    except Exception as e:
        print(f"Graph execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[2] Graph Execution Complete. Validating State Outputs...\n")

    # 1. Commander Output
    print(f"--> Commander Intent: {json.dumps(final_state.get('parsed_intent', {}))}")
    print(f"--> Active Agents: {final_state.get('active_agents')}")
    print(f"--> OAuth Required: {final_state.get('requires_oauth')}")
    print(f"--> OAuth Scopes: {final_state.get('oauth_scopes')}")
    print(f"--> Allowed Actions: {final_state.get('allowed_actions')}")

    # 2. Scout Output
    candidates = final_state.get('candidate_venues', [])
    print(f"\n--> Scout found {len(candidates)} candidates.")

    if not candidates:
        print("Scout failed to find anything. Cannot test analysts.")
        return

    # 3. Parallel Analysts Output
    print(f"\n--> Vibe Scores Processed: {len(final_state.get('vibe_scores', {}))}")
    print(f"--> Access Scores Processed: {len(final_state.get('accessibility_scores', {}))}")
    print(f"--> Cost Profiles Processed: {len(final_state.get('cost_profiles', {}))}")
    print(f"--> Risk Flags Processed: {len(final_state.get('risk_flags', {}))}")
    
    print(f"--> Fast Fail Triggered: {final_state.get('fast_fail', False)}")
    if final_state.get('fast_fail'):
        print(f"    Reason: {final_state.get('fast_fail_reason')}")

    # 4. Synthesiser Output
    ranked = final_state.get('ranked_results', [])
    print(f"\n--> Synthesiser Ranked Venues: {len(ranked)}")
    
    for venue in ranked:
        print(f"    - Rank {venue.get('rank')}: {venue.get('name')}")
        print(f"      Why: {venue.get('why')}")
        print(f"      Watch Out: {venue.get('watch_out')}")
        
    action_req = final_state.get('action_request')
    if action_req:
        print(f"\n--> Action Request (Node 7 / Auth0 Hook) generated successfully:")
        print(json.dumps(action_req, indent=2))
        
    print("\nIntegration Test Finished.")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(run_integration_test())

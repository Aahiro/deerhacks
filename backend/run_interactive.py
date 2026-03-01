import asyncio
import logging
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Add the backend directory to python path
sys.path.insert(0, os.path.dirname(__file__))

# Suppress excessive HTTP/API logging for a cleaner CLI experience
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# You can change this to logging.INFO if you want to see the agents' inner thoughts
logging.getLogger("app").setLevel(logging.WARNING) 

from app.graph import pathfinder_graph
from app.models.state import PathfinderState

async def main():
    print("======================================================")
    print("               PATHFINDER INTERACTIVE CLI             ")
    print("======================================================")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_input = input("\n[You]: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ['quit', 'exit']:
            break
            
        if not user_input.strip():
            continue
            
        print("\n[PATHFINDER is gathering intelligence... this may take 30-60 seconds]")
        
        # We initialize the LangGraph state with your prompt
        initial_state = PathfinderState(
            raw_prompt=user_input,
            auth_user_id="auth0|local_test" # Mock user
        )
        
        try:
            # Execute the LangGraph workflow
            final_state = await pathfinder_graph.ainvoke(initial_state)
            
            print("\n" + "="*50)
            print("                     RESULTS                    ")
            print("="*50 + "\n")
            
            # Check if the Critic killed the plan early
            if final_state.get('fast_fail'):
                print(f"❌ PLAN VETOED (Fast-Fail): {final_state.get('fast_fail_reason')}")
                continue
                
            # Print ranked venues from Synthesiser
            ranked = final_state.get('ranked_results', [])
            if not ranked:
                print("No viable venues found. Try changing your search parameters.")
            else:
                for venue in ranked:
                    print(f"🏆 Rank #{venue.get('rank', '?')}: \033[1m{venue.get('name')}\033[0m")
                    print(f"   💰 Est. Cost  : ${venue.get('cost', 'Unknown')} (Value Score: {venue.get('value_score', 0)})")
                    print(f"   ✨ Vibe Match : {venue.get('vibe_score', 0)}")
                    print(f"   📍 Accessibility: {venue.get('access_score', 0)}")
                    print(f"\n   ✅ Why it fits:")
                    print(f"      {venue.get('why', '')}")
                    print(f"\n   ⚠️ Watch out for:")
                    print(f"      {venue.get('watch_out', '')}")
                    print("-" * 50)
                    
            # Print any Auth0 Action Requests
            action_req = final_state.get('action_request')
            if action_req:
                print("\n🔐 [AUTH0 SECURE ACTION REQUIRED]")
                print(f"   {action_req.get('reason')}")
                print(f"   Requested Scopes: {', '.join(action_req.get('scopes', []))}")
                
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nGoodbye!")

if __name__ == "__main__":
    # Required to run asyncio event loops inside nested contexts (like some shells)
    import nest_asyncio
    nest_asyncio.apply()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")

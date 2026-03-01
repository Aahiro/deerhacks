"""Cost Analyst test -- testing venues with explicit pricing pages."""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
logging.basicConfig(level=logging.WARNING)

from app.agents.cost_analyst import cost_analyst_node

state = {
    "raw_prompt": "fun group activities in toronto",
    "parsed_intent": {"activity": "group activity", "location": "Toronto", "group_size": 4},
    "candidate_venues": [
        {
            "venue_id": "test_rock_climbing",
            "name": "Basecamp Climbing Queen West",
            "category": "rock_climbing_gym",
            "website": "https://basecampclimbing.ca/queen/pricing/",
            "source": "manual_test"
        },
        {
            "venue_id": "test_escape_room",
            "name": "Captive Escape Rooms Toronto",
            "category": "escape_room",
            "website": "https://captiverooms.com/toronto",
            "source": "manual_test"
        }
    ],
    "cost_profiles": {},
}

import logging
logging.getLogger("app.agents.cost_analyst").setLevel(logging.DEBUG)

print(f"Testing {len(state['candidate_venues'])} venues with explicit pricing websites...")
updated_state = cost_analyst_node(state)
profiles = updated_state.get("cost_profiles", {})

with open("cost_results_explicit.txt", "w", encoding="utf-8") as f:
    f.write("Cost Analyst Results (Explicit Pricing Venues)\n")
    f.write("=" * 50 + "\n\n")

    for vid, p in profiles.items():
        name = next((v["name"] for v in state["candidate_venues"] if v["venue_id"] == vid), vid)
        f.write(f"--- {name} ---\n")
        f.write(f"  Base cost:       ${p.get('base_cost', 0):.2f}\n")
        hidden = p.get("hidden_costs", [])
        for h in hidden:
            f.write(f"  Hidden fee:      {h.get('label','?')} -- ${h.get('amount',0):.2f}\n")
        f.write(f"  Total (TCA):     ${p.get('total_cost_of_attendance', 0):.2f}\n")
        f.write(f"  Per person (/4): ${p.get('per_person', 0):.2f}\n")
        f.write(f"  Value score:     {p.get('value_score', 0)}\n")
        f.write(f"  Confidence:      {p.get('pricing_confidence', 'n/a')}\n")
        f.write(f"  Notes:           {p.get('notes', '')}\n\n")

print("Written to tests/cost_results_explicit.txt")

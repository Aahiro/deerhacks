"""
Integration Test — Scout → Vibe Matcher pipeline
Verifies that Scout output flows correctly into Vibe Matcher.

Run:  python -m pytest tests/test_scout_vibe_integration.py -v -s
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.agents.scout import scout_node
from app.agents.vibe_matcher import vibe_matcher_node


@pytest.mark.integration
class TestScoutVibeIntegration:
    """End-to-end: Scout discovers venues → Vibe Matcher scores them (live APIs)."""

    @pytest.mark.integration
    def test_scout_output_feeds_vibe_matcher(self):
        """
        1. Scout finds candidates from a real query.
        2. Those candidates are passed directly to Vibe Matcher.
        3. Vibe Matcher scores every candidate Scout found.
        """
        # Step 1: Run Scout
        state = {
            "raw_prompt": "cozy café in downtown Toronto",
            "parsed_intent": {
                "activity": "cozy café",
                "location": "downtown Toronto",
                "vibe": "cozy",
            },
            "candidate_venues": [],
            "vibe_scores": {},
        }

        state.update(scout_node(state))

        candidates = state.get("candidate_venues", [])
        assert len(candidates) > 0, "Scout must find at least 1 venue for this test"

        print(f"\n📍 Scout found {len(candidates)} candidates")
        for v in candidates:
            print(f"   [{v['source']}] {v['name']}")

        # Step 2: Run Vibe Matcher on Scout's output (same state object)
        state.update(vibe_matcher_node(state))

        scores = state.get("vibe_scores", {})

        # Step 3: Verify every Scout candidate got a vibe score
        for venue in candidates:
            vid = venue["venue_id"]
            assert vid in scores, f"Vibe Matcher missed candidate {vid} ({venue['name']})"

        print(f"\n🎨 Vibe Matcher scored {len(scores)} venues:")
        for vid, s in scores.items():
            name = next((v["name"] for v in candidates if v["venue_id"] == vid), vid)
            # vibe_matcher now uses vibe_score / primary_style keys (upstream rewrite)
            print(f"   {name}: score={s.get('vibe_score')}, style={s.get('primary_style')}")

        # Verify the state shape is correct for downstream nodes
        assert "candidate_venues" in state, "State must still contain candidate_venues"
        assert "vibe_scores" in state, "State must contain vibe_scores"
        # Note: vibe_matcher may filter low-scoring venues but keeps at least 3
        assert len(state["candidate_venues"]) >= min(3, len(candidates)), "Vibe matcher should keep at least 3 venues"

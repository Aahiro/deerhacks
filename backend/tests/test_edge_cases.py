"""
Comprehensive edge case tests for the PATHFINDER backend.
All external API calls are mocked — no keys required.

Run: python -m pytest tests/test_edge_cases.py -v
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

# ─────────────────────────────────────────────────────────────────────────────
# COMMANDER
# ─────────────────────────────────────────────────────────────────────────────

class TestCommanderEdgeCases:

    @patch("app.agents.commander.generate_content", new_callable=AsyncMock)
    def test_gemini_returns_malformed_json_uses_fallback(self, mock_gen):
        from app.agents.commander import commander_node
        mock_gen.return_value = "this is not json at all"
        state = {"raw_prompt": "budget basketball court"}
        result = commander_node(state)
        # Fallback should still return valid structure
        assert "parsed_intent" in result
        assert "active_agents" in result
        assert "agent_weights" in result
        assert "scout" in result["active_agents"]

    @patch("app.agents.commander.generate_content", new_callable=AsyncMock)
    def test_empty_prompt_does_not_crash(self, mock_gen):
        from app.agents.commander import commander_node
        mock_gen.side_effect = Exception("no prompt")
        result = commander_node({"raw_prompt": ""})
        assert "active_agents" in result
        assert "scout" in result["active_agents"]

    @patch("app.agents.commander.generate_content", new_callable=AsyncMock)
    def test_retry_count_increments_on_veto(self, mock_gen):
        from app.agents.commander import commander_node
        mock_gen.side_effect = Exception("fail")
        # Simulate a retry run where veto was True coming in
        state = {"raw_prompt": "cafes", "veto": True, "retry_count": 0}
        result = commander_node(state)
        assert result["retry_count"] == 1
        assert result["veto"] is False  # must be cleared

    @patch("app.agents.commander.generate_content", new_callable=AsyncMock)
    def test_retry_count_does_not_increment_on_first_run(self, mock_gen):
        from app.agents.commander import commander_node
        mock_gen.side_effect = Exception("fail")
        state = {"raw_prompt": "cafes", "veto": False, "retry_count": 0}
        result = commander_node(state)
        assert result["retry_count"] == 0  # no veto incoming, stays at 0

    @patch("app.agents.commander.generate_content", new_callable=AsyncMock)
    def test_user_profile_weights_applied(self, mock_gen):
        from app.agents.commander import commander_node
        mock_gen.return_value = json.dumps({
            "parsed_intent": {"activity": "cafe", "group_size": 2, "budget": "low", "location": "toronto", "vibe": "cozy"},
            "complexity_tier": "tier_2",
            "active_agents": ["scout", "cost_analyst", "vibe_matcher"],
            "agent_weights": {"scout": 1.0, "cost_analyst": 0.5, "vibe_matcher": 0.4},
        })
        state = {
            "raw_prompt": "cafe",
            "veto": False,
            "retry_count": 0,
            "user_profile": {
                "app_metadata": {
                    "preferences": {"budget_sensitive": True}
                }
            }
        }
        result = commander_node(state)
        # cost_analyst weight should be boosted by 0.2
        assert result["agent_weights"]["cost_analyst"] > 0.5

    @patch("app.agents.commander.generate_content", new_callable=AsyncMock)
    def test_veto_is_cleared_on_commander_run(self, mock_gen):
        from app.agents.commander import commander_node
        mock_gen.side_effect = Exception("fail")
        state = {"raw_prompt": "anything", "veto": True, "retry_count": 1}
        result = commander_node(state)
        assert result["veto"] is False


# ─────────────────────────────────────────────────────────────────────────────
# SCOUT
# ─────────────────────────────────────────────────────────────────────────────

class TestScoutEdgeCases:

    @patch("app.agents.scout.search_places", new_callable=AsyncMock)
    @patch("app.agents.scout.search_yelp", new_callable=AsyncMock)
    def test_google_fails_yelp_succeeds(self, mock_yelp, mock_google):
        from app.agents.scout import scout_node
        mock_google.side_effect = Exception("Google down")
        mock_yelp.return_value = [
            {"venue_id": "yelp_1", "name": "Cafe A", "lat": 43.65, "lng": -79.38, "rating": 4.5, "review_count": 100, "photos": [], "category": "cafe", "website": "", "source": "yelp"},
        ]
        result = scout_node({"raw_prompt": "cafe", "parsed_intent": {"activity": "cafe", "location": "Toronto"}})
        # Should still return Yelp results even when Google fails
        assert len(result["candidate_venues"]) == 1
        assert result["candidate_venues"][0]["source"] == "yelp"

    @patch("app.agents.scout.search_places", new_callable=AsyncMock)
    @patch("app.agents.scout.search_yelp", new_callable=AsyncMock)
    def test_both_apis_fail_returns_empty(self, mock_yelp, mock_google):
        from app.agents.scout import scout_node
        mock_google.side_effect = Exception("down")
        mock_yelp.side_effect = Exception("down")
        result = scout_node({"raw_prompt": "cafe", "parsed_intent": {}})
        assert result["candidate_venues"] == []

    @patch("app.agents.scout.search_places", new_callable=AsyncMock)
    @patch("app.agents.scout.search_yelp", new_callable=AsyncMock)
    def test_dedup_merges_same_venue_from_both_sources(self, mock_yelp, mock_google):
        from app.agents.scout import scout_node
        venue = {"venue_id": "gp_1", "name": "Cool Spot", "lat": 43.65, "lng": -79.38, "rating": 4.2, "review_count": 50, "photos": [], "category": "cafe", "website": "", "source": "google_places"}
        venue_yelp = {**venue, "venue_id": "yelp_1", "rating": 4.5, "source": "yelp"}
        mock_google.return_value = [venue]
        mock_yelp.return_value = [venue_yelp]
        result = scout_node({"raw_prompt": "cafe", "parsed_intent": {"activity": "cafe", "location": "Toronto"}})
        # Should deduplicate to 1 result, keeping higher rating (4.5)
        assert len(result["candidate_venues"]) == 1
        assert result["candidate_venues"][0]["rating"] == 4.5

    @patch("app.agents.scout.search_places", new_callable=AsyncMock)
    @patch("app.agents.scout.search_yelp", new_callable=AsyncMock)
    def test_results_capped_at_10(self, mock_yelp, mock_google):
        from app.agents.scout import scout_node
        venues = [
            {"venue_id": f"gp_{i}", "name": f"Place {i}", "lat": 43.65 + i * 0.01, "lng": -79.38, "rating": 4.0, "review_count": 10, "photos": [], "category": "cafe", "website": "", "source": "google_places"}
            for i in range(8)
        ]
        mock_google.return_value = venues
        mock_yelp.return_value = venues  # 8 more, all unique names
        result = scout_node({"raw_prompt": "cafe", "parsed_intent": {"activity": "cafe", "location": "Toronto"}})
        assert len(result["candidate_venues"]) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# VIBE MATCHER
# ─────────────────────────────────────────────────────────────────────────────

class TestVibeMatcherEdgeCases:

    def test_no_candidates_returns_empty(self):
        from app.agents.vibe_matcher import vibe_matcher_node
        result = vibe_matcher_node({"candidate_venues": [], "parsed_intent": {}})
        assert result["vibe_scores"] == {}

    @patch("app.agents.vibe_matcher.generate_content", new_callable=AsyncMock)
    def test_gemini_returns_none_uses_fallback(self, mock_gen):
        from app.agents.vibe_matcher import vibe_matcher_node
        mock_gen.return_value = None
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Test", "address": "123 St", "category": "cafe", "photos": []}],
            "parsed_intent": {"vibe": "cozy"},
        }
        result = vibe_matcher_node(state)
        # Should not crash, should return fallback score
        assert "v1" in result["vibe_scores"]
        assert result["vibe_scores"]["v1"]["score"] is None

    @patch("app.agents.vibe_matcher.generate_content", new_callable=AsyncMock)
    def test_gemini_returns_malformed_json_uses_fallback(self, mock_gen):
        from app.agents.vibe_matcher import vibe_matcher_node
        mock_gen.return_value = "not json {"
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Test", "address": "123 St", "category": "cafe", "photos": []}],
            "parsed_intent": {},
        }
        result = vibe_matcher_node(state)
        assert "v1" in result["vibe_scores"]
        assert result["vibe_scores"]["v1"]["confidence"] == 0.0

    @patch("app.agents.vibe_matcher.generate_content", new_callable=AsyncMock)
    def test_valid_vibe_score_stored_correctly(self, mock_gen):
        from app.agents.vibe_matcher import vibe_matcher_node
        mock_gen.return_value = json.dumps({"score": 0.87, "style": "cozy", "descriptors": ["warm", "wooden"], "confidence": 0.9})
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Princess Cafe", "address": "1 Main St", "category": "cafe", "photos": []}],
            "parsed_intent": {"vibe": "cozy"},
        }
        result = vibe_matcher_node(state)
        assert result["vibe_scores"]["v1"]["score"] == 0.87
        assert result["vibe_scores"]["v1"]["style"] == "cozy"


# ─────────────────────────────────────────────────────────────────────────────
# COST ANALYST
# ─────────────────────────────────────────────────────────────────────────────

class TestCostAnalystEdgeCases:

    def test_no_candidates_returns_empty(self):
        from app.agents.cost_analyst import cost_analyst_node
        result = cost_analyst_node({"candidate_venues": [], "parsed_intent": {}})
        assert result["cost_profiles"] == {}

    def test_venue_with_no_website_returns_fallback(self):
        from app.agents.cost_analyst import cost_analyst_node
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Mystery Place", "category": "venue", "website": ""}],
            "parsed_intent": {"group_size": 4},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["pricing_confidence"] == "unknown"
        assert result["cost_profiles"]["v1"]["value_score"] == 0.3

    @patch("app.agents.cost_analyst._firecrawl_post", new_callable=AsyncMock)
    @patch("app.agents.cost_analyst.generate_content", new_callable=AsyncMock)
    def test_confirmed_pricing_keeps_gemini_score(self, mock_gen, mock_fc):
        from app.agents.cost_analyst import cost_analyst_node
        mock_fc.return_value = {"links": [], "data": {"markdown": "Bowling: $10/person"}}
        mock_gen.return_value = json.dumps({
            "base_cost": 40.0, "hidden_costs": [], "total_cost_of_attendance": 40.0,
            "per_person": 10.0, "value_score": 0.8, "pricing_confidence": "confirmed",
            "notes": "Confirmed $10/person from website."
        })
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Bowl-O", "category": "bowling", "website": "https://bowl.com"}],
            "parsed_intent": {"group_size": 4},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["pricing_confidence"] == "confirmed"
        assert result["cost_profiles"]["v1"]["value_score"] == 0.8

    @patch("app.agents.cost_analyst._firecrawl_post", new_callable=AsyncMock)
    @patch("app.agents.cost_analyst.generate_content", new_callable=AsyncMock)
    def test_estimated_pricing_caps_value_score_at_0_5(self, mock_gen, mock_fc):
        from app.agents.cost_analyst import cost_analyst_node
        mock_fc.return_value = {"links": [], "data": {"markdown": "Some content"}}
        mock_gen.return_value = json.dumps({
            "base_cost": 80.0, "hidden_costs": [], "total_cost_of_attendance": 80.0,
            "per_person": 20.0, "value_score": 0.9, "pricing_confidence": "estimated",
            "notes": "Estimated from market rates."
        })
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Venue X", "category": "bar", "website": "https://x.com"}],
            "parsed_intent": {"group_size": 4},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["value_score"] <= 0.5

    @patch("app.agents.cost_analyst._firecrawl_post", new_callable=AsyncMock)
    def test_firecrawl_returns_none_uses_fallback(self, mock_fc):
        from app.agents.cost_analyst import cost_analyst_node
        mock_fc.return_value = None  # simulates 429 exhausted or network failure
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Place", "category": "cafe", "website": "https://place.com"}],
            "parsed_intent": {"group_size": 2},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["pricing_confidence"] == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# ACCESS ANALYST
# ─────────────────────────────────────────────────────────────────────────────

class TestAccessAnalystEdgeCases:

    def test_no_candidates_returns_empty(self):
        from app.agents.access_analyst import access_analyst_node
        result = access_analyst_node({"candidate_venues": [], "parsed_intent": {}, "member_locations": []})
        assert result["accessibility_scores"] == {}
        assert result["isochrones"] == {}

    @patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
    def test_close_venue_gets_high_score(self, mock_dm, mock_iso):
        from app.agents.access_analyst import access_analyst_node
        # 5 minutes → score should be 1.0
        mock_dm.return_value = [{"duration_sec": 300, "distance_m": 800, "status": "OK"}]
        mock_iso.return_value = None
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Nearby Cafe", "lat": 43.65, "lng": -79.38}],
            "parsed_intent": {},
            "member_locations": [],
            "raw_prompt": "cafe",
        }
        result = access_analyst_node(state)
        score = result["accessibility_scores"]["v1"]["score"]
        assert score == 1.0, f"Expected 1.0 for 5-min venue, got {score}"

    @patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
    def test_far_venue_gets_low_score(self, mock_dm, mock_iso):
        from app.agents.access_analyst import access_analyst_node
        # 90 minutes → score should be 0.1
        mock_dm.return_value = [{"duration_sec": 5400, "distance_m": 40000, "status": "OK"}]
        mock_iso.return_value = None
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Far Away Venue", "lat": 44.0, "lng": -80.0}],
            "parsed_intent": {},
            "member_locations": [],
            "raw_prompt": "venue",
        }
        result = access_analyst_node(state)
        score = result["accessibility_scores"]["v1"]["score"]
        assert score == 0.1, f"Expected 0.1 for 90-min venue, got {score}"

    @patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
    def test_isochrone_stored_when_api_returns_geojson(self, mock_dm, mock_iso):
        from app.agents.access_analyst import access_analyst_node
        fake_geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon"}}]}
        mock_dm.return_value = [{"duration_sec": 600, "distance_m": 2000, "status": "OK"}]
        mock_iso.return_value = fake_geojson
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Test Venue", "lat": 43.65, "lng": -79.38}],
            "parsed_intent": {},
            "member_locations": [],
            "raw_prompt": "venue",
        }
        result = access_analyst_node(state)
        assert "v1" in result["isochrones"]
        assert result["isochrones"]["v1"]["type"] == "FeatureCollection"

    @patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
    def test_api_returns_no_duration_uses_neutral_fallback(self, mock_dm, mock_iso):
        from app.agents.access_analyst import access_analyst_node
        # No token → returns empty list → duration_sec = None → score = 0.5
        mock_dm.return_value = []
        mock_iso.return_value = None
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Unknown", "lat": 43.65, "lng": -79.38}],
            "parsed_intent": {},
            "member_locations": [],
            "raw_prompt": "venue",
        }
        result = access_analyst_node(state)
        assert result["accessibility_scores"]["v1"]["score"] == 0.5

    @patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
    def test_member_locations_centroid_used_as_origin(self, mock_dm, mock_iso):
        from app.agents.access_analyst import _resolve_origin
        # Two members: centroid should be the average lat/lng
        state = {
            "parsed_intent": {},
            "member_locations": [
                {"lat": 43.60, "lng": -79.40},
                {"lat": 43.70, "lng": -79.30},
            ],
        }
        origin = _resolve_origin(state)
        assert abs(origin[0] - 43.65) < 0.001
        assert abs(origin[1] - (-79.35)) < 0.001

    @patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
    def test_multiple_venues_all_scored(self, mock_dm, mock_iso):
        from app.agents.access_analyst import access_analyst_node
        # Each venue gets its own distance matrix call (one destination per call)
        mock_dm.return_value = [{"duration_sec": 600, "distance_m": 2000, "status": "OK"}]
        mock_iso.return_value = None
        venues = [
            {"venue_id": "v1", "name": "Venue 1", "lat": 43.65, "lng": -79.38},
            {"venue_id": "v2", "name": "Venue 2", "lat": 43.66, "lng": -79.39},
            {"venue_id": "v3", "name": "Venue 3", "lat": 43.67, "lng": -79.40},
        ]
        state = {
            "candidate_venues": venues,
            "parsed_intent": {},
            "member_locations": [],
            "raw_prompt": "venue",
        }
        result = access_analyst_node(state)
        assert len(result["accessibility_scores"]) == 3
        for vid in ["v1", "v2", "v3"]:
            assert vid in result["accessibility_scores"]
            assert result["accessibility_scores"][vid]["score"] != 0.5  # real score, not fallback

    def test_walking_mode_resolved_from_prompt(self):
        from app.agents.access_analyst import _resolve_travel_mode
        state = {"raw_prompt": "coffee shop I can walk to", "parsed_intent": {}}
        assert _resolve_travel_mode(state) == "walking"

    def test_transit_mode_resolved_from_prompt(self):
        from app.agents.access_analyst import _resolve_travel_mode
        state = {"raw_prompt": "take the TTC to a bar", "parsed_intent": {}}
        assert _resolve_travel_mode(state) == "transit"

    def test_driving_is_default_mode(self):
        from app.agents.access_analyst import _resolve_travel_mode
        state = {"raw_prompt": "escape room downtown", "parsed_intent": {}}
        assert _resolve_travel_mode(state) == "driving"


# ─────────────────────────────────────────────────────────────────────────────
# CRITIC
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticEdgeCases:

    def test_no_candidates_returns_no_veto(self):
        from app.agents.critic import critic_node
        result = critic_node({"candidate_venues": [], "parsed_intent": {}})
        assert result["veto"] is False
        assert result["risk_flags"] == {}

    @patch("app.agents.critic.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.get_weather", new_callable=AsyncMock)
    @patch("app.agents.critic.get_events", new_callable=AsyncMock)
    def test_veto_only_triggers_on_first_candidate(self, mock_events, mock_weather, mock_gen):
        from app.agents.critic import critic_node
        mock_weather.return_value = {"condition": "Rain", "description": "heavy rain", "temp_c": 5, "feels_like_c": 2}
        mock_events.return_value = []
        # First candidate gets vetoed, second does not
        mock_gen.side_effect = [
            json.dumps({"risks": [{"type": "weather", "severity": "high", "detail": "heavy rain"}], "veto": True, "veto_reason": "Rain"}),
            json.dumps({"risks": [], "veto": False, "veto_reason": None}),
            json.dumps({"risks": [], "veto": False, "veto_reason": None}),
        ]
        venues = [
            {"venue_id": "v1", "name": "Outdoor Park", "category": "park", "lat": 43.65, "lng": -79.38},
            {"venue_id": "v2", "name": "Indoor Gym", "category": "gym", "lat": 43.66, "lng": -79.39},
        ]
        result = critic_node({"candidate_venues": venues, "parsed_intent": {"activity": "outdoor"}})
        assert result["veto"] is True  # v1 is #1 candidate and got vetoed
        assert result["veto_reason"] == "Rain"

    @patch("app.agents.critic.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.get_weather", new_callable=AsyncMock)
    @patch("app.agents.critic.get_events", new_callable=AsyncMock)
    def test_no_dealbreaker_returns_no_veto(self, mock_events, mock_weather, mock_gen):
        from app.agents.critic import critic_node
        mock_weather.return_value = {"condition": "Clear", "description": "sunny", "temp_c": 22, "feels_like_c": 21}
        mock_events.return_value = []
        mock_gen.return_value = json.dumps({"risks": [], "veto": False, "veto_reason": None})
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Great Spot", "category": "cafe", "lat": 43.65, "lng": -79.38}],
            "parsed_intent": {},
        }
        result = critic_node(state)
        assert result["veto"] is False

    @patch("app.agents.critic.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.get_weather", new_callable=AsyncMock)
    @patch("app.agents.critic.get_events", new_callable=AsyncMock)
    def test_gemini_failure_does_not_crash(self, mock_events, mock_weather, mock_gen):
        from app.agents.critic import critic_node
        mock_weather.return_value = None
        mock_events.return_value = []
        mock_gen.side_effect = Exception("Gemini down")
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Place", "category": "cafe", "lat": 43.65, "lng": -79.38}],
            "parsed_intent": {},
        }
        result = critic_node(state)
        assert result["veto"] is False  # graceful fallback
        assert "v1" in result["risk_flags"]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESISER
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesiserEdgeCases:

    def test_no_candidates_returns_empty(self):
        from app.agents.synthesiser import synthesiser_node
        result = synthesiser_node({"candidate_venues": []})
        assert result["ranked_results"] == []

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    def test_top_3_only_get_explanations(self, mock_gen):
        from app.agents.synthesiser import synthesiser_node
        mock_gen.return_value = json.dumps({"why": "Great fit", "watch_out": ""})
        venues = [
            {"venue_id": f"v{i}", "name": f"Venue {i}", "address": f"{i} St", "lat": 43.65, "lng": -79.38, "category": "cafe"}
            for i in range(6)
        ]
        result = synthesiser_node({"candidate_venues": venues, "vibe_scores": {}, "accessibility_scores": {}, "cost_profiles": {}, "risk_flags": {}, "agent_weights": {}, "raw_prompt": "cafe", "isochrones": {}})
        assert len(result["ranked_results"]) == 3
        assert mock_gen.call_count == 3

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    def test_isochrone_forwarded_to_ranked_results(self, mock_gen):
        from app.agents.synthesiser import synthesiser_node
        mock_gen.return_value = json.dumps({"why": "Good", "watch_out": ""})
        fake_geojson = {"type": "FeatureCollection", "features": []}
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Test", "address": "1 St", "lat": 43.65, "lng": -79.38, "category": "cafe"}],
            "vibe_scores": {}, "accessibility_scores": {}, "cost_profiles": {},
            "risk_flags": {}, "agent_weights": {}, "raw_prompt": "cafe",
            "isochrones": {"v1": fake_geojson},
        }
        result = synthesiser_node(state)
        assert result["ranked_results"][0]["isochrone_geojson"] == fake_geojson

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    def test_high_risk_penalty_lowers_composite_score(self, mock_gen):
        from app.agents.synthesiser import _compute_composite_score
        # High-severity risks should reduce ranking
        score_clean = _compute_composite_score("v1", {}, {}, {}, {}, {})
        score_risky = _compute_composite_score(
            "v1", {}, {}, {},
            {"v1": [{"type": "weather", "severity": "high"}, {"type": "event", "severity": "high"}]},
            {}
        )
        assert score_risky < score_clean

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    def test_missing_agent_data_uses_defaults_not_crash(self, mock_gen):
        from app.agents.synthesiser import synthesiser_node
        mock_gen.return_value = json.dumps({"why": "Good match", "watch_out": ""})
        # No vibe/cost/access data at all
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Bare Venue", "address": "1 St", "lat": 43.65, "lng": -79.38, "category": "cafe"}],
            "vibe_scores": {}, "accessibility_scores": {}, "cost_profiles": {},
            "risk_flags": {}, "agent_weights": {}, "raw_prompt": "cafe", "isochrones": {},
        }
        result = synthesiser_node(state)
        assert len(result["ranked_results"]) == 1
        assert result["ranked_results"][0]["name"] == "Bare Venue"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH — RETRY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphRetryLogic:

    def test_no_veto_goes_to_synthesiser(self):
        from app.graph import _should_retry
        state = {"veto": False, "retry_count": 0}
        assert _should_retry(state) == "synthesiser"

    def test_first_veto_goes_to_commander(self):
        from app.graph import _should_retry
        state = {"veto": True, "retry_count": 0}
        assert _should_retry(state) == "commander"

    def test_second_veto_goes_to_synthesiser_not_loop(self):
        from app.graph import _should_retry
        state = {"veto": True, "retry_count": 1}
        assert _should_retry(state) == "synthesiser"

    def test_inactive_agent_is_skipped(self):
        from app.graph import _make_conditional_agent
        called = []
        def fake_agent(state):
            called.append(True)
            return state
        wrapped = _make_conditional_agent("vibe_matcher", fake_agent)
        # vibe_matcher not in active_agents → should skip
        wrapped({"active_agents": ["scout", "cost_analyst"]})
        assert len(called) == 0

    def test_active_agent_is_called(self):
        from app.graph import _make_conditional_agent
        called = []
        def fake_agent(state):
            called.append(True)
            return state
        wrapped = _make_conditional_agent("vibe_matcher", fake_agent)
        wrapped({"active_agents": ["scout", "vibe_matcher"]})
        assert len(called) == 1


# ─────────────────────────────────────────────────────────────────────────────
# AUTH0 DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestAuth0Dependency:

    def test_no_token_returns_empty_dict(self):
        from app.dependencies import get_optional_user
        result = asyncio.run(get_optional_user(credentials=None))
        assert result == {}

    @patch("app.dependencies.verify_jwt", new_callable=AsyncMock)
    def test_valid_token_returns_claims(self, mock_verify):
        from app.dependencies import get_optional_user
        from fastapi.security import HTTPAuthorizationCredentials
        mock_verify.return_value = {"sub": "auth0|abc123", "email": "user@test.com"}
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="fake.jwt.token")
        result = asyncio.run(get_optional_user(credentials=creds))
        assert result["sub"] == "auth0|abc123"

    @patch("app.dependencies.verify_jwt", new_callable=AsyncMock)
    def test_invalid_token_propagates_401(self, mock_verify):
        from app.dependencies import get_optional_user
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad.token")
        with pytest.raises(HTTPException) as exc:
            asyncio.run(get_optional_user(credentials=creds))
        assert exc.value.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# FULL MOCKED PIPELINE — All Agents End-to-End
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineAllAgents:
    """
    Mocked end-to-end test: Commander → Scout → Vibe → Access → Cost → Critic → Synthesiser.
    All external APIs mocked. Confirms every agent runs and contributes output
    to ranked_results with correct data shapes.
    """

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.get_weather", new_callable=AsyncMock)
    @patch("app.agents.critic.get_events", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
    @patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
    @patch("app.agents.cost_analyst._firecrawl_post", new_callable=AsyncMock)
    @patch("app.agents.cost_analyst.generate_content", new_callable=AsyncMock)
    @patch("app.agents.vibe_matcher.generate_content", new_callable=AsyncMock)
    @patch("app.agents.scout.search_yelp", new_callable=AsyncMock)
    @patch("app.agents.scout.search_places", new_callable=AsyncMock)
    @patch("app.agents.commander.generate_content", new_callable=AsyncMock)
    def test_all_agents_run_and_produce_ranked_results(
        self,
        mock_cmd_gen,
        mock_places,
        mock_yelp,
        mock_vibe_gen,
        mock_cost_gen,
        mock_fc,
        mock_iso,
        mock_dm,
        mock_events,
        mock_weather,
        mock_critic_gen,
        mock_synth_gen,
    ):
        from app.graph import pathfinder_graph
        from app.models.state import PathfinderState

        # ── Commander: returns a plan that activates ALL agents ──
        mock_cmd_gen.return_value = json.dumps({
            "parsed_intent": {
                "activity": "escape room",
                "location": "downtown Toronto",
                "group_size": 4,
                "budget": "moderate",
                "vibe": "thrilling",
                "time": "Saturday 2pm",
            },
            "complexity_tier": "tier_3",
            "active_agents": [
                "scout", "vibe_matcher", "access_analyst", "cost_analyst", "critic"
            ],
            "agent_weights": {
                "scout": 1.0,
                "vibe_matcher": 0.8,
                "access_analyst": 0.7,
                "cost_analyst": 0.9,
                "critic": 0.6,
            },
        })

        # ── Scout: returns 2 real-looking venues ──
        mock_places.return_value = [
            {
                "venue_id": "gp_escape1",
                "name": "Trapped! Escape Rooms",
                "address": "90 Eglinton Ave E, Toronto",
                "lat": 43.7079, "lng": -79.3961,
                "rating": 4.7, "review_count": 320,
                "photos": ["https://maps.googleapis.com/photo?ref=abc"],
                "category": "escape room",
                "website": "https://trapped.ca",
                "source": "google_places",
            },
        ]
        mock_yelp.return_value = [
            {
                "venue_id": "yelp_escape2",
                "name": "Escape Manor Toronto",
                "address": "55 Mercer St, Toronto",
                "lat": 43.6441, "lng": -79.3980,
                "rating": 4.5, "review_count": 210,
                "photos": [],
                "category": "escape room",
                "website": "https://escapemanor.com",
                "source": "yelp",
            },
        ]

        # ── Vibe Matcher: both venues score well ──
        mock_vibe_gen.side_effect = [
            json.dumps({"score": 0.88, "style": "thrilling", "descriptors": ["intense", "immersive"], "confidence": 0.9}),
            json.dumps({"score": 0.75, "style": "spooky", "descriptors": ["dramatic", "dark"], "confidence": 0.8}),
        ]

        # ── Access Analyst: short drive to both ──
        # get_distance_matrix returns list of dicts, one per destination
        mock_dm.return_value = [{"duration_sec": 720, "distance_m": 2100, "status": "OK"}]
        mock_iso.return_value = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}}],
        }

        # ── Cost Analyst: website scraped and priced ──
        mock_fc.return_value = {"links": [], "data": {"markdown": "Escape room: $28/person"}}
        mock_cost_gen.return_value = json.dumps({
            "base_cost": 112.0, "hidden_costs": ["tax ~13%"],
            "total_cost_of_attendance": 126.56, "per_person": 31.64,
            "value_score": 0.75, "pricing_confidence": "confirmed",
            "notes": "Confirmed $28/person from website."
        })

        # ── Critic: no veto ──
        mock_weather.return_value = {"condition": "Clear", "description": "sunny", "temp_c": 18, "feels_like_c": 17}
        mock_events.return_value = []
        mock_critic_gen.return_value = json.dumps({"risks": [], "veto": False, "veto_reason": None})

        # ── Synthesiser: generate explanations ──
        mock_synth_gen.return_value = json.dumps({
            "why": "Perfect thrilling escape room for your group, reasonably priced and easy to reach.",
            "watch_out": "Book in advance — popular on Saturdays.",
        })

        # ── Run the full graph ──
        state = PathfinderState(
            raw_prompt="I need a budget-friendly escape room for 4 people in downtown Toronto this Saturday at 2 PM.",
        )
        final_state = pathfinder_graph.invoke(state)

        # ── Assertions: complete pipeline ──
        ranked = final_state.get("ranked_results", [])
        assert len(ranked) > 0, "Pipeline produced no ranked results"
        assert len(ranked) <= 3, "Synthesiser should cap at top 3"

        top = ranked[0]
        assert "name" in top, "Ranked result missing name"
        assert "rank" in top, "Missing rank field"
        assert top["rank"] == 1, "Top result should be rank 1"
        assert "why" in top, "Missing Gemini explanation"
        assert "isochrone_geojson" in top, "Isochrone not forwarded to ranked_results"
        assert top["isochrone_geojson"] is not None, "Isochrone should not be None"
        assert top["isochrone_geojson"]["type"] == "FeatureCollection", "Isochrone should be GeoJSON"

        # ── Check Access Analyst score flowed to ranked result ──
        assert "accessibility_score" in top, "Missing accessibility_score in ranked result"
        assert top["accessibility_score"] > 0.5, \
            f"Expected real access score (>0.5), got {top['accessibility_score']}"

        # ── Check Vibe Matcher score flowed to ranked result ──
        assert "vibe_score" in top, "Missing vibe_score in ranked result"
        assert top["vibe_score"] is not None, "vibe_score should not be None"

        # ── Check Cost Analyst profile flowed to ranked result ──
        assert "cost_profile" in top, "Missing cost_profile in ranked result"
        assert top["cost_profile"]["pricing_confidence"] == "confirmed"

        # ── Verify each agent's raw output in final state ──
        vibe = final_state.get("vibe_scores", {})
        assert len(vibe) > 0, "Vibe Matcher produced no scores"

        access = final_state.get("accessibility_scores", {})
        assert len(access) > 0, "Access Analyst produced no scores"
        # Scores should reflect the 12-min drive (720s → ~0.96), not fallback 0.5
        for vid, acc in access.items():
            assert acc["score"] != 0.5, \
                f"Venue {vid} got neutral fallback — mock may not have been applied"

        cost = final_state.get("cost_profiles", {})
        assert len(cost) > 0, "Cost Analyst produced no profiles"

        isochrones = final_state.get("isochrones", {})
        assert len(isochrones) > 0, "No isochrones in final state"

        assert final_state.get("veto") is False, "Critic should not have vetoed"

        print(f"\n[OK] Full pipeline produced {len(ranked)} ranked result(s):")
        for r in ranked:
            print(f"   #{r['rank']}: {r['name']}")
            print(f"        Vibe={r.get('vibe_score')}, Access={r.get('accessibility_score')}")
            print(f"        Why: {r.get('why', '')[:80]}")
            print(f"        Isochrone: {'yes' if r.get('isochrone_geojson') else 'no'}")

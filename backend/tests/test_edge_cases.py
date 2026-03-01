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

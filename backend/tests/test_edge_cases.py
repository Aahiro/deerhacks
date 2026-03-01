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
            "requires_oauth": False,
            "oauth_scopes": [],
            "allowed_actions": [],
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
        # cost_analyst weight should be boosted by 0.2 (from 0.5 to 0.7)
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
        # Should not crash, should return fallback score (vibe_score key, not score)
        assert "v1" in result["vibe_scores"]
        assert result["vibe_scores"]["v1"]["vibe_score"] is None

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
        # Prompt instructs Gemini to use vibe_score / primary_style / visual_descriptors
        mock_gen.return_value = json.dumps({
            "vibe_score": 0.87, "primary_style": "cozy",
            "visual_descriptors": ["warm", "wooden"], "confidence": 0.9
        })
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Princess Cafe", "address": "1 Main St", "category": "cafe", "photos": []}],
            "parsed_intent": {"vibe": "cozy"},
        }
        result = vibe_matcher_node(state)
        assert result["vibe_scores"]["v1"]["vibe_score"] == 0.87
        assert result["vibe_scores"]["v1"]["primary_style"] == "cozy"


# ─────────────────────────────────────────────────────────────────────────────
# COST ANALYST  (heuristic-only rewrite — no Firecrawl or Gemini)
# ─────────────────────────────────────────────────────────────────────────────

class TestCostAnalystEdgeCases:

    def test_no_candidates_returns_empty(self):
        from app.agents.cost_analyst import cost_analyst_node
        result = cost_analyst_node({"candidate_venues": [], "parsed_intent": {}})
        assert result["cost_profiles"] == {}

    def test_venue_with_no_price_data_returns_none_range(self):
        """Venue with no google_price, yelp_price, or price_range → confidence 'none'."""
        from app.agents.cost_analyst import cost_analyst_node
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Mystery Place", "category": "venue", "source": "google_places"}],
            "parsed_intent": {},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["confidence"] == "none"
        assert result["cost_profiles"]["v1"]["price_range"] is None
        assert result["cost_profiles"]["v1"]["value_score"] == 0.3  # fallback

    def test_single_source_google_price_medium_confidence(self):
        """Google price only → confidence 'medium'."""
        from app.agents.cost_analyst import cost_analyst_node
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Bowl-O", "category": "bowling", "source": "google_places", "price_range": "$$"}],
            "parsed_intent": {},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["price_range"] == "$$"
        assert result["cost_profiles"]["v1"]["confidence"] == "medium"

    def test_single_source_yelp_price_medium_confidence(self):
        """Yelp price only → confidence 'medium'."""
        from app.agents.cost_analyst import cost_analyst_node
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Bar X", "source": "yelp", "price_range": "$$$"}],
            "parsed_intent": {},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["price_range"] == "$$$"
        assert result["cost_profiles"]["v1"]["confidence"] == "medium"

    def test_matching_google_and_yelp_price_high_confidence(self):
        """Both sources agree → confidence 'high'."""
        from app.agents.cost_analyst import cost_analyst_node
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Place", "google_price": "$$", "yelp_price": "$$"}],
            "parsed_intent": {},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["price_range"] == "$$"
        assert result["cost_profiles"]["v1"]["confidence"] == "high"

    def test_conflicting_prices_resolves_to_median_low_confidence(self):
        """Conflicting sources ($, $$$) → median ($$) with confidence 'low'."""
        from app.agents.cost_analyst import cost_analyst_node
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Place", "google_price": "$", "yelp_price": "$$$"}],
            "parsed_intent": {},
        }
        result = cost_analyst_node(state)
        assert result["cost_profiles"]["v1"]["price_range"] == "$$"
        assert result["cost_profiles"]["v1"]["confidence"] == "low"

    def test_cheaper_price_range_yields_higher_value_score(self):
        """$ should score higher value than $$$$."""
        from app.agents.cost_analyst import _calculate_value_score
        score_cheap = _calculate_value_score("$", "medium")
        score_pricey = _calculate_value_score("$$$$", "medium")
        assert score_cheap > score_pricey

    def test_none_confidence_gives_fallback_value_score(self):
        from app.agents.cost_analyst import _calculate_value_score
        assert _calculate_value_score(None, "none") == 0.3

    def test_multiple_venues_all_profiled(self):
        from app.agents.cost_analyst import cost_analyst_node
        state = {
            "candidate_venues": [
                {"venue_id": "v1", "source": "google_places", "price_range": "$"},
                {"venue_id": "v2", "source": "yelp", "price_range": "$$"},
                {"venue_id": "v3"},  # no price data
            ],
            "parsed_intent": {},
        }
        result = cost_analyst_node(state)
        assert len(result["cost_profiles"]) == 3
        assert result["cost_profiles"]["v1"]["price_range"] == "$"
        assert result["cost_profiles"]["v2"]["price_range"] == "$$"
        assert result["cost_profiles"]["v3"]["confidence"] == "none"


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
        # Upstream critic uses fast_fail key (not veto) in Gemini output
        mock_gen.side_effect = [
            json.dumps({"risks": [{"type": "weather", "severity": "high", "detail": "heavy rain"}], "fast_fail": True, "fast_fail_reason": "Rain"}),
            json.dumps({"risks": [], "fast_fail": False, "fast_fail_reason": None}),
        ]
        venues = [
            {"venue_id": "v1", "name": "Outdoor Park", "category": "park", "lat": 43.65, "lng": -79.38},
            {"venue_id": "v2", "name": "Indoor Gym", "category": "gym", "lat": 43.66, "lng": -79.39},
        ]
        result = critic_node({"candidate_venues": venues, "parsed_intent": {"activity": "outdoor"}})
        # Critic maps fast_fail → veto in return state
        assert result["veto"] is True  # v1 is #1 candidate and got fast_fail/vetoed
        assert result["veto_reason"] == "Rain"

    @patch("app.agents.critic.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.get_weather", new_callable=AsyncMock)
    @patch("app.agents.critic.get_events", new_callable=AsyncMock)
    def test_no_dealbreaker_returns_no_veto(self, mock_events, mock_weather, mock_gen):
        from app.agents.critic import critic_node
        mock_weather.return_value = {"condition": "Clear", "description": "sunny", "temp_c": 22, "feels_like_c": 21}
        mock_events.return_value = []
        mock_gen.return_value = json.dumps({"risks": [], "fast_fail": False, "fast_fail_reason": None})
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
        result = synthesiser_node({"candidate_venues": venues, "vibe_scores": {}, "cost_profiles": {}, "risk_flags": {}, "agent_weights": {}, "raw_prompt": "cafe"})
        assert len(result["ranked_results"]) == 3
        # generate_content called once per explanation + once for global_consensus = 4 calls
        assert mock_gen.call_count >= 3

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    def test_high_risk_penalty_lowers_composite_score(self, mock_gen):
        from app.agents.synthesiser import _compute_composite_score
        # High-severity risks should reduce ranking — new signature: 5 args
        score_clean = _compute_composite_score("v1", {}, {}, {}, {})
        score_risky = _compute_composite_score(
            "v1", {}, {},
            {"v1": [{"type": "weather", "severity": "high"}, {"type": "event", "severity": "high"}]},
            {}
        )
        assert score_risky < score_clean

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    def test_missing_agent_data_uses_defaults_not_crash(self, mock_gen):
        from app.agents.synthesiser import synthesiser_node
        mock_gen.return_value = json.dumps({"why": "Good match", "watch_out": ""})
        # No vibe/cost/risk data at all
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Bare Venue", "address": "1 St", "lat": 43.65, "lng": -79.38, "category": "cafe"}],
            "vibe_scores": {}, "cost_profiles": {},
            "risk_flags": {}, "agent_weights": {}, "raw_prompt": "cafe",
        }
        result = synthesiser_node(state)
        assert len(result["ranked_results"]) == 1
        assert result["ranked_results"][0]["name"] == "Bare Venue"

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    def test_ranked_result_has_correct_fields(self, mock_gen):
        from app.agents.synthesiser import synthesiser_node
        mock_gen.return_value = json.dumps({"why": "Good", "watch_out": "Busy on weekends"})
        state = {
            "candidate_venues": [{"venue_id": "v1", "name": "Test Spot", "address": "1 Main St", "lat": 43.65, "lng": -79.38, "category": "cafe"}],
            "vibe_scores": {"v1": {"vibe_score": 0.8, "primary_style": "cozy", "confidence": 0.9}},
            "cost_profiles": {"v1": {"price_range": "$$", "confidence": "high", "value_score": 0.6}},
            "risk_flags": {},
            "agent_weights": {},
            "raw_prompt": "cozy cafe",
        }
        result = synthesiser_node(state)
        top = result["ranked_results"][0]
        assert top["rank"] == 1
        assert top["name"] == "Test Spot"
        assert "vibe_score" in top
        assert "price_range" in top
        assert "price_confidence" in top
        assert top["price_range"] == "$$"
        assert top["price_confidence"] == "high"
        assert top["why"] == "Good"


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

    def test_fast_fail_goes_to_commander_on_first_pass(self):
        from app.graph import _should_retry
        state = {"fast_fail": True, "veto": False, "retry_count": 0}
        assert _should_retry(state) == "commander"

    def test_fast_fail_goes_to_synthesiser_after_retry(self):
        from app.graph import _should_retry
        state = {"fast_fail": True, "veto": False, "retry_count": 1}
        assert _should_retry(state) == "synthesiser"

    def test_no_flags_no_veto_goes_to_synthesiser(self):
        from app.graph import _should_retry
        state = {}
        assert _should_retry(state) == "synthesiser"


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
# FULL MOCKED PIPELINE — All Agents End-to-End (parallel architecture)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineAllAgents:
    """
    Mocked end-to-end test: Commander → Scout → parallel_analysts (Vibe + Cost + Critic) → Synthesiser.
    All external APIs mocked. Confirms the pipeline produces ranked_results with correct data shapes.

    NOTE: access_analyst was removed in the upstream merge; parallel_analysts_node now runs
    vibe_matcher, cost_analyst, and critic concurrently.
    """

    @patch("app.agents.synthesiser.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.generate_content", new_callable=AsyncMock)
    @patch("app.agents.critic.get_weather", new_callable=AsyncMock)
    @patch("app.agents.critic.get_events", new_callable=AsyncMock)
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
        mock_events,
        mock_weather,
        mock_critic_gen,
        mock_synth_gen,
    ):
        from app.graph import pathfinder_graph
        from app.models.state import PathfinderState

        # ── Commander: activates available agents (no access_analyst) ──
        mock_cmd_gen.return_value = json.dumps({
            "parsed_intent": {
                "activity": "escape room",
                "location": "downtown Toronto",
                "group_size": 4,
                "budget": "moderate",
                "vibe": "thrilling",
            },
            "complexity_tier": "tier_2",
            "active_agents": ["scout", "vibe_matcher", "cost_analyst", "critic"],
            "agent_weights": {
                "scout": 1.0,
                "vibe_matcher": 0.8,
                "cost_analyst": 0.9,
                "critic": 0.6,
            },
            "requires_oauth": False,
            "oauth_scopes": [],
            "allowed_actions": [],
        })

        # ── Scout: returns 2 real-looking venues with price data ──
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
                "price_range": "$$",
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
                "price_range": "$$",
            },
        ]

        # ── Vibe Matcher: both venues score well (use correct key names per prompt) ──
        mock_vibe_gen.side_effect = [
            json.dumps({"vibe_score": 0.88, "primary_style": "thrilling", "visual_descriptors": ["intense", "immersive"], "confidence": 0.9}),
            json.dumps({"vibe_score": 0.75, "primary_style": "spooky", "visual_descriptors": ["dramatic", "dark"], "confidence": 0.8}),
        ]

        # ── Critic: no veto ──
        mock_weather.return_value = {"condition": "Clear", "description": "sunny", "temp_c": 18, "feels_like_c": 17}
        mock_events.return_value = []
        mock_critic_gen.return_value = json.dumps({"risks": [], "fast_fail": False, "fast_fail_reason": None})

        # ── Synthesiser: generate explanations + global consensus ──
        mock_synth_gen.return_value = json.dumps({
            "why": "Perfect thrilling escape room for your group, reasonably priced and easy to reach.",
            "watch_out": "Book in advance — popular on Saturdays.",
            "global_consensus": "Trapped! is the top pick.",
            "email_draft": "Dear team, ...",
        })

        # ── Run the full graph ──
        # parallel_analysts_node is async, so we must use ainvoke (not sync invoke)
        state = PathfinderState(
            raw_prompt="I need an escape room for 4 people in downtown Toronto this Saturday.",
            veto=False,
            fast_fail=False,
            retry_count=0,
        )
        import nest_asyncio
        nest_asyncio.apply()
        final_state = asyncio.run(pathfinder_graph.ainvoke(state))

        # ── Assertions: pipeline produced results ──
        ranked = final_state.get("ranked_results", [])
        assert len(ranked) > 0, "Pipeline produced no ranked results"
        assert len(ranked) <= 3, "Synthesiser should cap at top 3"

        top = ranked[0]
        assert "name" in top, "Ranked result missing name"
        assert "rank" in top, "Missing rank field"
        assert top["rank"] == 1, "Top result should be rank 1"
        assert "why" in top, "Missing Gemini explanation"
        assert "vibe_score" in top, "Missing vibe_score in ranked result"
        assert "price_range" in top, "Missing price_range in ranked result"
        assert "price_confidence" in top, "Missing price_confidence in ranked result"

        # ── Verify each agent's raw output in final state ──
        vibe = final_state.get("vibe_scores", {})
        assert len(vibe) > 0, "Vibe Matcher produced no scores"

        cost = final_state.get("cost_profiles", {})
        assert len(cost) > 0, "Cost Analyst produced no profiles"

        assert final_state.get("veto") is False, "Critic should not have vetoed"
        assert final_state.get("retry_count", 0) == 0, "No retry should have occurred"

        print(f"\n[OK] Full pipeline produced {len(ranked)} ranked result(s):")
        for r in ranked:
            print(f"   #{r['rank']}: {r['name']}")
            print(f"        Vibe={r.get('vibe_score')}, Price={r.get('price_range')} ({r.get('price_confidence')})")
            print(f"        Why: {r.get('why', '')[:80]}")

"""
Test Node 2 — The SCOUT (standalone)
Calls the live Google Places + Yelp APIs to verify venue discovery works.

Run:  python -m pytest tests/test_scout.py -v -s
"""

import sys
import os

# Ensure the backend app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.agents.scout import scout_node, _deduplicate, _haversine


# ── Unit Tests (no API calls) ──────────────────────────────


class TestHaversine:
    """Test the distance calculation helper."""

    def test_same_point_is_zero(self):
        assert _haversine(43.65, -79.38, 43.65, -79.38) == 0.0

    def test_known_distance(self):
        # Toronto CN Tower → Rogers Centre ≈ 300m
        dist = _haversine(43.6426, -79.3871, 43.6414, -79.3894)
        assert 100 < dist < 500, f"Expected ~300m, got {dist:.0f}m"


class TestDeduplication:
    """Test the dedup logic without calling any API."""

    def test_exact_duplicates_kept_higher_rating(self):
        venues = [
            {"name": "Cool Cafe", "lat": 43.65, "lng": -79.38, "rating": 4.0},
            {"name": "Cool Cafe", "lat": 43.65, "lng": -79.38, "rating": 4.5},
        ]
        result = _deduplicate(venues)
        assert len(result) == 1
        assert result[0]["rating"] == 4.5

    def test_different_names_not_deduped(self):
        venues = [
            {"name": "Place A", "lat": 43.65, "lng": -79.38, "rating": 4.0},
            {"name": "Place B", "lat": 43.65, "lng": -79.38, "rating": 4.0},
        ]
        result = _deduplicate(venues)
        assert len(result) == 2

    def test_same_name_far_apart_not_deduped(self):
        venues = [
            {"name": "Starbucks", "lat": 43.65, "lng": -79.38, "rating": 4.0},
            {"name": "Starbucks", "lat": 43.70, "lng": -79.40, "rating": 4.2},
        ]
        result = _deduplicate(venues)
        assert len(result) == 2


# ── Live API Test ──────────────────────────────────────────
# These tests hit real APIs and require env vars — skip in standard test runs.


@pytest.mark.integration
class TestScoutNode:
    """Test the full scout_node with live API calls."""

    @pytest.mark.integration
    def test_scout_returns_candidates(self):
        """Scout should return a non-empty list of candidate venues."""
        state = {
            "raw_prompt": "basketball courts in downtown Toronto",
            "parsed_intent": {
                "activity": "basketball courts",
                "location": "downtown Toronto",
            },
            "candidate_venues": [],
        }

        result = scout_node(state)

        candidates = result.get("candidate_venues", [])
        assert isinstance(candidates, list), "candidate_venues should be a list"
        assert len(candidates) > 0, "Scout should find at least 1 venue"
        assert len(candidates) <= 10, "Scout should cap at 10 venues"

        # Check the first venue has the expected shape
        first = candidates[0]
        assert "venue_id" in first, "Missing venue_id"
        assert "name" in first, "Missing name"
        assert "lat" in first, "Missing lat"
        assert "lng" in first, "Missing lng"
        assert isinstance(first["lat"], float), "lat should be float"
        assert isinstance(first["lng"], float), "lng should be float"
        assert "source" in first, "Missing source"
        assert first["source"] in ("google_places", "yelp"), f"Unknown source: {first['source']}"

        print(f"\n✅ Scout found {len(candidates)} candidates:")
        for v in candidates:
            print(f"   [{v['source']}] {v['name']} — ⭐ {v['rating']} ({v['review_count']} reviews)")

    def test_scout_empty_intent(self):
        """Scout with no query should return empty list gracefully."""
        state = {
            "raw_prompt": "",
            "parsed_intent": {},
            "candidate_venues": [],
        }
        result = scout_node(state)
        assert result["candidate_venues"] == []

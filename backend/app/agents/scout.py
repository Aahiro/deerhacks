"""
Node 2 — The SCOUT (Discovery)
Discovers candidate venues via Google Places API & Yelp Fusion,
merges + deduplicates, and enriches with Snowflake intelligence.
"""

import asyncio
import logging
from math import radians, cos, sin, asin, sqrt

from app.models.state import PathfinderState
from app.services.google_places import search_places
from app.services.yelp import search_yelp

logger = logging.getLogger(__name__)


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return distance in metres between two lat/lng points."""
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 6_371_000 * 2 * asin(sqrt(a))


def _deduplicate(venues: list[dict], threshold_m: float = 100) -> list[dict]:
    """
    Remove near-duplicates by name similarity + geographic proximity.
    If two venues have the same lowercase name and are within `threshold_m`
    metres of each other, keep the one with the higher rating.
    """
    kept: list[dict] = []
    for v in venues:
        is_dup = False
        for k in kept:
            same_name = v["name"].lower().strip() == k["name"].lower().strip()
            close = _haversine(v["lat"], v["lng"], k["lat"], k["lng"]) < threshold_m
            if same_name and close:
                # Keep the higher-rated one
                if v.get("rating", 0) > k.get("rating", 0):
                    kept.remove(k)
                    kept.append(v)
                is_dup = True
                break
        if not is_dup:
            kept.append(v)
    return kept


def scout_node(state: PathfinderState) -> PathfinderState:
    """
    Discover 5-10 candidate venues.

    Steps
    -----
    1. Build a search query from parsed_intent.
    2. Call Google Places + Yelp in parallel.
    3. Merge & deduplicate by name + proximity.
    4. Cap at 10 results.
    5. Return updated state with candidate_venues.
    """
    intent = state.get("parsed_intent", {})

    # Build search query from intent fields
    activity = intent.get("activity", "")
    location = intent.get("location", "Toronto")
    raw_prompt = state.get("raw_prompt", "")
    query = activity if activity else raw_prompt

    if not query:
        logger.warning("Scout: no query to search — returning empty")
        return {"candidate_venues": []}

    # Run both APIs concurrently — return_exceptions=True so one failure
    # doesn't wipe out the other API's results
    async def _fetch():
        return await asyncio.gather(
            search_places(query=query, location=location, max_results=8),
            search_yelp(term=query, location=location, max_results=8),
            return_exceptions=True,
        )

    try:
        google_raw, yelp_raw = asyncio.run(_fetch())
    except Exception as exc:
        logger.error("Scout gather failed: %s", exc)
        google_raw, yelp_raw = [], []

    google_results = google_raw if isinstance(google_raw, list) else []
    yelp_results = yelp_raw if isinstance(yelp_raw, list) else []
    if isinstance(google_raw, Exception):
        logger.warning("Google Places failed: %s", google_raw)
    if isinstance(yelp_raw, Exception):
        logger.warning("Yelp failed: %s", yelp_raw)

    # Merge all results
    all_venues = google_results + yelp_results

    # Deduplicate
    unique_venues = _deduplicate(all_venues)

    # Cap at 10
    candidates = unique_venues[:10]

    logger.info("Scout found %d candidates (%d Google, %d Yelp, %d after dedup)",
                len(candidates), len(google_results), len(yelp_results), len(unique_venues))

    return {"candidate_venues": candidates}

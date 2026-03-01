"""
Node 4 — The ACCESS ANALYST (Logistics)
Spatial reality check: travel-time feasibility + isochrone generation.
Tools: Mapbox Isochrone API, Google Distance Matrix

Responsibilities:
  1. Compute travel-time feasibility for the entire group.
  2. Penalise venues that are geographically close but chronologically far.
  3. Generate GeoJSON isochrones for frontend map overlays.
  4. Return accessibility_scores + isochrones to the shared state.
"""

import asyncio
import logging

from app.models.state import PathfinderState
from app.services.mapbox import get_isochrone, get_distance_matrix

logger = logging.getLogger(__name__)

# ── Default origin when user doesn't specify ──
# Downtown Toronto (CN Tower) as the baseline fallback
_DEFAULT_ORIGIN = (43.6426, -79.3871)


async def _analyze_venue_access(
    venue: dict,
    origin: tuple[float, float],
    profile: str = "driving",
) -> dict:
    """
    Run the full accessibility pipeline for a single venue.

    Steps
    -----
    1. Fetch isochrone from Mapbox (10/20/30-minute contours).
    2. Fetch distance-matrix entry from origin → venue via Google.
    3. Compute a normalised score (0.0–1.0).
    """
    lat = venue.get("lat", 0)
    lng = venue.get("lng", 0)

    # Run both API calls concurrently
    isochrone_data, dm_results = await asyncio.gather(
        get_isochrone(lat, lng, profile=profile, contour_minutes=[10, 20, 30]),
        get_distance_matrix(origin[0], origin[1], [(lat, lng)], mode=profile),
    )

    # --- Parse distance-matrix result ---
    dm = dm_results[0] if dm_results else {}
    duration_sec = dm.get("duration_sec")
    distance_m = dm.get("distance_m")
    dm_status = dm.get("status", "UNKNOWN")

    # --- Compute accessibility score ---
    if duration_sec is not None:
        travel_min = duration_sec / 60.0
        # Score: ≤10 min → 1.0, 30 min → 0.5, 60+ min → ~0.1
        if travel_min <= 10:
            score = 1.0
        elif travel_min <= 60:
            score = max(0.1, 1.0 - (travel_min - 10) / 55.0)
        else:
            score = 0.1
        score = round(score, 2)
        transit_accessible = profile == "transit" and dm_status == "OK"
    else:
        # API didn't return duration — neutral fallback
        travel_min = None
        score = 0.5
        transit_accessible = False

    accessibility_entry = {
        "score": score,
        "avg_travel_min": round(travel_min, 1) if travel_min else None,
        "max_travel_min": round(travel_min, 1) if travel_min else None,  # single origin
        "distance_m": distance_m,
        "transit_accessible": transit_accessible,
        "travel_mode": profile,
        "status": dm_status,
    }

    return {
        "accessibility": accessibility_entry,
        "isochrone": isochrone_data,  # GeoJSON FeatureCollection or None
    }


def _resolve_origin(state: dict) -> tuple[float, float]:
    """
    Determine the group's origin point for travel-time scoring.

    Priority
    --------
    1. Explicit origin_lat/origin_lng in parsed_intent.
    2. Centroid of member_locations (average lat/lng across all members).
    3. Downtown Toronto fallback.
    """
    intent = state.get("parsed_intent", {})

    # 1. Explicit origin in intent
    origin_lat = intent.get("origin_lat")
    origin_lng = intent.get("origin_lng")
    if origin_lat and origin_lng:
        return (float(origin_lat), float(origin_lng))

    # 2. Centroid of member locations
    members = state.get("member_locations", [])
    valid = [m for m in members if m.get("lat") and m.get("lng")]
    if valid:
        avg_lat = sum(m["lat"] for m in valid) / len(valid)
        avg_lng = sum(m["lng"] for m in valid) / len(valid)
        logger.info("Access Analyst: using centroid of %d member locations as origin", len(valid))
        return (avg_lat, avg_lng)

    return _DEFAULT_ORIGIN


def _resolve_travel_mode(state: dict) -> str:
    """
    Infer the best travel mode from the user's intent.
    """
    intent = state.get("parsed_intent", {})
    raw = state.get("raw_prompt", "").lower()

    # Keywords that signal transit / walking / cycling
    if any(kw in raw for kw in ["transit", "subway", "bus", "ttc", "public transport"]):
        return "transit"
    if any(kw in raw for kw in ["walk", "walking", "on foot"]):
        return "walking"
    if any(kw in raw for kw in ["bike", "cycling", "bicycle"]):
        return "cycling"

    return "driving"


def access_analyst_node(state: PathfinderState) -> PathfinderState:
    """
    Evaluate travel-time feasibility for the group.

    Steps
    -----
    1. Resolve the group's origin and preferred travel mode.
    2. For each candidate venue, call Mapbox Isochrone + Google Distance Matrix.
    3. Score accessibility (penalise "close but slow" venues).
    4. Generate GeoJSON isochrone blobs for frontend rendering.
    5. Return updated state with accessibility_scores + isochrones.
    """
    candidates = state.get("candidate_venues", [])

    if not candidates:
        logger.info("Access Analyst: no candidates to analyze")
        return {"accessibility_scores": {}, "isochrones": {}}

    origin = _resolve_origin(state)
    travel_mode = _resolve_travel_mode(state)

    # Mapbox Matrix API only supports driving/walking/cycling — remap transit → walking
    mapbox_mode = "walking" if travel_mode == "transit" else travel_mode
    if travel_mode == "transit":
        logger.info("Access Analyst: transit mode remapped to walking for Mapbox")

    logger.info(
        "Access Analyst: origin=%s, mode=%s, venues=%d",
        origin, travel_mode, len(candidates),
    )

    async def _analyze_all():
        return await asyncio.gather(
            *[_analyze_venue_access(v, origin, profile=mapbox_mode) for v in candidates]
        )

    try:
        results = asyncio.run(_analyze_all())
    except RuntimeError:
        # If event loop already running (e.g. inside LangGraph / Jupyter)
        import nest_asyncio
        nest_asyncio.apply()
        results = asyncio.run(_analyze_all())
    except Exception as exc:
        logger.error("Access Analyst failed: %s", exc)
        results = [{"accessibility": {"score": 0.5, "status": "ERROR"}, "isochrone": None}] * len(candidates)

    accessibility_scores = {}
    isochrones = {}

    for venue, result in zip(candidates, results):
        vid = venue.get("venue_id", "")
        acc = result["accessibility"]

        accessibility_scores[vid] = acc
        if result["isochrone"]:
            isochrones[vid] = result["isochrone"]

    scored = sum(1 for v in accessibility_scores.values() if v.get("score", 0) != 0.5)
    logger.info("Access Analyst scored %d/%d venues", scored, len(candidates))

    return {"accessibility_scores": accessibility_scores, "isochrones": isochrones}

"""
Mapbox Isochrone API service.

Fetches GeoJSON isochrone polygons representing reachable areas
from a given point within a specified travel time.
"""

import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_ISOCHRONE_URL = "https://api.mapbox.com/isochrone/v1/mapbox"


async def get_isochrone(
    lat: float,
    lng: float,
    profile: str = "driving",
    contour_minutes: int | list[int] = 15,
) -> Optional[dict]:
    """
    Fetch an isochrone polygon from the Mapbox Isochrone API.

    Parameters
    ----------
    lat, lng : float
        Centre point coordinates.
    profile : str
        Travel mode: "driving", "walking", or "cycling".
    contour_minutes : int or list[int]
        Travel time(s) in minutes (1–60). Pass a list for multiple contours,
        e.g. [10, 20, 30] returns three nested polygons in one response.

    Returns
    -------
    dict | None
        A GeoJSON FeatureCollection, or None on failure.
    """
    token = settings.MAPBOX_ACCESS_TOKEN
    if not token:
        logger.warning("Mapbox: no MAPBOX_ACCESS_TOKEN configured — skipping isochrone")
        return None

    url = f"{_ISOCHRONE_URL}/{profile}/{lng},{lat}"

    # Mapbox expects a comma-separated string for multiple contours
    if isinstance(contour_minutes, list):
        contours_str = ",".join(str(m) for m in contour_minutes)
    else:
        contours_str = str(contour_minutes)

    params = {
        "contours_minutes": contours_str,
        "polygons": "true",
        "access_token": token,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        # Validate it looks like GeoJSON
        if data.get("type") == "FeatureCollection" and data.get("features"):
            return data

        logger.warning("Mapbox isochrone returned unexpected shape for (%s, %s)", lat, lng)
        return data if data.get("type") else None

    except httpx.HTTPStatusError as exc:
        logger.warning("Mapbox isochrone HTTP %s for (%s, %s): %s",
                       exc.response.status_code, lat, lng, exc)
        return None
    except httpx.HTTPError as exc:
        logger.warning("Mapbox isochrone request failed for (%s, %s): %s", lat, lng, exc)
        return None

_MATRIX_URL = "https://api.mapbox.com/directions-matrix/v1/mapbox"


async def get_distance_matrix(
    origin_lat: float,
    origin_lng: float,
    destinations: list[tuple[float, float]],
    mode: str = "driving",
) -> list[dict]:
    """
    Fetch travel durations from one origin to multiple destinations via Mapbox Matrix API.
    
    Parameters
    ----------
    origin_lat, origin_lng : float
        Starting coordinates.
    destinations : list[(lat, lng)]
        List of target venue coordinates.
    mode : str
        "driving", "walking", or "cycling".
        
    Returns
    -------
    list[dict]
        Each dict has {"duration_sec": float, "distance_m": float, "status": "OK"|"FAIL"}.
    """
    token = settings.MAPBOX_ACCESS_TOKEN
    if not token:
        return []

    # Mapbox Matrix expects {lng,lat};{lng,lat}... 
    # First coordinate is origin (index 0)
    coords_str = f"{origin_lng},{origin_lat}"
    for dlat, dlng in destinations:
        coords_str += f";{dlng},{dlat}"

    url = f"{_MATRIX_URL}/{mode}/{coords_str}"
    params = {
        "sources": "0",  # Origin is at index 0
        "annotations": "duration,distance",
        "access_token": token,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        durations = data.get("durations", [[]])[0]  # first row because sources="0"
        distances = data.get("distances", [[]])[0]
        
        results = []
        # Index 0 in durations/distances is the origin-to-origin which we skip (it's 0)
        # So we iterate through the rest
        for i in range(1, len(durations)):
            results.append({
                "duration_sec": durations[i],
                "distance_m": distances[i],
                "status": "OK" if durations[i] is not None else "FAIL"
            })
        return results

    except Exception as exc:
        logger.warning("Mapbox matrix failed: %s", exc)
        return [{"duration_sec": None, "distance_m": None, "status": "ERROR"}] * len(destinations)

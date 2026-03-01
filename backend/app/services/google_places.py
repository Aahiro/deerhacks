"""
Google Places API service — venue discovery via Text Search.
All API keys stay server-side; nothing is exposed to the frontend.
"""

import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Google Places (New) Text Search endpoint
_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


async def search_places(
    query: str,
    location: Optional[str] = None,
    max_results: int = 10,
) -> list[dict]:
    """
    Search Google Places and return normalised venue dicts.

    Parameters
    ----------
    query : str
        Free-text search (e.g. "basketball courts downtown Toronto").
    location : str | None
        Optional area hint appended to the query.
    max_results : int
        Cap on returned results (default 10).

    Returns
    -------
    list[dict]  — each dict matches the Scout output schema.
    """
    if not settings.GOOGLE_CLOUD_API_KEY:
        logger.warning("GOOGLE_CLOUD_API_KEY not set — skipping Google Places")
        return []

    search_text = f"{query} {location}" if location else query

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.GOOGLE_CLOUD_API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.location,places.rating,places.userRatingCount,"
            "places.photos,places.primaryType,places.websiteUri,places.priceLevel"
        ),
    }

    body = {
        "textQuery": search_text,
        "maxResultCount": min(max_results, 20),
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(_TEXT_SEARCH_URL, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Google Places request failed: %s", exc)
        return []

    results = []
    for place in data.get("places", []):
        loc = place.get("location", {})
        photos = place.get("photos", [])

        # Build photo URLs (limited to first 3 to stay frugal)
        photo_urls = []
        for photo in photos[:3]:
            photo_name = photo.get("name", "")
            if photo_name:
                photo_urls.append(
                    f"https://places.googleapis.com/v1/{photo_name}/media"
                    f"?maxWidthPx=800&key={settings.GOOGLE_CLOUD_API_KEY}"
                )

        price_level = place.get("priceLevel")
        price_range = None
        if price_level == "PRICE_LEVEL_INEXPENSIVE":
            price_range = "$"
        elif price_level == "PRICE_LEVEL_MODERATE":
            price_range = "$$"
        elif price_level == "PRICE_LEVEL_EXPENSIVE":
            price_range = "$$$"
        elif price_level == "PRICE_LEVEL_VERY_EXPENSIVE":
            price_range = "$$$$"

        results.append({
            "venue_id": f"gp_{place.get('id', '')}",
            "name": place.get("displayName", {}).get("text", "Unknown"),
            "address": place.get("formattedAddress", ""),
            "lat": float(loc.get("latitude", 0)),
            "lng": float(loc.get("longitude", 0)),
            "rating": float(place.get("rating", 0)),
            "review_count": int(place.get("userRatingCount", 0)),
            "photos": photo_urls,
            "category": place.get("primaryType", ""),
            "price_range": price_range,
            "website": place.get("websiteUri", ""),
            "source": "google_places",
        })

    return results[:max_results]

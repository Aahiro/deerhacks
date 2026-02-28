"""
OpenWeather API service for the Critic node.
Uses the 5-day / 3-hour forecast so the Critic evaluates future conditions,
not just what it's like right now.
"""

import logging
import httpx
from typing import Optional, Dict, List
from app.core.config import settings

logger = logging.getLogger(__name__)


async def get_weather(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch the 5-day / 3-hour forecast and extract the next 24 hours
    of conditions in a compact form suitable for Gemini reasoning.
    """
    api_key = settings.OPENWEATHER_API_KEY
    if not api_key:
        logger.warning("OPENWEATHER_API_KEY not set")
        return None

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "cnt": 8,  # next 8 × 3h = 24 hours
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        periods: List[Dict] = []
        for entry in data.get("list", []):
            weather = entry.get("weather", [{}])[0]
            main = entry.get("main", {})
            periods.append({
                "time": entry.get("dt_txt", ""),
                "condition": weather.get("main", "Unknown"),
                "description": weather.get("description", ""),
                "temp_c": main.get("temp"),
                "feels_like_c": main.get("feels_like"),
                "pop": entry.get("pop", 0),  # probability of precipitation (0–1)
            })

        # Surface the worst condition across the window for easy Gemini parsing
        heavy_rain = any(
            p.get("pop", 0) >= 0.6 or
            p.get("condition", "") in ("Rain", "Drizzle", "Thunderstorm", "Snow")
            for p in periods
        )

        return {
            "forecast_24h": periods,
            "heavy_precipitation_likely": heavy_rain,
            "summary": (
                "Heavy precipitation expected in the next 24 hours."
                if heavy_rain
                else "No significant precipitation expected."
            ),
        }

    except Exception as exc:
        logger.error("OpenWeather forecast API failed: %s", exc)
        return None

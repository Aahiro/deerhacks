"""
OpenWeather API service for the Critic node.
"""

import logging
import httpx
from typing import Optional, Dict
from app.core.config import settings

logger = logging.getLogger(__name__)

async def get_weather(lat: float, lon: float) -> Optional[Dict]:
    """ Fetch current weather and simple forecast warnings. """
    api_key = settings.OPENWEATHER_API_KEY
    if not api_key:
        logger.warning("OPENWEATHER_API_KEY not set")
        return None

    # We use the current weather endpoint and a simple 5-day forecast for risk checks
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract high-level summary suitable for Gemini to read
            weather = data.get("weather", [{}])[0]
            main_temps = data.get("main", {})
            return {
                "condition": weather.get("main", "Unknown"),
                "description": weather.get("description", "Unknown"),
                "temp_c": main_temps.get("temp"),
                "feels_like_c": main_temps.get("feels_like")
            }
    except Exception as exc:
        logger.error("OpenWeather API failed: %s", exc)
        return None

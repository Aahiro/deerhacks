"""
Application configuration — loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Google Cloud (Gemini + Google Places) ──
    GOOGLE_CLOUD_API_KEY: str = ""

    # ── Yelp ──
    YELP_API_KEY: str = ""

    # ── Mapbox ──
    MAPBOX_ACCESS_TOKEN: str = ""

    # ── OpenWeather ──
    OPENWEATHER_API_KEY: str = ""

    # ── PredictHQ ──
    PREDICTHQ_API_KEY: str = ""

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Firecrawl ──
    FIRECRAWL_API_KEY: str = ""

    # ── Auth0 ──
    AUTH0_DOMAIN: str = ""      # e.g. dev-xyz.us.auth0.com
    AUTH0_AUDIENCE: str = ""    # e.g. https://pathfinder.api

    # ── CORS ──
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()

"""
Application configuration — loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import json


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

    # ── CORS ──
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Firecrawl ──
    FIRECRAWL_API_KEY: str = ""

    # ── Auth0 ──
    AUTH0_DOMAIN: Optional[str] = None
    AUTH0_CLIENT_ID: Optional[str] = None
    AUTH0_CLIENT_SECRET: Optional[str] = None
    AUTH0_AUDIENCE: Optional[str] = None
    AUTH0_SECRET: Optional[str] = None

    # ── ElevenLabs ──
    ELEVENLABS_API_KEY: str = ""

    # ── Snowflake ──
    SNOWFLAKE_ACCOUNT: Optional[str] = None
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_SCHEMA: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    SNOWFLAKE_ROLE: Optional[str] = None

    # ── CORS ──
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()

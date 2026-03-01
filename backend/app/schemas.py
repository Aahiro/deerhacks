"""
Pydantic schemas for API request / response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request ──────────────────────────────────────────────


class PlanRequest(BaseModel):
    """Incoming user prompt for venue planning."""

    prompt: str = Field(..., description="Natural-language activity request")
    group_size: int = Field(1, ge=1, description="Number of people in the group")
    budget: Optional[str] = Field(None, description="Budget preference: low | medium | high")
    location: Optional[str] = Field(None, description="Preferred area / neighbourhood")
    vibe: Optional[str] = Field(None, description="Desired vibe or aesthetic")
    member_locations: Optional[List[dict]] = Field(
        None,
        description="List of {lat, lng} dicts for each group member",
    )
    chat_history: Optional[List[dict]] = Field(
        None,
        description="Previous conversation turns for reprompting [{role, content}]",
    )


# ── Response ─────────────────────────────────────────────


class VenueResult(BaseModel):
    """A single ranked venue returned to the frontend."""

    rank: int
    name: str
    address: str
    lat: float
    lng: float
    vibe_score: Optional[float] = None
    cost_profile: Optional[dict] = None
    why: str = ""
    watch_out: str = ""


class PlanResponse(BaseModel):
    """Final response containing ranked venues + metadata."""

    venues: List[VenueResult]
    execution_summary: Optional[str] = None

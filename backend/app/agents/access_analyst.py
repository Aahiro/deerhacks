"""
Node 4 — The ACCESS ANALYST (Logistics)
Spatial reality check: travel-time feasibility + isochrone generation.
Tools: Mapbox Isochrone API, Google Distance Matrix
"""

from app.models.state import PathfinderState


def access_analyst_node(state: PathfinderState) -> PathfinderState:
    """
    Evaluate travel-time feasibility for the group.

    Steps
    -----
    1. For each candidate venue, call Mapbox Isochrone API.
    2. For each member location, call Google Distance Matrix.
    3. Score accessibility (penalise "close but slow" venues).
    4. Generate GeoJSON isochrone blobs for frontend rendering.
    5. Return updated state with accessibility_scores + isochrones.
    """
    return {"accessibility_scores": {}, "isochrones": {}}

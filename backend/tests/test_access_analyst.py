
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, AsyncMock
from app.agents.access_analyst import access_analyst_node

_MOCK_ISOCHRONE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-79.4, 43.6], [-79.3, 43.6], [-79.3, 43.7], [-79.4, 43.7], [-79.4, 43.6]]]
            }
        }
    ]
}

_MOCK_DM = [{"duration_sec": 600, "distance_m": 5000, "status": "OK"}]

@patch("app.agents.access_analyst.auth0_service", create=True)
@patch("app.agents.access_analyst.get_isochrone", new_callable=AsyncMock)
@patch("app.agents.access_analyst.get_distance_matrix", new_callable=AsyncMock)
def test_access_analyst_with_candidates(mock_dm, mock_isochrone, mock_auth0):
    mock_isochrone.return_value = _MOCK_ISOCHRONE
    mock_dm.return_value = _MOCK_DM
    
    # Mock Auth0 IDP token response
    mock_auth0.get_idp_token = AsyncMock(return_value="mock_google_token_12345")

    state = {
        "candidate_venues": [
            {"venue_id": "v1", "name": "Venue A", "lat": 43.65, "lng": -79.38},
        ],
        "parsed_intent": {},
        "auth_user_id": "test_user_calendar",
    }

    result = access_analyst_node(state)

    assert "accessibility_scores" in result
    assert "isochrones" in result
    assert "v1" in result["accessibility_scores"]
    assert result["accessibility_scores"]["v1"]["score"] <= 1.0  # Due to modular arithmetic in the mock, it could be penalized
    assert "v1" in result["isochrones"]
    
def test_access_analyst_no_candidates():
    state = {"candidate_venues": []}
    result = access_analyst_node(state)
    assert result["accessibility_scores"] == {}
    assert result["isochrones"] == {}

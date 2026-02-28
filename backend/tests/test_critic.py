from unittest.mock import patch, AsyncMock
from app.agents.critic import critic_node
from app.models.state import PathfinderState

@patch('app.agents.critic.generate_content', new_callable=AsyncMock)
@patch('app.agents.critic.get_events', new_callable=AsyncMock)
@patch('app.agents.critic.get_weather', new_callable=AsyncMock)
def test_critic_node_veto(mock_get_weather, mock_get_events, mock_generate_content):
    # Mock services
    mock_get_weather.return_value = {"condition": "Rain"}
    mock_get_events.return_value = [{"title": "Marathon"}]
    
    # Mock Gemini telling us it's a veto
    mock_generate_content.return_value = '''
    {
        "risks": [
            {"type": "weather", "severity": "high", "detail": "Rain"}
        ],
        "veto": true,
        "veto_reason": "Too rainy for an outdoor event."
    }
    '''
    
    state: PathfinderState = {
        "parsed_intent": {"activity": "picnic"},
        "candidate_venues": [
            {"venue_id": "v1", "name": "Park A", "lat": 43.0, "lng": -79.0}
        ]
    }
    
    new_state = critic_node(state)
    
    assert new_state["veto"] is True
    assert new_state["veto_reason"] == "Too rainy for an outdoor event."
    assert "v1" in new_state["risk_flags"]
    assert len(new_state["risk_flags"]["v1"]) == 1

@patch('app.agents.critic.generate_content', new_callable=AsyncMock)
@patch('app.agents.critic.get_events', new_callable=AsyncMock)
@patch('app.agents.critic.get_weather', new_callable=AsyncMock)
def test_critic_node_no_candidates(mock_get_weather, mock_get_events, mock_generate_content):
    state: PathfinderState = {"candidate_venues": []}
    new_state = critic_node(state)
    
    assert new_state["veto"] is False
    assert not new_state["risk_flags"]
    mock_get_weather.assert_not_called()

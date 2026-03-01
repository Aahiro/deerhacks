
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.agents.commander import commander_node

@patch('app.agents.commander.auth0_service', create=True)
@patch('app.agents.commander.generate_content', new_callable=AsyncMock)
def test_commander_node_success(mock_generate_content, mock_auth0):
    # Mock Gemini response
    mock_generate_content.return_value = '''
    {
      "parsed_intent": {"activity": "basketball", "budget": "low"},
      "complexity_tier": "tier_2",
      "active_agents": ["scout", "cost_analyst"],
      "agent_weights": {"scout": 1.0, "cost_analyst": 0.8}
    }
    '''
    
    # Mock Auth0 response
    mock_auth0.get_user_profile = AsyncMock(return_value={
        "app_metadata": {"preferences": {"budget_sensitive": True}}
    })
    
    state = {"raw_prompt": "basketball court cheap", "auth_user_id": "test_user"}
    new_state = commander_node(state)
    
    assert new_state["parsed_intent"]["activity"] == "basketball"
    assert new_state["complexity_tier"] == "tier_2"
    assert "scout" in new_state["active_agents"]
    assert new_state["user_profile"] is not None

@patch('app.agents.commander.generate_content', new_callable=AsyncMock)
def test_commander_node_fallback_on_error(mock_generate_content):
    # Mock Gemini throwing an error
    mock_generate_content.side_effect = Exception("API Error")
    
    state = {"raw_prompt": "basketball court"}
    new_state = commander_node(state)
    
    # Assert fallback values (Now tier_2 because of heuristic fallback in code)
    assert new_state["complexity_tier"] == "tier_2"
    assert "scout" in new_state["active_agents"]

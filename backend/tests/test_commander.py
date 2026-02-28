import pytest
from unittest.mock import patch, MagicMock
from app.agents.commander import commander_node
from app.models.state import PathfinderState

@patch('app.agents.commander.snowflake_service.cortex_search')
@patch('app.agents.commander.generate_content')
def test_commander_node_success(mock_generate_content, mock_cortex_search):
    # Mock Gemini response
    mock_generate_content.return_value = '''
    {
      "parsed_intent": {"activity": "basketball", "budget": "low"},
      "complexity_tier": "tier_2",
      "active_agents": ["scout", "cost_analyst"],
      "agent_weights": {"scout": 1.0, "cost_analyst": 0.8}
    }
    '''
    
    # Mock Snowflake Cortex response
    mock_cortex_search.return_value = ["Risk memory 1"]
    
    state: PathfinderState = {"raw_prompt": "basketball court cheap"}
    new_state = commander_node(state)
    
    assert new_state["parsed_intent"]["activity"] == "basketball"
    assert new_state["complexity_tier"] == "tier_2"
    assert "scout" in new_state["active_agents"]
    assert new_state["snowflake_context"][0] == "Risk memory 1"

@patch('app.agents.commander.generate_content')
def test_commander_node_fallback_on_error(mock_generate_content):
    # Mock Gemini throwing an error
    mock_generate_content.side_effect = Exception("API Error")
    
    state: PathfinderState = {"raw_prompt": "basketball court"}
    new_state = commander_node(state)
    
    # Assert fallback values
    assert new_state["complexity_tier"] == "tier_1"
    assert new_state["active_agents"] == ["scout"]
    assert new_state["agent_weights"] == {"scout": 1.0}

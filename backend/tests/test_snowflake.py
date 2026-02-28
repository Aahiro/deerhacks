import pytest
from unittest.mock import patch, MagicMock
from app.services.snowflake import SnowflakeService

@pytest.fixture
def mock_snowflake_service():
    service = SnowflakeService()
    service.account = "test_account"
    service.user = "test_user"
    service.password = "test_password"
    return service

@patch('app.services.snowflake.snowflake.connector.connect')
def test_snowflake_connection(mock_connect, mock_snowflake_service):
    """Test that the Snowflake connection calls the connector with right credentials."""
    mock_snowflake_service._get_connection()
    mock_connect.assert_called_once_with(
        user="test_user",
        password="test_password",
        account="test_account",
        database="PATHFINDER",
        schema="PUBLIC",
        warehouse="COMPUTE_WH",
        role="ACCOUNTADMIN"
    )

@patch('app.services.snowflake.snowflake.connector.connect')
def test_log_risk(mock_connect, mock_snowflake_service):
    """Test logging risk execution."""
    mock_cur = MagicMock()
    mock_connect.return_value.cursor.return_value = mock_cur
    
    venue_id = "v_123"
    risk_type = "weather"
    details = {"rain": True}
    
    mock_snowflake_service.log_risk(venue_id, risk_type, details)
    mock_cur.execute.assert_called_once()
    assert "INSERT INTO risk_history" in mock_cur.execute.call_args[0][0]

@patch('app.services.snowflake.snowflake.connector.connect')
def test_cortex_search(mock_connect, mock_snowflake_service):
    """Test Cortex search execution."""
    mock_cur = MagicMock()
    # Mock return value
    mock_cur.fetchall.return_value = [("Context 1",), ("Context 2",)]
    mock_connect.return_value.cursor.return_value = mock_cur
    
    res = mock_snowflake_service.cortex_search("test query", top_k=2)
    
    assert len(res) == 2
    assert res[0] == "Context 1"
    mock_cur.execute.assert_called_once()
    assert "SNOWFLAKE.CORTEX.SEARCH_MATCH" in mock_cur.execute.call_args[0][0]

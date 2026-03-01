"""
pytest configuration for PATHFINDER backend tests.

Marks:
  integration — tests that hit live APIs (require env vars + network).
               Skipped by default. Run with: pytest -m integration
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as a live API integration test (skipped by default)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests unless -m integration is passed explicitly."""
    # Only skip if the user hasn't explicitly asked for integration tests
    if "integration" in (config.option.markexpr or ""):
        return  # user asked for them — let them run
    skip_integration = pytest.mark.skip(reason="integration test — run with: pytest -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

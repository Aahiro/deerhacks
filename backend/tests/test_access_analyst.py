"""
Access Analyst tests — skipped.

The Access Analyst agent was intentionally removed in the upstream merge
(parallel_analysts_node now coordinates vibe_matcher, cost_analyst, and critic only).
Accessibility scoring has been retired from this pipeline version.
"""
import pytest

pytestmark = pytest.mark.skip(reason="access_analyst removed in upstream merge")

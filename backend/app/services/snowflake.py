"""
Snowflake service — long-term memory & predictive intelligence (Node 7).
Handles connections, risk storage, RAG queries, and trend analysis.
"""

import json
import logging
import snowflake.connector
from app.core.config import settings

logger = logging.getLogger(__name__)


class SnowflakeService:
    """Interface to Snowflake for PATHFINDER memory & intelligence."""

    def __init__(self):
        self.account = settings.SNOWFLAKE_ACCOUNT
        self.user = settings.SNOWFLAKE_USER
        self.password = settings.SNOWFLAKE_PASSWORD
        self.database = settings.SNOWFLAKE_DATABASE
        self.schema = settings.SNOWFLAKE_SCHEMA
        self.warehouse = settings.SNOWFLAKE_WAREHOUSE
        self.role = settings.SNOWFLAKE_ROLE

    # ── Connection ────────────────────────────────────────

    def _get_connection(self):
        """Return a Snowflake connection (lazy-initialized)."""
        return snowflake.connector.connect(
            user=self.user,
            password=self.password,
            account=self.account,
            database=self.database,
            schema=self.schema,
            warehouse=self.warehouse,
            role=self.role
        )

    # ── Risk Storage ──────────────────────────────────────

    def log_risk(self, venue_id: str, risk_type: str, details: dict):
        """Persist a historical risk event."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            query = """
                INSERT INTO risk_history (venue_id, risk_type, details)
                VALUES (%s, %s, PARSE_JSON(%s))
            """
            cur.execute(query, (venue_id, risk_type, json.dumps(details)))
        except Exception as e:
            logger.error(f"Failed to log risk: {e}")
        finally:
            if 'cur' in locals(): cur.close()
            if 'conn' in locals(): conn.close()

    def get_risks(self, venue_id: str) -> list:
        """Retrieve historical risk events for a venue."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            query = """
                SELECT risk_type, details, created_at
                FROM risk_history
                WHERE venue_id = %s
                ORDER BY created_at DESC
            """
            cur.execute(query, (venue_id,))
            rows = cur.fetchall()
            return [{"risk_type": r[0], "details": json.loads(r[1]) if isinstance(r[1], str) else r[1], "created_at": str(r[2])} for r in rows]
        except Exception as e:
            logger.error(f"Failed to get risks: {e}")
            return []
        finally:
            if 'cur' in locals(): cur.close()
            if 'conn' in locals(): conn.close()

    # ── RAG / Cortex Search ───────────────────────────────

    def cortex_search(self, query: str, top_k: int = 5) -> list:
        """Run a Snowflake Cortex Search for RAG enrichment."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            # Hypothetical Cortex Search query structure
            query = f"""
                SELECT context_chunk 
                FROM PATHFINDER.PUBLIC.venue_knowledge_base
                WHERE SNOWFLAKE.CORTEX.SEARCH_MATCH(context_chunk, '{query.replace("'", "''")}') > 0.5
                LIMIT {top_k}
            """
            cur.execute(query)
            rows = cur.fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.error(f"Cortex search failed: {e}")
            return []
        finally:
            if 'cur' in locals(): cur.close()
            if 'conn' in locals(): conn.close()

    # ── Trend Analysis ────────────────────────────────────

    def get_price_trends(self, venue_id: str) -> dict:
        """Return seasonal pricing trend data for a venue."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            query = """
                SELECT json_data 
                FROM price_trends
                WHERE venue_id = %s
            """
            cur.execute(query, (venue_id,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0]) if isinstance(row[0], str) else row[0]
            return {}
        except Exception as e:
            logger.error(f"Failed to get price trends: {e}")
            return {}
        finally:
            if 'cur' in locals(): cur.close()
            if 'conn' in locals(): conn.close()


snowflake_service = SnowflakeService()

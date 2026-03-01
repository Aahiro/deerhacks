"""
Snowflake memory service — log and retrieve historical risk data.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_connection():
    """Create a Snowflake connection using settings."""
    try:
        import snowflake.connector
        return snowflake.connector.connect(
            account=settings.SNOWFLAKE_ACCOUNT,
            user=settings.SNOWFLAKE_USER,
            password=settings.SNOWFLAKE_PASSWORD,
            database=settings.SNOWFLAKE_DATABASE,
            schema=settings.SNOWFLAKE_SCHEMA,
            warehouse=settings.SNOWFLAKE_WAREHOUSE,
            role=settings.SNOWFLAKE_ROLE,
        )
    except Exception as e:
        logger.error(f"Snowflake connection failed: {e}")
        return None


def ensure_tables():
    """Create the risk_log table if it doesn't exist."""
    conn = _get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_log (
                id INTEGER AUTOINCREMENT PRIMARY KEY,
                venue_id VARCHAR(200) NOT NULL,
                venue_name VARCHAR(500),
                risk_type VARCHAR(100) NOT NULL,
                description TEXT NOT NULL,
                severity VARCHAR(20) DEFAULT 'medium',
                logged_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                query_context TEXT
            )
        """)
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to create risk_log table: {e}")
        return False
    finally:
        conn.close()


def log_risk(
    venue_id: str,
    risk_type: str,
    description: str,
    venue_name: Optional[str] = None,
    severity: str = "medium",
    query_context: Optional[str] = None,
) -> bool:
    """
    Persist a risk event to Snowflake.

    Args:
        venue_id: Unique venue identifier.
        risk_type: Category (e.g., "weather", "noise", "closure", "congestion").
        description: Human-readable risk description.
        venue_name: Optional venue display name.
        severity: "low", "medium", or "high".
        query_context: The original user query that triggered this risk.

    Returns:
        True if logged successfully, False otherwise.
    """
    conn = _get_connection()
    if not conn:
        logger.warning("Snowflake unavailable — risk not logged.")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO risk_log (venue_id, venue_name, risk_type, description, severity, query_context)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (venue_id, venue_name, risk_type, description, severity, query_context),
        )
        conn.commit()
        logger.info(f"Logged risk for {venue_id}: {risk_type}")
        return True
    except Exception as e:
        logger.error(f"Failed to log risk: {e}")
        return False
    finally:
        conn.close()


def get_risks(
    venue_id: Optional[str] = None,
    risk_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Retrieve historical risks from Snowflake.

    Args:
        venue_id: Filter by venue (optional).
        risk_type: Filter by risk type (optional).
        limit: Max results to return.

    Returns:
        List of risk records as dicts.
    """
    conn = _get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()
        query = "SELECT venue_id, venue_name, risk_type, description, severity, logged_at FROM risk_log"
        params = []
        conditions = []

        if venue_id:
            conditions.append("venue_id = %s")
            params.append(venue_id)
        if risk_type:
            conditions.append("risk_type = %s")
            params.append(risk_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY logged_at DESC LIMIT %s"
        params.append(limit)

        cursor.execute(query, params)
        columns = [desc[0].lower() for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"Failed to retrieve risks: {e}")
        return []
    finally:
        conn.close()


def get_venue_risk_summary(venue_id: str) -> Dict[str, Any]:
    """
    Get a summary of historical risks for a specific venue.
    Useful for the Critic and Scout enrichment.
    """
    risks = get_risks(venue_id=venue_id)

    if not risks:
        return {"venue_id": venue_id, "total_risks": 0, "risks": []}

    risk_counts = {}
    for r in risks:
        rtype = r.get("risk_type", "unknown")
        risk_counts[rtype] = risk_counts.get(rtype, 0) + 1

    return {
        "venue_id": venue_id,
        "total_risks": len(risks),
        "risk_breakdown": risk_counts,
        "most_recent": risks[0] if risks else None,
        "risks": risks[:10],  # Return the 10 most recent
    }

"""
Node 4 -- The COST ANALYST (Financial)
"No-surprises" auditor: Normalizes price signals extracted from discovery APIs.

Pricing is now informational only, and we rely on heuristic analysis of 
the price ranges ($ to $$$$) returned by Google Places and Yelp, 
resolving conflicts appropriately.
"""

import logging
from app.models.state import PathfinderState

logger = logging.getLogger(__name__)

# Convert $ strings to numeric values for median calculation
_PRICE_VALUES = {
    "$": 1,
    "$$": 2,
    "$$$": 3,
    "$$$$": 4
}

_NUM_TO_PRICE = {
    1: "$",
    2: "$$",
    3: "$$$",
    4: "$$$$"
}

def _calculate_value_score(price_range: str, confidence: str) -> float:
    """
    Assign a simple subjective value_score for the Synthesiser based on price.
    We assume lower price is generally 'better value' for sorting,
    but capped by confidence.
    """
    if confidence == "none" or not price_range:
        return 0.3
    
    val = _PRICE_VALUES.get(price_range, 2)
    
    # Simple heuristic: $ -> 0.8, $$ -> 0.6, $$$ -> 0.4, $$$$ -> 0.2
    base_score = 1.0 - (val * 0.2)
    
    # Penalize low confidence
    if confidence == "low":
        base_score -= 0.1
    elif confidence == "estimated" or confidence == "medium":
        base_score -= 0.05
        
    return max(0.1, round(base_score, 2))


def _analyze_venue_cost(venue: dict) -> dict:
    """
    Determine price_range and confidence for a single venue by resolving conflicts 
    if both Google and Yelp data were somehow passed (e.g. merged), OR just by 
    reading the scout's `price_range` if it's a single source.
    """
    # If scout.py passed google_price and yelp_price explicitly:
    google_price = venue.get("google_price")
    yelp_price = venue.get("yelp_price")
    
    # If not explicitly merged strings but we just have one:
    if not google_price and not yelp_price:
        source = venue.get("source")
        if source == "google_places":
            google_price = venue.get("price_range")
        elif source == "yelp":
            yelp_price = venue.get("price_range")
        else:
            google_price = venue.get("price_range")

    if google_price and yelp_price:
        if google_price == yelp_price:
            resolved_price = google_price
            confidence = "high"
        else:
            # Conflict -> median
            val_g = _PRICE_VALUES.get(google_price, 2)
            val_y = _PRICE_VALUES.get(yelp_price, 2)
            median_val = round((val_g + val_y) / 2) # e.g. 1 and 2 -> 1.5 -> 2
            resolved_price = _NUM_TO_PRICE.get(median_val, "$$")
            confidence = "low"
    elif google_price:
        resolved_price = google_price
        confidence = "medium"
    elif yelp_price:
        resolved_price = yelp_price
        confidence = "medium"
    else:
        resolved_price = None
        confidence = "none"

    return {
        "price_range": resolved_price,
        "confidence": confidence,
        "value_score": _calculate_value_score(resolved_price, confidence)
    }

def cost_analyst_node(state: PathfinderState) -> PathfinderState:
    """
    Compute price normalization per venue.
    """
    candidates = state.get("candidate_venues", [])

    if not candidates:
        logger.info("Cost Analyst: no candidates to analyze")
        return {"cost_profiles": {}}

    cost_profiles = {}
    for venue in candidates:
        vid = venue.get("venue_id", venue.get("name", "unknown"))
        cost_profiles[vid] = _analyze_venue_cost(venue)

    scored = sum(1 for v in cost_profiles.values() if v.get("price_range"))
    logger.info("Cost Analyst priced %d/%d venues", scored, len(candidates))

    return {
        "cost_profiles": cost_profiles
    }

"""
Node 5 -- The COST ANALYST (Financial)
"No-surprises" auditor: scrapes true cost and compares to Snowflake history.

Scraping Strategy:
  1. Firecrawl /map    -> discover "Pricing" / "Menu" page on venue website
  2. Firecrawl /scrape -> extract page content as markdown
  3. Gemini            -> extract structured pricing from content

Fallback Tiers:
  - Confirmed pricing  -> value_score from Gemini assessment
  - Estimated pricing   -> value_score capped at 0.5 with estimation note
  - Unknown pricing     -> value_score 0.3 with uncertainty warning

Tools: Firecrawl, Gemini
"""

import asyncio
import json
import logging
from typing import Optional

import httpx

from app.models.state import PathfinderState
from app.services.gemini import generate_content
from app.core.config import settings

logger = logging.getLogger(__name__)


# -- Firecrawl ----------------------------------------------------------
_FIRECRAWL_BASE = "https://api.firecrawl.dev/v1"


async def _firecrawl_map(website_url: str) -> list[str]:
    """
    Use Firecrawl /map to discover sub-pages on a venue website.
    Returns a list of page URLs, filtered for pricing-related pages.
    """
    if not settings.FIRECRAWL_API_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{_FIRECRAWL_BASE}/map",
                headers={"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}"},
                json={
                    "url": website_url,
                    "search": "pricing rates cost fees menu package book reserve",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        all_links = data.get("links", [])

        # Filter for pricing / menu / rates pages
        pricing_keywords = ["pric", "rate", "cost", "fee", "menu", "book", "package"]
        relevant = [
            link for link in all_links
            if any(kw in link.lower() for kw in pricing_keywords)
        ]

        return relevant if relevant else all_links[:3]

    except httpx.HTTPError as exc:
        logger.warning("Firecrawl /map failed for %s: %s", website_url, exc)
        return []


async def _firecrawl_scrape(page_url: str) -> Optional[str]:
    """
    Use Firecrawl /scrape to extract page content as markdown.
    Returns the markdown text of the page.
    """
    if not settings.FIRECRAWL_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{_FIRECRAWL_BASE}/scrape",
                headers={"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}"},
                json={
                    "url": page_url,
                    "formats": ["markdown"],
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return data.get("data", {}).get("markdown", "")

    except httpx.HTTPError as exc:
        logger.warning("Firecrawl /scrape failed for %s: %s", page_url, exc)
        return None



# -- Pricing extraction via Gemini --------------------------------------

_COST_PROMPT = """You are a pricing analyst for a group activity planning app in Toronto, Canada. Your job is to extract or estimate costs.

Venue: {name}
Category: {category}
Group size: {group_size}

Website content:
{content}

INSTRUCTIONS:
1. Search for ANY pricing signals: hourly rates, per-person fees, packages, menu prices, rental costs, booking rates, membership fees, event packages.
2. Identify hidden costs: shoe rentals, equipment fees, minimum spends, service charges, cleaning fees, parking fees, lane fees, food minimums.
3. If you find EXPLICIT prices on the page, use them and set pricing_confidence to "confirmed".
4. If prices are NOT explicitly listed, ESTIMATE based on:
   - Typical Toronto market rates for this type of venue/activity
   - The venue's category and perceived quality
   - Group size of {group_size} people
   Set pricing_confidence to "estimated".
5. NEVER return base_cost as 0 unless you are confident the activity is genuinely free. Bowling in Toronto typically costs $6-12 per person per game. Cafes cost $5-15 per person. Use your knowledge.

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "base_cost": <float, primary cost for the group in CAD>,
  "hidden_costs": [
    {{"label": "<fee name>", "amount": <float>}}
  ],
  "total_cost_of_attendance": <float, base + all hidden costs>,
  "per_person": <float, total / group_size>,
  "value_score": <float 0.0 to 1.0, subjective value for money>,
  "pricing_confidence": "<confirmed | estimated | unknown>",
  "notes": "<pricing observations, source of estimates, or warnings>"
}}
"""


async def _extract_pricing(
    venue: dict, content: str, group_size: int
) -> dict:
    """Use Gemini to extract structured pricing from scraped content."""
    prompt = _COST_PROMPT.format(
        name=venue.get("name", "Unknown"),
        category=venue.get("category", "venue"),
        group_size=group_size,
        content=content[:50000],  # Increased to support multi-page combined content
    )

    try:
        raw = await generate_content(prompt=prompt, model="gemini-2.5-flash")
        if not raw:
            return _no_data_fallback(venue)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]

        result = json.loads(cleaned.strip())
        return _apply_confidence_tier(result, venue, group_size)

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Cost extraction failed for %s: %s", venue.get("name"), exc)
        return _no_data_fallback(venue)


def _apply_confidence_tier(result: dict, venue: dict, group_size: int) -> dict:
    """
    Apply tiered value_score and notes based on pricing confidence.

    Tiers:
      confirmed -> use Gemini's value_score as-is
      estimated -> cap value_score at 0.5, add estimation warning
      unknown   -> value_score 0.3, add high-uncertainty warning
    """
    confidence = result.get("pricing_confidence", "unknown")
    base = result.get("base_cost", 0)

    if confidence == "unknown" or base == 0:
        # -- Tier: No price found --
        result["value_score"] = 0.3
        result["pricing_confidence"] = "unknown"
        result["notes"] = (
            "High uncertainty. Rates are quote-based or unlisted. "
            "Budget fit is unverified. Recommend contacting venue directly."
        )

    elif confidence == "estimated":
        # -- Tier: Estimated price --
        result["value_score"] = min(result.get("value_score", 0.5), 0.5)
        est_note = result.get("notes", "")
        result["notes"] = (
            f"Estimated from Toronto market rates for {venue.get('category', 'this venue type')}. "
            f"{est_note}"
        ).strip()

    # else: confirmed -> keep Gemini's original value_score and notes

    # Ensure per_person is always calculated
    tca = result.get("total_cost_of_attendance", 0)
    if tca > 0 and group_size > 0:
        result["per_person"] = round(tca / group_size, 2)

    return result


def _no_data_fallback(venue: dict = None) -> dict:
    """Fallback when scraping fails entirely -- no content was retrieved."""
    name = venue.get("name", "This venue") if venue else "This venue"
    return {
        "base_cost": 0,
        "hidden_costs": [],
        "total_cost_of_attendance": 0,
        "per_person": 0,
        "value_score": 0.3,
        "pricing_confidence": "unknown",
        "notes": (
            f"High uncertainty. {name} does not publish rates online. "
            "Budget fit is unverified. Recommend contacting venue directly."
        ),
    }


# -- Main pipeline per venue -------------------------------------------

async def _analyze_venue_cost(venue: dict, group_size: int) -> dict:
    """
    Full cost pipeline for a single venue:
    1. Try Firecrawl /map to find pricing pages
    2. Firecrawl /scrape to get page content
    3. Use Gemini to extract structured pricing
    """
    website = venue.get("website", "")
    combined_content = ""

    if website:
        # Step 1: Discover pricing pages
        pricing_pages = await _firecrawl_map(website)

        # Step 2: Scrape up to 3 pricing pages, plus always include the homepage
        pages_to_scrape = pricing_pages[:3]
        if website not in pages_to_scrape:
            pages_to_scrape.append(website)

        content_parts = []
        for target in pages_to_scrape:
            content = await _firecrawl_scrape(target) or ""
            if content:
                content_parts.append(f"--- Content from {target} ---\n{content}\n")

        combined_content = "\n".join(content_parts)

    if not combined_content:
        return _no_data_fallback(venue)

    # Step 3: Extract pricing with Gemini
    return await _extract_pricing(venue, combined_content, group_size)


# -- Node entry point ---------------------------------------------------

def cost_analyst_node(state: PathfinderState) -> PathfinderState:
    """
    Compute Total Cost of Attendance (TCA) per venue.

    Steps
    -----
    1. For each venue, run the Firecrawl -> Gemini pipeline.
    2. Write cost_profiles dict to state.
    """
    candidates = state.get("candidate_venues", [])
    intent = state.get("parsed_intent", {})
    group_size = intent.get("group_size", 1)

    if not candidates:
        logger.info("Cost Analyst: no candidates to analyze")
        return {"cost_profiles": {}}

    async def _analyze_all():
        return await asyncio.gather(*[_analyze_venue_cost(v, group_size) for v in candidates])

    try:
        results = asyncio.run(_analyze_all())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        results = asyncio.run(_analyze_all())
    except Exception as exc:
        logger.error("Cost Analyst failed: %s", exc)
        results = [_no_data_fallback(v) for v in candidates]

    cost_profiles = {}
    for venue, result in zip(candidates, results):
        vid = venue.get("venue_id", "")
        cost_profiles[vid] = result

    scored = sum(1 for v in cost_profiles.values() if v.get("base_cost", 0) > 0)
    logger.info("Cost Analyst priced %d/%d venues", scored, len(candidates))

    return {"cost_profiles": cost_profiles}

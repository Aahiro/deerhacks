# System Architecture Documentation

## Project: PATHFINDER

**Goal:** Intelligent, vibe-aware group activity and venue planning with predictive risk analysis and spatial visualization.

---

## 🏗️ Architecture Overview

PATHFINDER is an agentic, graph-orchestrated decision system designed to recommend where groups should go—not just based on availability, but on vibe, accessibility, cost realism, and failure risk.

The system is built around a multi-agent LangGraph workflow, coordinated by a central Orchestrator (the Commander), with Snowflake acting as long-term memory and predictive intelligence.

### Core Design Philosophy

> Move from "What places exist?" → "What will actually work for this group, at this time?"

---

## 🧭 PATHFINDER: Integrated Agentic Workflow

### Node 1: The COMMANDER (Orchestrator Node)

**Status:** ✅ Implemented
- **Intent Parsing:** Uses Gemini 1.5 Flash to extract JSON intent (`activity`, `group_size`, `budget`, etc.).
- **Complexity Tiering:** Maps queries to `tier_1`, `tier_2`, or `tier_3` and assigns active agents and weights based on the tier.
- **Snowflake Pre-Check:** Integrates with `SnowflakeService.cortex_search()` to inject historical risk context.

**Role:** Central brain and LangGraph Supervisor.
**Model:** Gemini 1.5 Flash
**Never calls external APIs directly.**

**Responsibilities:**

- **Intent Parsing:** Converts prompts like
  `"Basketball for 10 people, budget-friendly, west end"`
  into a structured execution state.

- **Complexity Tiering:** Classifies every query into one of three tiers:

  | Tier | Description | Agents Activated | Example |
  |------|-------------|------------------|---------|
  | **Tier 1** — Simple Lookup | Straightforward location query | Fast-path: Scout + light LLM layer | *"Where's a good pizza place nearby?"* |
  | **Tier 2** — Multi-Factor Personal | Group activity with multiple constraints | 3–4 agents (Scout, Cost, Access, Critic ± Vibe) | *"Basketball court for 10 people, budget-friendly"* |
  | **Tier 3** — Strategic / Business | Deep research across all dimensions | All 5 worker agents, deeper analysis | *"Open a bakery in Austin targeting young professionals"* |

- **Dynamic Agent Activation:**
  The Commander decides **which agents to activate per query** based on intent:

  | Query | Agents Activated | Rationale |
  |-------|------------------|-----------|
  | Basketball court rental | SCOUT, COST, ACCESS, CRITIC | Vibe is lower priority for a sports booking |
  | Birthday venue for kids | ALL five agents | Every dimension matters (vibe, cost, access, risk) |
  | Bubble tea shop | ALL five agents | Heavier COST + VIBE weighting |
  | Bakery site selection | ALL five agents | Strategic query — deep research across all dimensions |

- **Dynamic Weighting:**
  Adjusts agent influence in real time:
  - "Cheap" / "budget" → Cost Analyst ↑
  - "Aesthetic" / "vibe" / "cozy" → Vibe Matcher ↑
  - "Outdoor" / "weather" → Critic ↑
  - "Near me" / "transit" → Access Analyst ↑

- **Snowflake Pre-Check:**
  Queries Snowflake Cortex for historical risk patterns (e.g., weather failures, noise complaints) and preemptively boosts the Critic's priority if needed.

**Output:**
A fully weighted execution plan passed into LangGraph.

**Structured Output Schema:**
```json
{
  "parsed_intent": {
    "activity": "basketball",
    "group_size": 10,
    "budget": "low",
    "location": "west end",
    "vibe": null
  },
  "complexity_tier": "tier_2",
  "active_agents": ["scout", "cost_analyst", "access_analyst", "critic"],
  "agent_weights": {
    "scout": 1.0,
    "vibe_matcher": 0.2,
    "access_analyst": 0.8,
    "cost_analyst": 0.9,
    "critic": 0.7
  }
}
```

---

### Node 2: The SCOUT (Discovery Node)

**Status:** ✅ Implemented
- **Dual-API Discovery:** Queries Google Places and Yelp Fusion concurrently via `asyncio.gather()`.
- **Deduplication:** Merges results by normalized name to avoid duplicate venues across sources.
- **Structured Output:** Returns up to 10 enriched candidates with coordinates, ratings, photos, and category metadata.

**Role:** The system's "eyes."
**Tools:** Google Places API, Yelp Fusion

**Responsibilities:**

- Discovers 5–10 candidate venues based on the Commander's intent.

- Collects:
  - Coordinates
  - Ratings & reviews
  - Photos
  - Category metadata

- **Snowflake Enrichment:**
  Immediately enriches candidates with internal intelligence:
  - Past noise complaints
  - Seasonal closures
  - Known operational issues not visible on Maps/Yelp

**Output:**
A shortlist of enriched candidate venues.

**Structured Output Schema:**
```json
{
  "candidates": [
    {
      "venue_id": "gp_abc123",
      "name": "West End Courts",
      "address": "123 King St W, Toronto",
      "lat": 43.6452,
      "lng": -79.3961,
      "rating": 4.3,
      "review_count": 87,
      "photos": ["url1", "url2"],
      "category": "sports_complex",
      "source": "google_places",
      "snowflake_flags": ["seasonal_closure_dec"]
    }
  ]
}
```

---

### Node 3: The VIBE MATCHER (Qualitative Node)

**Status:** ✅ Implemented
- **Multimodal Scoring:** Sends venue photos + metadata to Gemini for aesthetic analysis.
- **Concurrent Processing:** Scores all candidates in parallel via `asyncio.gather()`.
- **Graceful Fallback:** Venues that fail scoring get a neutral `score: null, confidence: 0.0` entry instead of crashing.

**Role:** Aesthetic and subjective reasoning engine.
**Model:** Gemini 1.5 Pro (Multimodal)

**Responsibilities:**

- Analyzes:
  - Venue photos
  - Review sentiment
  - Visual composition

- Matches venues against subjective styles such as:
  - Minimalist
  - Cyberpunk
  - Cozy
  - Dark Academia

- Computes a **Vibe Score** based on:
  - Color palettes
  - Lighting
  - Symmetry
  - Architectural mood

**Output:**
A normalized Vibe Score per venue + qualitative descriptors.

**Structured Output Schema:**
```json
{
  "vibe_scores": {
    "gp_abc123": {
      "score": 0.72,
      "style": "cozy",
      "descriptors": ["warm lighting", "exposed brick", "intimate seating"],
      "confidence": 0.85
    }
  }
}
```

---

### Node 4: The ACCESS ANALYST (Logistics Node)

**Status:** ⬜ Not Yet Implemented (stub — returns empty scores)

**Role:** Spatial reality check.
**Tools:**
- Mapbox Isochrone API
- Google Distance Matrix

**Responsibilities:**

- Computes travel-time feasibility for the entire group.

- Penalizes venues that are:
  - Close geographically
  - Far chronologically (traffic, transit gaps)

- Generates GeoJSON isochrones representing reachable areas.

- **Frontend Integration:**
  GeoJSON blobs are passed directly to the Mapbox SDK.
  Rendered as interactive travel-time overlays on the user's map.

**Output:**
Accessibility scores + map-ready spatial data.

**Structured Output Schema:**
```json
{
  "accessibility_scores": {
    "gp_abc123": {
      "score": 0.81,
      "avg_travel_min": 18,
      "max_travel_min": 32,
      "transit_accessible": true
    }
  },
  "isochrones": {
    "gp_abc123": { "type": "FeatureCollection", "features": ["..."] }
  }
}
```

---

### Node 5: The COST ANALYST (Financial Node)

**Status:** ✅ Implemented
- **Firecrawl Pipeline:** Uses `/map` to discover pricing pages, `/scrape` to extract content as markdown.
- **Gemini Extraction:** Feeds up to 50k chars of scraped content into Gemini 2.5 Flash for structured pricing extraction.
- **Confidence Tiers:** Confirmed pricing keeps Gemini's value_score; estimated pricing caps at 0.5; unknown pricing defaults to 0.3 with uncertainty warnings.

**Role:** "No-surprises" auditor.
**Model:** Gemini 2.5 Flash
**Tools:** Firecrawl

**Responsibilities:**

- **Web Scraping Strategy (Firecrawl + Gemini 2.5 Flash):**
  1. **Semantic Search via Firecrawl `/map`** — Scans the venue's domain utilizing deep semantic search (e.g., matching keywords like "pricing", "rates", "fees", "menu") to discover the most relevant sub-pages.
  2. **Multi-Page Aggregation** — Scrapes the top 3 pricing-related pages along with the venue's **homepage** (since many smaller venues embed prices directly on their main page).
  3. **Holistic LLM Extraction** — Combines up to 50,000 characters of scraped markdown and feeds it into **Gemini 2.5 Flash** to extract complex pricing structures (base entry, gear rentals, taxes) directly into a structured JSON schema.

- Computes **Total Cost of Attendance (TCA)**:
  - Hidden fees
  - Equipment rentals
  - Minimum spends

- **Robust Fallback Systems:**
  Gracefully handles unlisted pricing or interactive booking widgets by estimating rates based on the venue category and passing "High uncertainty" notes to the Critic node.

**Output:**
Transparent, normalized cost profiles per venue.

**Structured Output Schema:**
```json
{
  "cost_profiles": {
    "gp_abc123": {
      "base_cost": 25.00,
      "hidden_costs": [{"label": "2-hr minimum", "amount": 25.00}],
      "total_cost_of_attendance": 50.00,
      "per_person": 5.00,
      "value_score": 0.78,
      "price_trend": "stable"
    }
  }
}
```

---

### Node 6: The CRITIC (Adversarial Node)

**Status:** ✅ Implemented
- **Async API Fetching:** Gathers real-time constraints concurrently using `OpenWeather API` and `PredictHQ API`.
- **Adversarial Reasoning:** Passes the venue details, weather, and events to Gemini to identify dealbreakers.
- **Veto Mechanism:** Evaluates the top 3 candidates and sets the global `veto` flag if the #1 candidate presents a critical risk, triggering a LangGraph retry.

**Role:** Actively tries to break the plan.
**Model:** Gemini (Adversarial Reasoning)
**Tools:** OpenWeather API, PredictHQ

**Responsibilities:**

- Cross-references top venues with real-world risks:
  - Weather conditions
  - City events
  - Road closures

- Identifies dealbreakers:
  - Rain-prone parks
  - Marathon routes
  - Event congestion

- **The Veto Mechanism:**
  If a critical issue is found:
  - Triggers a LangGraph retry
  - Forces the Commander to re-rank candidates

**Output:**
Risk flags, veto signals, and explicit warnings.

**Structured Output Schema:**
```json
{
  "risk_flags": {
    "gp_abc123": [
      {
        "type": "weather",
        "severity": "high",
        "detail": "80% chance of rain Saturday afternoon",
        "source": "openweather"
      }
    ]
  },
  "veto": false,
  "veto_reason": null
}
```

---

### Node 7: SNOWFLAKE (Memory & Intelligence Layer)

**Status:** ✅ Implemented
- **Connection Engine:** Uses `snowflake-connector-python` to establish sessions using configured credentials.
- **Memory Tools:** Implemented `log_risk` and `get_risks` to persist and retrieve historical anomalies.
- **RAG Engine:** Implemented `cortex_search` using `SEARCH_MATCH` to extract relevant context based on queries.

**Role:** Long-term memory and predictive intelligence.

**Functions:**

- **Risk Storage:** Logs historical failures
  (e.g., "Park floods after 5mm rain")

- **RAG Engine:** Snowflake Cortex Search powers:
  - Scout enrichment
  - Critic forecasting

- **Trend Analysis:** Seasonal pricing surges, congestion patterns

**Value Proposition:**
Transforms PATHFINDER from reactive to predictive.

---

## 🎨 Final Synthesis & Output

The Commander collects all node outputs, applies final dynamic weights, and emits a clean JSON response to the frontend.

### The User Receives:

1. **Ranked Top 3 Venues**
   - Displayed as interactive pins on a Mapbox canvas

2. **"Why" & "Watch Out" Cards**
   - Human-readable reasoning
   - Explicit warnings surfaced from the Critic

3. **Spatial Visualization**
   - Travel-time isochrones
   - Feasibility overlays for the entire group

---

## ⚙️ Frontend Integration

### Tech Stack:
- React + Next.js
- Tailwind CSS
- Mapbox SDK

### Map Experience:
- Interactive Mapbox canvas (Google Maps–like UX)
- Pins for ranked venues
- Isochrone overlays for reachability
- Hover & click interactions tied to agent explanations

---

## 🚀 Optional Enhancements

- **Redis:**
  Cache Google/Yelp results for popular queries to reduce cost and latency.

- **FAISS:**
  Local similarity scoring for fast pre-ranking before Snowflake persistence.

- **Auth0 Favorites:**
  Save "High Vibe" locations and feed them back into Commander weight personalization.

### 👥 Crowd Analyst — Social Proof Node (Optional)

Add a dedicated agent for review aggregation, competitor density mapping, and social proof scoring.

**Key Ideas:**
- Aggregate and weight reviews across sources (star ratings, volume, recency, sentiment).
- Demographic-specific filtering (e.g., parent reviews for kid venues).
- Map competitor density — identify underserved zones (valuable for business/strategic queries).
- Compute a normalised **Crowd Score** per venue.

**Why optional:** The Scout already collects ratings/reviews from Google Places & Yelp, and the Vibe Matcher performs sentiment analysis. A dedicated Crowd Analyst adds value for business-tier queries but is not essential for the core pipeline.

**Integration Point:** Sits alongside the other analysts in the fan-out from Scout → Critic.

### 🎙️ ElevenLabs — Voice Interface for the Commander

Add a conversational voice layer so users can speak queries
(e.g. *"find me a birthday venue for 25 kids under $500"*)
and PATHFINDER responds with a natural-sounding voice summary of its recommendations.

**Key Ideas:**
- **Speech-to-Text input** → feeds directly into the Commander's intent parser.
- **Text-to-Speech output** → each agent could have a **distinct voice personality** when presenting its findings:
  - The **Critic** sounds skeptical.
  - The **Vibe Matcher** sounds enthusiastic.
  - The **Cost Analyst** sounds measured and precise.
- Creates a memorable, demo-ready experience.

**Integration Point:** Commander Node (input) + Final Synthesis (output).

---

### 🪙 Solana — Booking & Payment Layer

Add an on-chain micro-payment or escrow system tied to PATHFINDER's venue recommendations.

**Key Ideas:**
- **Deposit lock-in:** When PATHFINDER finds the best venue, users can lock in a deposit via Solana.
- **Tokenised group cost-splitting:** If 10 people are splitting a court rental, a Solana transaction handles instant, near-zero-fee splits.
- **Escrow protection:** Funds are held until the booking is confirmed, providing trust for both sides.

**Integration Point:** Cost Analyst Node → triggers a Solana pay flow on the frontend after the user confirms a venue.

---

### ☁️ Vultr — Cloud Infrastructure

Host the multi-agent system on Vultr's cloud compute for production-grade performance.

**Key Ideas:**
- **GPU-accelerated inference:** If any agent uses GPU-intensive tasks (Gemini calls, embedding generation for semantic search over reviews, etc.), Vultr Cloud GPUs can power that.
- **One-click deployment:** Use Vultr's deployment tooling to spin up the FastAPI backend quickly.
- **Scalable compute:** Scale agent workers independently based on traffic.

**Integration Point:** Infrastructure layer — backend hosting, GPU compute, and deployment pipeline.

---

## 🛠️ Troubleshooting

### 1. "No Results / Empty Map"

**Symptoms:**
- No venue pins appear on the Mapbox canvas.
- Scout returns an empty candidate list.

**Checks:**
- Verify `GOOGLE_PLACES_API_KEY` and `YELP_API_KEY` are set in backend environment variables.
- Confirm the Commander did not over-constrain filters (e.g., strict budget + niche vibe).
- Inspect LangGraph logs to ensure the Scout node executed (not short-circuited by intent classification).

---

### 2. "Map Loads but No Isochrone Overlays"

**Symptoms:**
- Mapbox renders, but no travel-time blobs appear.

**Checks:**
- Ensure `MAPBOX_ACCESS_TOKEN` is valid and scoped for Isochrone API usage.
- Confirm the Access Analyst returned valid GeoJSON.
- Verify the frontend Mapbox layer is added after the map `onLoad` event.
- Check that coordinates are in `[longitude, latitude]` order (Mapbox requirement).

---

### 3. "Results Look Good but Fail in Reality"

**Symptoms:**
- A recommended venue is closed, flooded, or inaccessible.

**Checks:**
- Confirm Snowflake Cortex is reachable and returning RAG context.
- Inspect Critic node execution — ensure veto conditions are not disabled.
- Check PredictHQ quota and response validity for event congestion data.

---

### 4. "High Latency or Timeouts"

**Symptoms:**
- Requests exceed acceptable response times.

**Checks:**
- Ensure Scout and Cost Analyst API calls are parallelized.
- Enable Redis caching for Google Places and Yelp queries.
- Reduce candidate pool size (default: 5–10).
- Confirm LangGraph retry limits are not too permissive.

---

### 5. "Pricing Seems Wrong or Incomplete"

**Symptoms:**
- Users report unexpected fees.

**Checks:**
- Verify Firecrawl selectors are still valid.
- Confirm Cost Analyst is computing Total Cost of Attendance, not just entry price.
- Check Snowflake historical pricing baseline is populated.

---

## 🧠 Model Summary

| Node | Model / Tooling | Purpose |
|------|----------------|---------|
| Commander | Gemini 1.5 Flash | Intent parsing, complexity tiering, dynamic agent activation & weighting |
| Scout | Google Places API, Yelp Fusion | Venue discovery and raw metadata collection |
| Vibe Matcher | Gemini 1.5 Pro (Multimodal) | Aesthetic, photo-based, and sentiment-driven vibe analysis |
| Access Analyst | Mapbox Isochrone API, Google Distance Matrix | Travel-time feasibility and spatial scoring |
| Cost Analyst | Firecrawl + Snowflake Cortex | True cost extraction and pricing anomaly detection |
| Crowd Analyst *(optional)* | Google Places Reviews, Yelp Reviews + Snowflake | Review aggregation, competitor density, social proof scoring |
| Critic | Gemini (Adversarial Reasoning) + OpenWeather, PredictHQ | Failure detection, risk forecasting, veto logic |
| Memory & RAG | Snowflake + Snowflake Cortex Search | Historical risk storage and predictive intelligence |
| Orchestration | LangGraph | Execution order, shared state, conditional retries |
| Frontend Mapping | Mapbox SDK | Interactive maps, pins, isochrone overlays |

---

## 🎯 Demo Strategy

Three pre-tested queries that showcase PATHFINDER's versatility — demonstrating the system isn't a one-trick demo but a genuine **location intelligence platform**.

### Query 1 — Personal / Fun

> *"My friends and I (8 people) want to rent a basketball court this Saturday afternoon in downtown Toronto. Under $200."*

**Expected agent behaviour:**

| Agent | Output |
|-------|--------|
| SCOUT | Map shows 5–6 candidate courts |
| COST ANALYST | Flags: "One court is $25/hr but requires a 2-hour minimum" |
| ACCESS ANALYST | Shows drive-time isochrones from group's central location |
| CRITIC | "This outdoor court has no lights — sunset is at 5:30 PM Saturday" |

**Judge takeaway:** *"Oh, this is useful for regular people."*

---

### Query 2 — Family / Emotional

> *"Birthday party for my daughter turning 7. She loves dinosaurs and painting. 20 kids, budget $400–600, in the Waterloo/Kitchener area."*

**Expected agent behaviour:**

| Agent | Output |
|-------|--------|
| VIBE MATCHER | "This venue offers themed parties including 'Dino Discovery' package" |
| COST ANALYST | "The $450 package includes 20 kids but painting supplies are $8/kid extra — total $610, over budget. Venue B's $500 package includes art supplies." |
| SCOUT | "Venue A: 4.8★ from 200 parent reviews vs. Venue B: 4.1★ from 45" |
| CRITIC | "Venue A's parking lot only fits 12 cars — with 20 kids that's a logistics problem" |

**Judge takeaway:** *"Wait, it handles this too? And the analysis is different?"*

---

### Query 3 — Business / Strategic

> *"I want to open a small bakery in Austin. Targeting young professionals. Budget for lease under $4k/month."*

**Expected agent behaviour:**

| Agent | Output |
|-------|--------|
| SCOUT | Finds available retail spaces; maps competitor bakeries; identifies underserved zones |
| ACCESS ANALYST | Shows foot traffic and transit data |
| COST ANALYST | Compares lease rates against market averages |

**Judge takeaway:** *"This is the same architecture handling wildly different problems."*

> The "wait what" moment is the **versatility**. Judges realise this isn't a one-trick demo — it's a genuine location intelligence platform.

---

## ✅ Quality Control

### 1. Complexity-Based Agent Activation

The Commander classifies each query and only activates the agents that matter. This keeps simple queries fast (Tier 1: ≤2 agents) and complex queries thorough (Tier 3: all 5 agents).

### 2. Structured Output Schemas

Every agent returns **typed JSON** — not free text. This ensures:
- Consistent quality regardless of query domain
- Reliable frontend rendering (no parsing surprises)
- Easy testing and validation

See each node's "Structured Output Schema" section above for the exact shape.

### 3. Pre-Seeded Demo Scenarios

For the live demo, use the three queries documented in the Demo Strategy section. These should be tested extensively beforehand. Let the system handle novel queries, but **don't demo untested queries live** unless confidence is high.

---

## Design Rationale

- **Gemini 1.5 Flash** is used where speed and classification matter.
- **Gemini 1.5 Pro** is reserved for high-value multimodal reasoning (vibe).
- **Snowflake Cortex** ensures the system improves over time instead of repeating failures.
- **LangGraph** enables controlled retries without infinite loops or silent failures.
- **Dynamic agent activation** means simple queries use 3–4 agents efficiently, while complex queries engage all 5 deeply — the activation itself is a demo-worthy feature.

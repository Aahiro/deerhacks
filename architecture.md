# System Architecture Documentation

## Project: PATHFINDER

**Goal:** Intelligent, vibe-aware group activity and venue planning with predictive risk analysis and spatial visualization.

---

## 🏗️ Architecture Overview

PATHFINDER is an agentic, graph-orchestrated decision system designed to recommend where groups should go—not just based on availability, but on vibe, accessibility, cost realism, and failure risk.

The system is built around a multi-agent LangGraph workflow, coordinated by a central Orchestrator (the Commander).

### Core Design Philosophy

> Move from "What places exist?" → "What will actually work for this group, at this time?"

---

## 🧭 PATHFINDER: Integrated Agentic Workflow

### Node 1: The COMMANDER (Orchestrator Node)

**Status:** ✅ Implemented
- **Fallback Heuristics:** Keyword-based intent extraction when LLM services are unavailable.
- **Auth0 + Commander (UPDATED):** Uses Auth0 Universal Login to authenticate users and injects a sanitized "Identity Profile" containing roles and preferences (not raw tokens).
  - **The Flow:** User authenticates → Backend receives `sub`, `roles`, `app_metadata` → Commander consumes sanitized Identity Profile.
  - **The Benefit:** Dynamically adjusts agent weights based on profile metadata (e.g., student = higher cost weight).
  - **Important Rule:** Agents never talk to Auth0 directly — only the Commander consumes identity context.
  - **Identity Profile Schema (NEW):**
    ```json
    {
      "user_id": "auth0|abc123",
      "roles": ["student"],
      "preferences": {
        "budget_sensitivity": 0.9,
        "vibe_sensitivity": 0.4,
        "accessibility_needs": true
      },
      "risk_tolerance": "low",
      "region": "ontario"
    }
    ```

**Role:** Intent parser and weight configurator. LangGraph manages all routing and orchestration.
**Model:** Gemini 1.5 Flash
**Calls Gemini 1.5 Flash directly** for intent parsing and tier classification.

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

- **NEW: OAuth Requirement Detection:**
  Detects whether OAuth-backed actions (like sending emails or checking calendars) are required, and determines the minimum necessary scopes.
  - **Important Rule:** The Commander never touches tokens — it only declares the intent and adds action requirements to the execution plan.

**Output:**
A fully weighted execution plan passed into LangGraph, annotated with OAuth requirements.

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
  "active_agents": ["scout", "cost_analyst", "critic"],
  "agent_weights": {
    "scout": 1.0,
    "vibe_matcher": 0.2,
    "cost_analyst": 0.9,
    "critic": 0.7
  },
  "requires_oauth": true,
  "oauth_scopes": ["calendar.read", "email.send"],
  "allowed_actions": ["send_email", "check_availability"]
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
- Collects coordinates, ratings, reviews, photos, and category metadata.
- **NEW: Price Signal Extraction:** Passes through price metadata from discovery APIs.
  - Google Places `price_level` (0-4) mapped to $–$$$$ symbols.
  - Yelp Fusion `price` field ("$", "$$", etc.).
  - If both APIs provide a price, both are passed downstream. If neither, `price_range` is `null`.

**Scout Output Schema:** (per venue)
```json
{
  "venue_id": "gp_abc123",
  "name": "Example Cafe",
  "rating": 4.4,
  "price_range": "$$",
  "price_source": "google",
  "lat": 43.65,
  "lng": -79.38
}
```

---

### ⚡ Parallel Analyst Execution (UPDATED)

**Nodes Executed in Parallel:**
Once Scout completes, the following agents are launched concurrently using `asyncio.gather()`:
- Node 3 — Vibe Matcher
- Node 4 — Cost Analyst (low priority)
- Node 5 — Critic

**Design Principle:**
Any agent that does not depend on another agent's output must run in parallel. The Commander decides which of these nodes to run and which ones to skip based on the prompt given to it.

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

### Node 4: The COST ANALYST (Price Signal Normalizer)

**Status:** ✅ Implemented
- **Pricing is informational only:** Displays price tiers but does not trigger transactional OAuth flows. Runs in parallel with Vibe and Critic.

**Role:** Price Signal Normalizer
**Model:** None (Heuristic)
**Tools:** None

**Responsibilities:**

- **Price Normalization Strategy:**
  - Normalize $ / $$ / $$$ / $$$$ values from Scout.
  - Resolve conflicts between Google and Yelp price tiers.
  - Assign a confidence label.

- **Conflict Resolution Logic:**
  | Scenario | Outcome |
  |----------|---------|
  | Google + Yelp agree | Use value, confidence = high |
  | Only one source | Use value, confidence = medium |
  | Conflict | Choose median, confidence = low |
  | No data | `price_range`: `null`, confidence = none |

- **Explicit Constraints:**
  - Do NOT scrape websites.
  - Do NOT compute numeric costs.
  - Do NOT trigger OAuth or payments.
  - Must run fully in parallel and never block synthesis.

**Output:**
Price profiles containing normalized ranges and confidence labels.

**Structured Output Schema:**
```json
{
  "price_profiles": {
    "gp_abc123": {
      "price_range": "$$",
      "confidence": "medium",
      "source": "yelp"
    }
  }
}
```

---

### Node 5: The CRITIC (Adversarial Node)

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

- **NEW: Fast-Fail Logic:**
  The Critic can trigger early termination under two conditions:
  - **Condition A — No Viable Options:** (e.g., "Fewer than 3 viable venues after risk filtering")
  - **Condition B — Top Candidate Veto:** (e.g., "Outdoor venue during heavy rain + city marathon")

  **Fast-Fail Behavior:**
  - Skips Cost Analyst results (if still running).
  - Triggers Commander retry (reprompts Node 2 in parallel to search for more options), OR
  - Triggers immediate fallback explanation to the user.

**Output:**
Risk flags, veto signals, and explicit warnings.

**Structured Output Schema:**
```json
{
  "risk_flags": {
    "venue_id": ["risk description", "..."]
  },
  "veto": true,
  "veto_reason": "Outdoor court has no lights — sunset is at 5:30 PM Saturday"
}
```

---

### Node 6: The SYNTHESISER (Final Ranking Node)

**Status:** ✅ Implemented

**Role:** Final aggregator and explainer. Applies dynamic weights from the Commander to all agent scores, computes composite ranked scores, and generates the human-readable `summary`, `why`, and `watch_out` text for each venue.
**Model:** Gemini 1.5 Flash

**Responsibilities:**

- Applies `agent_weights` to vibe, cost, and critic scores → computes a composite ranked score per venue.
- Runs `asyncio.gather()` with Gemini to generate `why` and `watch_out` text concurrently for all candidates.
- **Global Consensus:** After scoring, passes the top remaining venues to Gemini to generate a single comparative summary highlighting the best option based on price consensus, star rating, weather/risks, and vibe.
- **NEW: Chat Reprompting:** Facilitates a conversational loop. If the user provides feedback (e.g., "Do you have more budget-friendly options?"), this node triggers a reprompt back to Node 1 (The Commander) to restart the pipeline with the updated constraints.
- **NEW: OAuth Interaction Layer:**
  - Detects when an action planned by the Commander requires OAuth approval.
  - Explains why the permission is needed in plain language.
  - Triggers Auth0 OAuth or CIBA flows and pauses execution until user approval is granted.
- Emits the final `ranked_results` list consumed by the `/plan` endpoint.

**Note:** Commander, Scout, and Synthesiser always run. Vibe Matcher, Cost Analyst, and Critic are conditionally activated based on `active_agents` from the Commander.

**Structured Output Schema:**
```json
{
  "ranked_results": [
    {
      "rank": 1,
      "name": "Example Cafe",
      "address": "123 Main St, Toronto",
      "lat": 43.65,
      "lng": -79.38,
      "rating": 4.4,
      "price_range": "$$",
      "price_confidence": "medium",
      "vibe_score": 0.72,
      "why": "Affordable pricing with a cozy interior.",
      "watch_out": "Busy on weekends."
    }
  ],
  "global_consensus": "Based on the options, Example Cafe offers the best value ($$) and highest rating, avoiding the rain risk of Option B.",
  "action_request": {
    "type": "oauth_consent",
    "reason": "To email the venue on your behalf",
    "scopes": ["email.send"]
  }
}
```

---

## 🎨 Final Synthesis & Output

The Synthesiser collects all node outputs, applies the Commander's dynamic weights, and emits a clean JSON response to the frontend.

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
- Mapbox SDK

### Map Experience:
- Interactive Mapbox canvas (Google Maps–like UX)
- Ranked pins for venues (gold #1, silver #2, bronze #3, grey rest)
- Isochrone overlays for reachability
- Click interactions: marker or sidebar card → map flies to venue

### Pricing Display (Frontend Contract):
- Price appears under rating: ⭐ 4.4 $$
- If `price_range === null`, price row is hidden.
- No placeholders or inferred pricing.

---

## 🔌 API Reference

### `GET /api/health`
Health check. Returns `{ "status": "ok" }`.

---

### `POST /api/plan`
Synchronous full-pipeline execution. Blocks until all agents complete.

**Request body:** `PlanRequest`
```json
{
  "prompt": "Basketball court for 10 people under $200",
  "group_size": 1,
  "budget": null,
  "location": null,
  "vibe": null,
  "member_locations": []
}
```

**Response:** `PlanResponse`
```json
{
  "venues": [ /* VenueResult[] — see schema below */ ],
  "global_consensus": "Based on the options, Example Cafe offers the best value ($$) and highest rating.",
  "execution_summary": "Pipeline complete."
}
```

---

### `WS /api/ws/plan`
**Primary frontend endpoint.** Streams agent progress events in real time, then emits the final result.

**Client sends (on connect):**
```json
{ "prompt": "...", "member_locations": [] }
```

**Server streams — `progress` events** (one per agent node as it completes):
```json
{
  "type": "progress",
  "node": "scout",
  "label": "Discovering venues..."
}
```

Node → label mapping:
| `node` | `label` |
|--------|---------|
| `commander` | Parsing your request... |
| `scout` | Discovering venues... |
| `vibe_matcher` | Analyzing vibes... |
| `cost_analyst` | Calculating costs... |
| `critic` | Running risk assessment... |
| `synthesiser` | Ranking results... |

**Server sends — `result` event** (once, after all nodes complete):
```json
{
  "type": "result",
  "data": {
    "venues": [ /* VenueResult[] */ ],
    "global_consensus": "Based on the options, Example Cafe offers the best value ($$) and highest rating.",
    "execution_summary": "Pipeline complete."
  }
}
```

---

### `VenueResult` Schema
```json
{
  "rank": 1,
  "name": "Example Cafe",
  "address": "123 Main St, Toronto",
  "lat": 43.65,
  "lng": -79.38,
  "rating": 4.4,
  "price_range": "$$",
  "price_confidence": "medium",
  "vibe_score": 0.72,
  "why": "Affordable pricing with a cozy interior.",
  "watch_out": "Busy on weekends."
}
```

If no price data exists:
- `"price_range": null`
- `"price_confidence": "none"`


---

## 🚀 Optional Enhancements

- **Redis:**
  Cache Google/Yelp results for popular queries to reduce cost and latency.

- **FAISS:**
  Local similarity scoring for fast pre-ranking of candidates.

- **User Favorites:**
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

### 🪙 Solana — Booking & Payment Layer (Planned)

Add an on-chain micro-payment or escrow system tied to PATHFINDER's venue recommendations. **Note: For the current implementation, we are pretending this is implemented and integrated, but it is officially planned for future development.**

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

### 🔐 Auth0 — OAuth Token Vault & Execution Authority

OAuth allows PATHFINDER to move from planning to doing (checking calendars, sending emails, booking venues) — but only with explicit user permission.

**Key Rule:** Agents reason. They do not act. No agent ever accesses OAuth tokens directly.

**How it works (End-to-End Flow):**
1. **Node 1 (Commander):** Detects if an action requires acting on behalf of the user and determines the required scopes (e.g., `email.send`). It never touches tokens.
2. **Node 7 (Synthesizer):** Detects an OAuth requirement, explains to the user *why* permission is needed, and triggers the Auth0 flow. Execution pauses.
3. **Infrastructure (Auth0):** Acts as the system's execution authority. It securely stores tokens, manages refresh/revocation, executes the approved action natively, and then signals LangGraph to resume.

---

### ❄️ Snowflake — Persistence & RAG (Optional)
Move the intelligence layer to a persistent database like Snowflake for long-term risk storage and predictive analysis.
  - **Memory Tools:** Implement `log_risk` and `get_risks` to persist and retrieve historical anomalies.
  - **RAG Engine:** Use Snowflake Cortex Search to power scout enrichment and critic forecasting.
  - **Auth0 FGA Integration:** Ensure fine-grained data access using Auth0 FGA to filter results.
Knowledge memory and long-term risk logging.

- **Historical Logic:** Will pull past results (e.g., *"Park A has been flooded 2 times this month"*).
- **Core Strategy:** Uses Snowflake Cortex Search to power scout enrichment and critic forecasting.
- **Planned: Auth0 Fine-Grained Authorization (FGA):** Ensures agents only retrieve "Risk Data" the specific user is authorized to see (e.g., student vs. admin access levels).

**Role:** Long-term memory.
**Tools:** Snowflake, Snowflake Cortex, Auth0 FGA

---

---

## �🛠️ Troubleshooting

### 1. "No Results / Empty Map"

**Symptoms:**
- No venue pins appear on the Mapbox canvas.
- Scout returns an empty candidate list.

**Checks:**
- Verify `GOOGLE_PLACES_API_KEY` and `YELP_API_KEY` are set in backend environment variables.
- Confirm the Commander did not over-constrain filters (e.g., strict budget + niche vibe).
- Inspect LangGraph logs to ensure the Scout node executed (not short-circuited by intent classification).

---

### 2. "Results Look Good but Fail in Reality"

**Symptoms:**
- A recommended venue is closed, flooded, or inaccessible.

**Checks:**
- Inspect Critic node execution — ensure veto conditions are not disabled.
- Check PredictHQ quota and response validity for event congestion data.

---

### 3. "High Latency or Timeouts"

**Symptoms:**
- Requests exceed acceptable response times.

**Checks:**
- Ensure Scout and Cost Analyst API calls are parallelized.
- Enable Redis caching for Google Places and Yelp queries.
- Reduce candidate pool size (default: 5–10).
- Confirm LangGraph retry limits are not too permissive.

---



## 🧠 Model Summary

| Node | Model / Tooling | Purpose |
|------|----------------|---------|
| Commander | Gemini 1.5 Flash | Intent parsing, complexity tiering, dynamic agent activation & weighting |
| Scout | Google Places API, Yelp Fusion | Venue discovery and raw metadata collection |
| Vibe Matcher | Gemini 1.5 Pro (Multimodal) | Aesthetic, photo-based, and sentiment-driven vibe analysis |
| Cost Analyst | None (Heuristic) | Price signal normalization |
| Crowd Analyst *(optional)* | Google Places Reviews, Yelp Reviews | Review aggregation, competitor density, social proof scoring |
| Critic | Gemini (Adversarial Reasoning) + OpenWeather, PredictHQ | Failure detection, risk forecasting, veto logic |
| Synthesiser | Gemini 1.5 Flash | Composite score ranking, `why`/`watch_out` generation, final output assembly |
| Memory & RAG *(optional)* | Vector Database | Historical risk storage and predictive intelligence |
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
- **LangGraph** enables controlled retries without infinite loops or silent failures.
- **Dynamic agent activation** means simple queries use 3–4 agents efficiently, while complex queries engage all 5 deeply — the activation itself is a demo-worthy feature.

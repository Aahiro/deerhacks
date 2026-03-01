# Idea: 'EquiMeet'

**Theme \- Innovation**  
The best projects don’t just work \- they make people ask “why didn’t this exist before?” That’s innovation. That’s what this weekend is about.

**Think differently. Build boldly. Ship something real.**

**Ideas:**   
**‘**EquiMeet’  
[https://www.youtube.com/watch?v=Lu3QH05z3w4](https://www.youtube.com/watch?v=Lu3QH05z3w4) (inspo)  
Multi-agent location intelligence system ‘Pathfinder’ for Deerhacks V hackathon

**Tech stack:**

**Frontend:**

* [Next.js](http://Next.js)  
* React  
* Tailwind CSS  
* Map box SDK

**Backend:**

* Python  
* FastAPI  
* Auth0 \<- secure payments  
* Langraph  
* Gemini  
* 

**Database:**

* Snowflake  
* 

# Tech Stack

### **LLMs & AI Reasoning**

* **Google Gemini**

  * Commander: intent parsing, complexity tiering, dynamic weighting

  * Vibe Matcher: multimodal image \+ text analysis

  * Critic: adversarial reasoning

* **Snowflake Cortex**

  * RAG over historical risk, pricing, and event data

  * Used primarily by Scout and Critic

---

### **Agent Orchestration**

* **LangGraph**

  * Controls agent execution order

  * Manages shared state

  * Enables conditional routing and retries

---

### **Backend**

* FastAPI (Python)

* Docker (local \+ deployment)

* Auth & user identity: **Auth0**

---

### **Data & Intelligence Layer**

* **Snowflake**

  * Historical weather failures

  * Past noise complaints

  * Seasonal pricing surges

  * Event congestion patterns

* Optional:

  * Redis (API caching)

  * FAISS (similarity search / scoring)

---

### **External APIs**

* Search & discovery:

  * **Google Places API**

  * **Yelp Fusion**

* Logistics:

  * **Mapbox Isochrone API**

  * **Google Distance Matrix**

* Cost & risk:

  * **Firecrawl**

  * **OpenWeather API**

  * **PredictHQ**

---

### **Frontend**

* React

* Vite

* Next.js

* Tailwind CSS  
    
* Mapbox SDK

---

### **Deployment**

* Frontend: Vercel

* Backend: AWS

* Secrets managed via environment variables

# Agent Workflow

## **🧭 PATHFINDER: The Integrated Agentic Workflow**

### **1\. The COMMANDER (Orchestrator Node)**

* **Role:** The project’s central brain. It functions as the **LangGraph Supervisor** that delegates tasks but never searches directly.  
* **Intelligence:** **Gemini 1.5 Flash** (for high-speed classification).  
* **Function:**  
  * **Intent Parsing:** Breaks down the prompt (e.g., "Basketball for 10 people") into a structured state object.  
  * **Dynamic Weighting:** If the query is "Budget-friendly," it boosts the **Cost Analyst’s** influence; if "Wes Anderson vibe," it prioritizes the **Vibe Matcher**.  
  * **Snowflake Pre-Check:** Hits the **Snowflake Knowledge Base** before execution to detect if the query context (e.g., "Saturday in Toronto") has historical risk patterns, pre-emptively raising the **Critic's** priority.

  ### **2\. The SCOUT (Discovery Node)**

* **Role:** The "Eyes" of the system.  
* **Tools:** Google Places API, Yelp Fusion.  
* **Function:** Finds 5–10 candidate locations and gathers raw metadata (coordinates, reviews, photos).  
* **Snowflake Enrichment:** Immediately enriches these candidates by querying **Snowflake** for non-obvious data like past noise complaints or known seasonal closures not listed on Maps.

  ### **3\. The VIBE MATCHER (Qualitative Node)**

* **Role:** The aesthetic analyst.  
* **Tools:** **Gemini 1.5 Pro (Multimodal)**.  
* **Function:** Analyzes venue photos and review sentiment to match subjective styles (e.g., "Cyberpunk," "Minimalist," "Dark Academia").  
* **Output:** Generates a "Vibe Score" based on color palettes, symmetry, and architectural mood.

  ### **4\. The ACCESS ANALYST (Logistics Node)**

* **Role:** The spatial reality check.  
* **Tools:** **Mapbox Isochrone API**, Google Distance Matrix.  
* **Function:** Calculates "Time-to-Destination" polygons for the entire group. It applies feasibility penalties if a location is geographically close but chronologically far due to traffic.  
* **Frontend Integration:** Passes GeoJSON to the **Mapbox SDK** on the frontend to render travel-time "blobs" on the user's map.

  ### **5\. THE COST ANALYST (Financial Node)**

* **Role:** The "No-Surprises" Auditor.  
* **Tools:** Firecrawl  
* **Function:** Scrapes venue websites for "Total Cost of Attendance" (TCA), finding hidden fees like table minimums or equipment rentals.  
* **Snowflake Integration:** Uses **Snowflake Cortex** to compare live prices against historical trends to detect "Misleading Pricing" patterns.

  ### **6\. THE CRITIC (Adversarial Node)**

* **Role:** The skeptic. It exists to find reasons why the current plan will fail.  
* **Tools:** OpenWeather API, PredictHQ.  
* **Function:** Cross-references top picks with real-world context (rain, marathons, road closures).  
* **The Veto:** If it finds a dealbreaker, it triggers a **LangGraph retry**, forcing the Commander to reconsider the next best candidates.

  ### **7\. SNOWFLAKE (Memory & Intelligence Layer)**

* **Role:** The persistent "Long-Term Memory."  
* **Function:**  
  * **Risk Storage:** Logs every venue failure (e.g., "Park flooded last time it rained 5mm").  
  * **RAG Engine:** Powers the Scout and Critic with historical trends via **Snowflake Cortex Search**.  
* **Value:** Turns the system from **Reactive** (what is there?) to **Predictive** (what will go wrong?).  
  ---

  ## **🎨 Final Synthesis: The Result**

The **Commander** re-collects all agent outputs from the graph, applies final weights, and pushes a clean JSON to the **Next.js** frontend.

**The User Receives:**

* **Ranked Top 3:** Rendered as interactive pins on a **Mapbox** canvas.  
* **The "Why" & "Watch Out":** Transparent reasoning cards powered by Gemini, including explicit warnings from the Critic (e.g., "Road closures nearby due to marathon").  
* **Spatial Visualization:** Mapbox overlays showing exactly where the group can reach within their time limit.  
  ---

  ### **➕ Optional Enhancements**

* **Redis:** For API caching to save costs on redundant Google/Yelp calls for popular queries.  
* **FAISS:** For local similarity scoring to handle high-speed ranking before final data is pushed to Snowflake.  
* **Auth0 Favorites:** Allow users to save their "High Vibe" spots to a personal profile, which the Commander uses to personalize future weightings.

**Inspo (map interface):**  
[https://www.youtube.com/watch?v=Lu3QH05z3w4](https://www.youtube.com/watch?v=Lu3QH05z3w4)

**Sponsor API Ideas** 

**Gemini API → VIBE MATCHER agent:** This is the most natural fit. Use Gemini's multimodal capabilities as the engine for your Vibe Matcher — feed it location photos and reviews to analyze aesthetics, ambiance, and "feel." The "Wes Anderson park" example is literally a Gemini demo. You could also use it as the LLM backbone for the Commander's reasoning.

**ElevenLabs → Voice interface for the Commander:** Add a conversational voice layer so users can speak queries ("find me a birthday venue for 25 kids under $500") and PATHFINDER responds with a natural-sounding voice summary of its recommendations. Each agent could even have a distinct voice personality when presenting its findings — the Critic sounds skeptical, the Vibe Matcher sounds enthusiastic, etc. That would be a memorable demo.

**Solana → Booking/payment layer:** Add an on-chain micro-payment or escrow system. When PATHFINDER finds the best venue, users can lock in a deposit via Solana. You could also tokenize group cost-splitting — if 10 people are splitting a court rental, a Solana transaction handles instant, near-zero-fee splits. This ties directly into your Cost Analyst agent.

**Vultr → Cloud infrastructure:** Host your multi-agent system on Vultr's cloud compute. If any agent uses GPU-intensive tasks (Gemini calls, embeddings for semantic search over reviews, etc.), Vultr Cloud GPUs can power that. Use their one-click deployment to spin up your backend. This is your infrastructure layer.

**Snowflake → Data warehouse \+ RAG for the Scout/Critic:** Store and query structured location data (pricing history, review aggregates, weather patterns, event data) in Snowflake. Use their REST API to power a RAG pipeline so your Critic agent can reference historical data — "this venue had noise complaints 3 of the last 5 weekends" or "courts in this area average 40% price hikes on weekends." The 120-day student trial makes this free.

**Auth0 → User authentication \+ saved preferences:** Secure your app with Auth0 for login, and use it to manage user profiles with saved preferences (budget ranges, accessibility needs, group size defaults). If you frame PATHFINDER as a multi-user tool where groups collaborate on a decision, Auth0 handles the multi-user auth cleanly. Their "Auth0 for AI Agents" angle also fits — you're literally securing an AI agent system.

**The demo strategy**: Show one query flowing through all six sponsors' APIs in a single request. User speaks a query (ElevenLabs) → authenticates (Auth0) → Commander orchestrates on Vultr → Scout \+ Critic pull from Snowflake → Vibe Matcher analyzes with Gemini → group splits the booking cost on Solana. That's all six prizes in one coherent flow.

# Architecture

# **System Architecture Documentation**

**Project:** PATHFINDER  
**Goal:** Intelligent, vibe-aware group activity and venue planning with predictive risk analysis and spatial visualization.

---

## **🏗️ Architecture Overview**

PATHFINDER is an **agentic, graph-orchestrated decision system** designed to recommend *where groups should go*—not just based on availability, but on **vibe, accessibility, cost realism, and failure risk**.

The system is built around a **multi-agent LangGraph workflow**, coordinated by a central Orchestrator (the Commander), with Snowflake acting as long-term memory and predictive intelligence.

**Core Design Philosophy:**

Move from *“What places exist?”* → *“What will actually work for this group, at this time?”*

---

## **🧭 PATHFINDER: Integrated Agentic Workflow**

### **Node 1: The COMMANDER (Orchestrator Node)**

**Role:** Central brain and LangGraph Supervisor.  
 **Model:** Gemini 1.5 Flash  
 **Never calls external APIs directly.**

**Responsibilities:**

* **Intent Parsing:** Converts prompts like  
   *“Basketball for 10 people, budget-friendly, west end”*  
   into a structured execution state.

* **Complexity Tiering:** Determines whether the request needs:

  * Quick lookup

  * Full multi-agent evaluation

  * Adversarial re-checks

* **Dynamic Weighting:**  
   Adjusts agent influence in real time:

  * “Cheap” → Cost Analyst ↑

  * “Aesthetic / vibe” → Vibe Matcher ↑

  * “Outdoor” → Critic ↑

* **Snowflake Pre-Check:**  
   Queries Snowflake Cortex for historical risk patterns (e.g., weather failures, noise complaints) and preemptively boosts the Critic’s priority if needed.

**Output:**  
 A fully weighted execution plan passed into LangGraph.

---

### **Node 2: The SCOUT (Discovery Node)**

**Role:** The system’s “eyes.”  
 **Tools:** Google Places API, Yelp Fusion

**Responsibilities:**

* Discovers **5–10 candidate venues** based on the Commander’s intent.

* Collects:

  * Coordinates

  * Ratings & reviews

  * Photos

  * Category metadata

* **Snowflake Enrichment:**  
   Immediately enriches candidates with internal intelligence:

  * Past noise complaints

  * Seasonal closures

  * Known operational issues not visible on Maps/Yelp

**Output:**  
 A shortlist of enriched candidate venues.

---

### **Node 3: The VIBE MATCHER (Qualitative Node)**

**Role:** Aesthetic and subjective reasoning engine.  
 **Model:** Gemini 1.5 Pro (Multimodal)

**Responsibilities:**

* Analyzes:

  * Venue photos

  * Review sentiment

  * Visual composition

* Matches venues against **subjective styles** such as:

  * Minimalist

  * Cyberpunk

  * Cozy

  * Dark Academia

* Computes a **Vibe Score** based on:

  * Color palettes

  * Lighting

  * Symmetry

  * Architectural mood

**Output:**  
 A normalized Vibe Score per venue \+ qualitative descriptors.

---

### **Node 4: The ACCESS ANALYST (Logistics Node)**

**Role:** Spatial reality check.  
 **Tools:**

* Mapbox Isochrone API

* Google Distance Matrix

**Responsibilities:**

* Computes **travel-time feasibility** for the entire group.

* Penalizes venues that are:

  * Close geographically

  * Far chronologically (traffic, transit gaps)

* Generates **GeoJSON isochrones** representing reachable areas.

**Frontend Integration:**

* GeoJSON blobs are passed directly to the Mapbox SDK.

* Rendered as interactive travel-time overlays on the user’s map.

**Output:**  
 Accessibility scores \+ map-ready spatial data.

---

### **Node 5: The COST ANALYST (Financial Node)**

**Role:** “No-surprises” auditor.  
 **Tools:** Firecrawl

**Responsibilities:**

* Scrapes venue websites to compute **Total Cost of Attendance (TCA)**:

  * Hidden fees

  * Equipment rentals

  * Minimum spends

* **Snowflake Cortex Comparison:**  
   Compares live prices against historical trends to detect:

  * Seasonal price spikes

  * Misleading “discounts”

**Output:**  
 Transparent, normalized cost profiles per venue.

---

### **Node 6: The CRITIC (Adversarial Node)**

**Role:** Actively tries to break the plan.  
 **Model:** Gemini (Adversarial Reasoning)  
 **Tools:** OpenWeather API, PredictHQ

**Responsibilities:**

* Cross-references top venues with real-world risks:

  * Weather conditions

  * City events

  * Road closures

* Identifies **dealbreakers**:

  * Rain-prone parks

  * Marathon routes

  * Event congestion

**The Veto Mechanism:**

* If a critical issue is found:

  * Triggers a **LangGraph retry**

  * Forces the Commander to re-rank candidates

**Output:**  
 Risk flags, veto signals, and explicit warnings.

---

### **Node 7: SNOWFLAKE (Memory & Intelligence Layer)**

**Role:** Long-term memory and predictive intelligence.

**Functions:**

* **Risk Storage:** Logs historical failures  
   *(e.g., “Park floods after 5mm rain”)*

* **RAG Engine:** Snowflake Cortex Search powers:

  * Scout enrichment

  * Critic forecasting

* **Trend Analysis:** Seasonal pricing surges, congestion patterns

**Value Proposition:**  
 Transforms PATHFINDER from **reactive** to **predictive**.

---

## **🎨 Final Synthesis & Output**

The Commander collects all node outputs, applies final dynamic weights, and emits a **clean JSON response** to the frontend.

### **The User Receives:**

* **Ranked Top 3 Venues**

  * Displayed as interactive pins on a Mapbox canvas

* **“Why” & “Watch Out” Cards**

  * Human-readable reasoning

  * Explicit warnings surfaced from the Critic

* **Spatial Visualization**

  * Travel-time isochrones

  * Feasibility overlays for the entire group

---

## **⚙️ Frontend Integration**

**Tech Stack:**

* React \+ Next.js

* Tailwind CSS

* Mapbox SDK

**Map Experience:**

* Interactive Mapbox canvas (Google Maps–like UX)

* Pins for ranked venues

* Isochrone overlays for reachability

* Hover & click interactions tied to agent explanations

---

## **🚀 Optional Enhancements**

* **Redis:**  
   Cache Google/Yelp results for popular queries to reduce cost and latency.

* **FAISS:**  
   Local similarity scoring for fast pre-ranking before Snowflake persistence.

* **Auth0 Favorites:**  
   Save “High Vibe” locations and feed them back into Commander weight personalization.

## **🛠️ Troubleshooting**

### **1\. “No Results / Empty Map”**

**Symptoms:**

* No venue pins appear on the Mapbox canvas.

* Scout returns an empty candidate list.

**Checks:**

* Verify `GOOGLE_PLACES_API_KEY` and `YELP_API_KEY` are set in backend environment variables.

* Confirm the Commander did not over-constrain filters (e.g., strict budget \+ niche vibe).

* Inspect LangGraph logs to ensure the Scout node executed (not short-circuited by intent classification).

---

### **2\. “Map Loads but No Isochrone Overlays”**

**Symptoms:**

* Mapbox renders, but no travel-time blobs appear.

**Checks:**

* Ensure `MAPBOX_ACCESS_TOKEN` is valid and scoped for Isochrone API usage.

* Confirm the Access Analyst returned valid GeoJSON.

* Verify the frontend Mapbox layer is added **after** the map `onLoad` event.

* Check that coordinates are in `[longitude, latitude]` order (Mapbox requirement).

---

### **3\. “Results Look Good but Fail in Reality”**

**Symptoms:**

* A recommended venue is closed, flooded, or inaccessible.

**Checks:**

* Confirm Snowflake Cortex is reachable and returning RAG context.

* Inspect Critic node execution — ensure veto conditions are not disabled.

* Check PredictHQ quota and response validity for event congestion data.

---

### **4\. “High Latency or Timeouts”**

**Symptoms:**

* Requests exceed acceptable response times.

**Checks:**

* Ensure Scout and Cost Analyst API calls are parallelized.

* Enable Redis caching for Google Places and Yelp queries.

* Reduce candidate pool size (default: 5–10).

* Confirm LangGraph retry limits are not too permissive.

---

### **5\. “Pricing Seems Wrong or Incomplete”**

**Symptoms:**

* Users report unexpected fees.

**Checks:**

* Verify Firecrawl selectors are still valid.

* Confirm Cost Analyst is computing **Total Cost of Attendance**, not just entry price.

* Check Snowflake historical pricing baseline is populated.

---

## **🧠 Model Summary**

| Node | Model / Tooling | Purpose |
| ----- | ----- | ----- |
| **Commander** | Gemini 1.5 Flash | Intent parsing, complexity tiering, dynamic agent weighting |
| **Scout** | Google Places API, Yelp Fusion | Venue discovery and raw metadata collection |
| **Vibe Matcher** | Gemini 1.5 Pro (Multimodal) | Aesthetic, photo-based, and sentiment-driven vibe analysis |
| **Access Analyst** | Mapbox Isochrone API, Google Distance Matrix | Travel-time feasibility and spatial scoring |
| **Cost Analyst** | Firecrawl \+ Snowflake Cortex | True cost extraction and pricing anomaly detection |
| **Critic** | Gemini (Adversarial Reasoning) \+ OpenWeather, PredictHQ | Failure detection, risk forecasting, veto logic |
| **Memory & RAG** | Snowflake \+ Snowflake Cortex Search | Historical risk storage and predictive intelligence |
| **Orchestration** | LangGraph | Execution order, shared state, conditional retries |
| **Frontend Mapping** | Mapbox SDK | Interactive maps, pins, isochrone overlays |

---

### **Design Rationale**

* **Gemini 1.5 Flash** is used where speed and classification matter.

* **Gemini 1.5 Pro** is reserved for high-value multimodal reasoning (vibe).

* **Snowflake Cortex** ensures the system improves over time instead of repeating failures.

* **LangGraph** enables controlled retries without infinite loops or silent failures.

# Roles

### **Role 1: The Architect & UI Integrator**

**Primary Focus:** System Flow, State Management, and Spatial UI.

* **Orchestration (LangGraph):** Define the State object that flows between agents. Set up the "Commander" node to parse intent and route to other agents.  
* **The Commander:** Implement the Gemini 1.5 Flash logic for complexity tiering and dynamic weighting.  
* **Frontend & Mapbox (Next.js):**  
  * Integrate the Mapbox SDK.  
  * Create the "bridge" that takes the **Access Analyst's** GeoJSON and renders the Isochrone polygons.  
  * Build the "Why" & "Watch Out" UI cards to display agent reasoning.  
* **Integration:** Ensure the FastAPI backend correctly streams LangGraph updates to the Next.js frontend.

### **Role 2: The Data Scout & Aesthetic Lead**

**Primary Focus:** Discovery, Computer Vision, and Scraping.

* **The Scout (Discovery):** Implement the search logic using **Google Places** and **Yelp Fusion**. Handle data normalization (ensuring lat/lng are always consistent).  
* **The Vibe Matcher:** Build the **Gemini 1.5 Pro Multimodal** pipeline. Create the prompt logic that compares user-uploaded "vibe" images against venue photos retrieved by the Scout.  
* **The Cost Analyst (Scraping):** Set up **Firecrawl** to scrape venue websites for hidden fees (rentals, minimum spends).  
* **Deliverable:** A cleaned list of candidate venues enriched with "Vibe Scores" and "Total Cost of Attendance."

### **Role 3: The Risk & Adversary Specialist**

**Primary Focus:** Failure Detection and Real-World Context.

* **The Critic (Adversarial):** Build the "Veto" logic. This person is responsible for making the AI "skeptical."  
* **Real-World APIs:** Integrate **OpenWeather API** and **PredictHQ**.  
  * *Logic:* If venue is "Outdoor" \+ Weather is "Rain" → Trigger Veto.  
  * *Logic:* If venue location \+ PredictHQ Event (e.g., Marathon) → Trigger Veto.  
* **The Access Analyst (Logistics):** Integrate the **Mapbox Isochrone API** and **Google Distance Matrix**. Calculate the "Center of Gravity" for group travel times.  
* **Deliverable:** Risk flags and spatial GeoJSON data that prevents the system from making "stupid" recommendations.

### **Role 4: The Memory & Infrastructure Engineer**

**Primary Focus:** Persistent Intelligence, Security, and Performance.

* **Snowflake Integration:**  
  * Set up **Snowflake Cortex** for RAG.  
  * Build the tables for "Historical Risks" (past noise complaints, weather failures).  
  * Create the "Pre-Check" query for the Commander to detect historical patterns.  
* **Authentication & Safety:** Implement **Auth0** to secure the API and manage "User Favorites."  
* **Backend & Performance:** \* Set up **FastAPI** and **Vultr/AWS** deployment.  
  * (Optional) Implement **Redis** caching for the Scout’s API calls to save budget and decrease latency.  
* **Deliverable:** A high-performance, secure backend where every recommendation is backed by historical data.

---

### **Team Integration Checkpoint**

To make sure everyone’s work fits together, you must agree on the **Shortlist JSON Schema** by the end of the first few hours:

| Agent | Contribution to JSON |
| :---- | :---- |
| **Scout** | name, coords, raw\_photos, rating |
| **Vibe Matcher** | vibe\_score, aesthetic\_tags (e.g., "Industrial") |
| **Access Analyst** | travel\_time\_score, isochrone\_geojson |
| **Cost Analyst** | estimated\_total\_cost, hidden\_fee\_warnings |
| **Critic** | risk\_level (Low/High), veto\_reason |

# .env

\# ─── Google Cloud (Gemini \+ Google Places) ───  
GOOGLE\_CLOUD\_API\_KEY\=\<REDACTED\>

\# ─── Yelp Fusion ───  
YELP\_API\_KEY\=\<REDACTED\>

\# ─── Mapbox ───  
MAPBOX\_ACCESS\_TOKEN\=\<REDACTED\>

\# ─── OpenWeather ───  
OPENWEATHER\_API\_KEY\=\<REDACTED\>

\# ─── PredictHQ ───  
PREDICTHQ\_API\_KEY\=\<REDACTED\>

\# \--- Auth0 \---

AUTH0\_SECRET\=\<REDACTED\>  
AUTH0\_BASE\_URL\=http://localhost:3000  
AUTH0\_DOMAIN\=dev-czwx72m6i5120vzj.us.auth0.com  
AUTH0\_CLIENT\_ID\=\<REDACTED\>  
AUTH0\_CLIENT\_SECRET\=\<REDACTED\>  
AUTH0\_AUDIENCE\=https://api.pathfinder.com

\# \--- Snowflake Credentials \---  
SNOWFLAKE\_ACCOUNT\=\<REDACTED\>  
SNOWFLAKE\_USER\=\<REDACTED\>  
SNOWFLAKE\_PASSWORD\=\<REDACTED\>  
SNOWFLAKE\_DATABASE\=PATHFINDER  
SNOWFLAKE\_SCHEMA\=PUBLIC  
SNOWFLAKE\_WAREHOUSE\=COMPUTE\_WH  
SNOWFLAKE\_ROLE\=ACCOUNTADMIN

\# ─── Redis (optional) ───  
REDIS\_URL\=redis://localhost:6379/0

\# ─── Firecrawl ───  
FIRECRAWL\_API\_KEY\=\<REDACTED\>

Client id:\<REDACTED\>  
Client secret:\<REDACTED\>  


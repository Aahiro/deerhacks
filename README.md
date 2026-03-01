# PATHFINDER

Intelligent, vibe-aware group activity and venue planning with predictive risk analysis and spatial visualization.

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # fill in your API keys
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Architecture

See [architecture.md](./architecture.md) for the full system design.

## Project Structure

```
deerhacks/
├── architecture.md
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py              # FastAPI entry point
│       ├── graph.py             # LangGraph workflow
│       ├── schemas.py           # Pydantic request/response models
│       ├── core/
│       │   └── config.py        # Environment settings
│       ├── agents/
│       │   ├── commander.py     # Node 1 — Orchestrator
│       │   ├── scout.py         # Node 2 — Discovery
│       │   ├── vibe_matcher.py  # Node 3 — Vibe scoring
│       │   ├── cost_analyst.py  # Node 4 — Cost auditing
│       │   └── critic.py        # Node 5 — Adversarial risk
│       ├── services/
│       │   ├── snowflake.py     # Node 7 — Memory & RAG
│       │   └── cache.py         # Redis caching (optional)
│       ├── api/
│       │   └── routes.py        # REST endpoints
│       └── models/
│           └── state.py         # LangGraph shared state
└── frontend/
    ├── package.json
    ├── next.config.js
    ├── public/
    ├── src/
    │   ├── app/
    │   │   ├── layout.js
    │   │   ├── page.js
    │   │   └── globals.css
    │   ├── components/
    │   │   ├── Map.jsx
    │   │   ├── SearchBar.jsx
    │   │   ├── VenueCard.jsx
    │   │   └── IsochroneLayer.jsx
    │   └── lib/
    │       └── api.js
    └── .env.local.example
```

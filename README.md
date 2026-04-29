# AI Business Diagnostic Backend (SaaS-Ready MVP)

This is a modular FastAPI backend for your AI business diagnostic product. It is designed to work now as a consultant/operator tool and scale later into a real SaaS.

## What it includes
- FastAPI API structure
- SQLite persistence (easy to swap to Postgres later)
- SQLAlchemy models
- Deterministic scoring engine (v2-ready shape)
- Claude prompt builder + Anthropic client wrapper
- Assessment and report persistence
- Versioned scoring model + prompt version fields

## Suggested product path
- Stage 1: internal operator tool
- Stage 2: client portal
- Stage 3: multi-tenant SaaS

## Quick start
1. Create a virtual environment
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment file
   ```bash
   cp .env.example .env
   ```
4. Run the API
   ```bash
   uvicorn app.main:app --reload
   ```

## Main endpoints
- `GET /health`
- `POST /api/v1/assessments/score`
- `POST /api/v1/assessments/report`
- `GET /api/v1/assessments/{assessment_id}`

## Notes
- Claude is always called from the backend, never the frontend.
- Scoring is deterministic. Claude explains, but does not decide the core diagnosis.
- Database currently uses SQLite, but the code is structured so you can move to Postgres later.

# Live Demo
## https://risk-scoring-platform.onrender.com/ui/

# Risk Scoring Platform

Production-style fraud risk scoring platform built with FastAPI, PostgreSQL, and scikit-learn.  
It supports:
- real-time transaction scoring
- model versioning and metadata registry
- feature engineering with leakage-safe historical windows
- API + interactive UI

## What This Project Does

Given a `transaction_id` already stored in the database, the platform:
1. Loads transaction + user + merchant context
2. Builds raw + aggregated risk features
3. Runs a trained model to estimate fraud probability
4. Applies a decision threshold to classify fraud vs legit
5. Logs prediction output for monitoring and analysis

## Live Endpoints

- UI: `https://risk-scoring-platform.onrender.com/ui/`
- Health: `https://risk-scoring-platform.onrender.com/api/v1/health`
- OpenAPI Docs: `https://risk-scoring-platform.onrender.com/docs`

## Architecture Overview

Core flow:
- `src/main.py`: app startup, model caching, middleware, route wiring, UI mount
- `src/api/routes/predict.py`: `/predict` and `/predict/batch`
- `src/services/prediction_service.py`: orchestration layer (DB fetch -> features -> model -> log)
- `src/features/feature_pipeline.py`: feature computation pipeline
- `src/training/train.py`: training + evaluation + artifact persistence
- `models/registry.json`: active model registry

UI:
- `src/ui/index.html`
- `src/ui/styles.css`
- `src/ui/app.js`

## API Summary

Base path: `/api/v1`

- `GET /health`: liveness/readiness + DB + model state
- `GET /model-info`: active model metadata + metrics
- `GET /models`: model registry listing
- `POST /predict`: score one transaction
- `POST /predict/batch`: score multiple transactions

Example request:

```json
{
  "transaction_id": "c1212b7d-809e-416a-9983-6001405ac382",
  "include_features": false
}
```

## Local Setup

### 1) Create and activate virtualenv

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements-run.txt
```

### 3) Configure environment

Copy `.env.example` to `.env` and update:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `SECRET_KEY`
- `ENVIRONMENT` / `DEBUG`

### 4) Run app

```powershell
$env:DEBUG='false'
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open:
- UI: `http://127.0.0.1:8000/ui/`
- Docs: `http://127.0.0.1:8000/docs`

## Training & Data

Train model:

```powershell
python scripts/train_model.py
```

Seed data (if needed):

```powershell
python scripts/seed_db.py
```

Smoke test API:

```powershell
python scripts/test_api.py
```

## Deployment (Render)

This repo includes:
- `render.yaml` (Blueprint for web service + free Postgres)
- `DEPLOY_FREE.md` (deployment notes)

Important Render note:
- In production mode, `SECRET_KEY` must not be default.
- `render.yaml` is configured to generate `SECRET_KEY` automatically.

## Repository Notes

- Generated outputs and old artifacts are trimmed to keep deploys and diffs manageable.
- Active model files are kept for runtime startup compatibility.

## License

This project is for educational and demo purposes unless a separate license is added.

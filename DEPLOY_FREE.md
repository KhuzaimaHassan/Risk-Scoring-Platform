# Free Deployment Guide (Render)

This project now includes a Render Blueprint file: `render.yaml`.

It provisions:
- 1 free Python web service (FastAPI + UI at `/ui`)
- 1 free PostgreSQL database

## 1) Push to GitHub

Render deploys from a Git repository.

## 2) Create services on Render

1. Open Render dashboard.
2. Choose **New +** -> **Blueprint**.
3. Connect your repo and select this project.
4. Render will detect `render.yaml` and show:
   - `risk-scoring-platform` (web)
   - `risk-scoring-db` (postgres)
5. Click **Apply**.

## 3) Validate after deploy

Once live, test:
- `https://<your-service>.onrender.com/api/v1/health`
- `https://<your-service>.onrender.com/ui/`

## 4) (Optional) Seed data for demo predictions

The service can start without seed data, but `/predict` needs transactions in DB.

From Render Shell on the web service:

```bash
python scripts/seed_db.py
```

Then test prediction from the UI with a known transaction id.

## Notes

- Free instances can sleep after inactivity, so first request may be slow.
- Keep `models/` committed or otherwise available at runtime because startup loads the latest model from local `models/registry.json`.
- If you rotate credentials or recreate DB, Render will re-inject env vars automatically from `render.yaml`.

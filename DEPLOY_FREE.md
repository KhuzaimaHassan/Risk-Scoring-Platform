# Deploy the demo on Render Free

Last reviewed: **2026-11-01**. FastAPI serves the backend and built React UI from one web service, using an external PostgreSQL database.

## Existing deployment

| Setting | Current demo |
| --- | --- |
| Render service | `risk-scoring-platform-sg` |
| Region / compute | Singapore / Free ($0/month, 0.1 CPU, 512 MB) |
| Branch | `main` |
| UI | <https://risk-scoring-platform-sg.onrender.com/ui/> |
| Readiness | <https://risk-scoring-platform-sg.onrender.com/api/v1/health> |
| API docs | <https://risk-scoring-platform-sg.onrender.com/docs> |
| Database | External Supabase PostgreSQL in Tokyo; session pooler, port 5432 |
| Background monitoring | Disabled (`MONITORING_ENABLED=false`) |

Update the existing Singapore service for future releases. Creating another Blueprint is unnecessary and can create duplicate services. The former Oregon URL is no longer canonical. Supabase billing is separate; working database connectivity does not verify that account's billing plan.

## Fresh installation

Use this section only when you intentionally need a new installation.

1. Prepare an external PostgreSQL database with the application tables and transactions. Startup does not initialize, migrate, or seed data.
2. In Render, choose **New → Web Service**, connect this repository, and select `main`.
3. Choose a region close to the database and select **Free**. Keep a single process/instance for this demo.
4. Set the commands and environment below, then deploy.

Alternatively, `render.yaml` creates one Free web service named `risk-scoring-platform`, with manually supplied external database values. It creates **no Render Postgres database**. Its region is unspecified; choose an appropriate region during setup. Supply `MONITORING_ENABLED=false` for a simple demo. Applying a new Blueprint does not migrate the existing Singapore service.

### Commands

| Field | Value |
| --- | --- |
| Python / Node versions | `3.13.0` / `22.14.0` |
| Build | `pip install -r requirements-run.txt && npm --prefix frontend ci && npm --prefix frontend run build` |
| Start | `python -m uvicorn src.main:app --host 0.0.0.0 --port $PORT` |
| Health check | `/api/v1/health` |

The frontend is served at `/ui/`; it needs no separate static-site service.

### Environment

| Variable | Value |
| --- | --- |
| `ENVIRONMENT` | `production` |
| `DEBUG`, `DB_ECHO` | `false` |
| `LOG_LEVEL` | `INFO` |
| `ADMIN_USERNAME` | Your administrator username; default `admin` |
| `ADMIN_PASSWORD` | Unique password: at least 12 characters, at most 72 UTF-8 bytes; use generated ASCII |
| `SECRET_KEY` | Separate randomly generated signing secret |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Exact provider connection details |
| `DB_PREPARED_STATEMENT_CACHE_SIZE` | `100` for the current session pooler; `0` for a transaction-mode pooler |
| `MONITORING_ENABLED` | `false` for this hosted demo |
| `RETRAINING_AUTO_PROMOTE` | `false` unless deliberately enabling guarded promotion |
| `ALLOWED_ORIGINS` | `[]` for the same-origin UI; explicit JSON origins for a separate frontend |

Render injects `PORT` and `RENDER_GIT_COMMIT`. The Blueprint includes `API_HOST=0.0.0.0` and `API_PORT=10000`, but the start command binds Uvicorn using Render's `$PORT`.

Use Render's generation controls or a password manager, and save credentials privately. There are no default admin credentials; invalid production configuration prevents startup. Changing `ADMIN_PASSWORD` or `SECRET_KEY` invalidates previous tokens after deployment.

For Supabase, copy the exact connection details from its connection panel. Session mode on port 5432 supports prepared statements; transaction mode on 6543 does not. Set this app's prepared-statement cache size to `0` with a transaction pooler. The demo uses session mode. See [Supabase connection modes](https://supabase.com/docs/guides/database/connecting-to-postgres).

## Validate a deployment

1. Wait for the intended commit to show **Live**. Free-instance startup can take a minute or longer; initial requests can time out.
2. Require `/api/v1/health` to return HTTP 200, `status: "ok"`, `model_loaded: true`, and `db_connected: true`. Degraded readiness returns 503. `/api/v1/live` checks only the process.
3. Compare `X-App-Revision` with the intended commit and check the expected model version.
4. Sign in at `/ui/`, select a stored transaction, score it, and compare the result/feed and USD amount. Reload to verify persistence.
5. Check the model registry and review queue. Review acknowledgement does not label fraud or enforce payment decisions.
6. Confirm **Free** compute and inspect workspace billing/usage.

Run the authenticated smoke script from a trusted machine with that service's credentials:

```powershell
python scripts/test_api.py --base-url https://risk-scoring-platform-sg.onrender.com
```

Supply credentials privately through the environment or uncommitted `.env`; never paste secrets into shell history. The script creates prediction records but never seeds/resets data. It requires a populated database. Run it after startup, which can exceed its 45-second timeout.

## Persistence and data safety

- Database storage is external to Render. Deleting an obsolete web service is separate from deleting the shared database; check identities before cleanup.
- Never run `scripts/seed_db.py` against hosted or valuable data. It drops/recreates tables and is blocked in production. The HTTP reset route is removed.
- Free web services have ephemeral filesystems and no persistent disks. Runtime-generated model artifacts, registry edits, and reports disappear on restart, sleep, or redeploy.
- Include reviewed artifacts, matching metadata, and `models/registry.json` in releases. Generated model files are ignored by Git unless explicitly curated.
- The optional scheduler cannot run continuously on sleeping Free instances. The Singapore demo disables it. Monitoring is not a separate deployed service.
- Offline registry promotion requires deployment/restart to load the new bundle. The guarded background path can refresh one process; it does not coordinate multiple workers.

## Free limits and spending

Provider limits reviewed on 2026-11-01:

- Free web services sleep after 15 minutes without inbound traffic. Waking normally takes about a minute; startup requests can time out.
- Each workspace shares 750 Free instance hours per calendar month. Running duplicates consume that allowance; sleeping services do not. Exhaustion suspends Free web services until the next month.
- Hobby includes 5 GB outbound bandwidth and 500 Starter pipeline minutes monthly, shared across the workspace.
- A **$0 pipeline spending limit** stops further builds after included minutes are exhausted. Existing deployments continue running. This cap does **not** cap bandwidth charges.
- With a payment method attached, excess bandwidth can incur charges even on Free compute. A card on file does not change the compute plan; $0 compute does not guarantee a $0 invoice.
- Separately created Free Render Postgres databases expire after 30 days. This Blueprint creates none; the demo uses Supabase.

Inspect **Billing → Monthly Included Usage**, **Unbilled Charges**, and **Settings → Build Pipeline**. Historical billing snapshots are not current spending guarantees. Check Supabase's plan/quotas separately.

References: [Render Free](https://render.com/docs/free), [bandwidth](https://render.com/docs/outbound-bandwidth), [build pipeline](https://render.com/docs/build-pipeline), [Blueprint reference](https://render.com/docs/blueprint-spec).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| First request times out | Free startup; wait for Live/readiness and retry |
| Readiness 503 | Database credentials/connectivity and artifact/metadata/schema compatibility |
| Login 401 / 429 | Configured credentials / temporary failed-login throttle |
| Login 503 locally | Missing `ADMIN_PASSWORD` |
| Score 404 / 422 | Missing stored transaction / invalid UUID or body |
| Database 503 | Dependency unavailable; inspect logs with the request ID |
| Empty chart | No predictions in the last 30 minutes; samples are not fabricated |
| Slow scoring | `Server-Timing`, database region/round trips, and Free CPU |

Live warm scoring on 2026-10-31 was 655–955 ms (887 ms mean). The under-50 ms target remains unmet; inference-only timing is not total scoring latency.

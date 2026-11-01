# Local setup

[Project overview](../README.md)

Use Python **3.13**, Node.js **22.14**, npm, and PostgreSQL. CI uses PostgreSQL 18. Run commands from the repository root; examples use PowerShell.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
npm --prefix frontend ci
```

Edit `.env` before starting:

- Set `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD` for your own database.
- Set `ADMIN_USERNAME` and a unique `ADMIN_PASSWORD` of at least 12 characters and at most 72 UTF-8 bytes. Use a generated ASCII password to avoid byte-length ambiguity.
- Replace `SECRET_KEY` with a separate randomly generated secret.
- Keep `DEBUG=false`. Use `ENVIRONMENT=development` for local development.
- Keep `MONITORING_ENABLED=false` for a demo with no background training jobs.

Keep credentials out of Git and screenshots. Without `ADMIN_PASSWORD`, local sign-in returns 503; production startup requires it.

### Prepare local data

Scoring needs the expected tables and stored transactions. The app does not initialize or seed them on startup, and there is no HTTP seed/reset route.

For a **new, disposable local database only**, create an empty database and matching role using PostgreSQL administration tools. Confirm `.env` points to that disposable database, then generate synthetic data:

```powershell
python scripts/seed_db.py --users 20 --merchants 5 --transactions 400
```

**The seeder drops and recreates application tables, deleting existing data.** It refuses `ENVIRONMENT=production`; never change a hosted environment to bypass that guard. New synthetic data does not reproduce the committed model's evaluation dataset.

### Run

```powershell
npm --prefix frontend run build
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open [the local dashboard](http://127.0.0.1:8000/ui/) or [local API docs](http://127.0.0.1:8000/docs). Sign in with your configured credentials and select a stored transaction.

For frontend development, keep the API running and run `npm --prefix frontend run dev`. Vite proxies `/api` to `http://127.0.0.1:8000`. See [frontend instructions](FRONTEND.md).

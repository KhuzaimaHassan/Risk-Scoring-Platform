# Risk Platform frontend

React dashboard using Vite, Recharts, Lucide icons, and plain CSS. See the [root README](../README.md) for backend/database setup and [API contracts](../docs/API.md).

## Develop

Use Node.js 22.14. From the repository root:

```powershell
npm --prefix frontend ci
npm --prefix frontend run dev
```

Keep FastAPI running at `http://127.0.0.1:8000`. Open Vite's printed URL; `vite.config.js` proxies `/api` to the backend. Sign in using the configured `ADMIN_USERNAME` and `ADMIN_PASSWORD`. Credentials are not built into the frontend.

## Check and build

```powershell
npm --prefix frontend run lint
npm --prefix frontend run build
npm --prefix frontend audit --audit-level=moderate
```

FastAPI serves `frontend/dist` at `/ui/`, including on Render. Rebuild after source changes; Render rebuilds these assets during deployment.

## Behavior

- The JWT is held in `sessionStorage` for the browser tab and removed on sign-out or an authenticated API 401 response.
- Data refreshes every 30 seconds, with manual Refresh. Errors and empty states are visible.
- Scores use a 0–100 scale, matching `risk_score`. Monetary values use normalized USD.
- KPIs count distinct scored transactions using their latest predictions. Repeated scoring adds audit entries without increasing that distinct count.
- The chart shows up to 200 predictions in the last 30 minutes; the feed shows the latest 10 prediction records.
- Manual scoring uses a stored UUID or dropdown choice; it does not create transactions.
- Registry data comes from actual model versions/metrics. Review acknowledgement records inspection only.

[Live dashboard](https://risk-scoring-platform-sg.onrender.com/ui/). Free startup delays and the model/latency limitations in the root README apply.

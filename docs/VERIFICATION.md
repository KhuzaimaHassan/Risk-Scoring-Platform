# Verification

[Project overview](../README.md)

```powershell
python -m pytest tests -q
npm --prefix frontend run lint
npm --prefix frontend run build
npm --prefix frontend audit --audit-level=moderate
```

The PostgreSQL integration check requires `INTEGRATION_DATABASE_URL` pointing to an isolated test database; otherwise that check is skipped. CI provisions PostgreSQL 18 and runs the full suite plus frontend lint/build and the advisory check. The last full code verification before these documentation changes passed 58 tests.

With the local API running and credentials configured:

```powershell
python scripts/test_api.py --base-url http://127.0.0.1:8000
```

The smoke script never seeds or resets data. It creates prediction records while testing single/batch scoring; use a test database. It prints latency/stage samples, but never credentials.

For the recorded live checks and their scope, see [performance and known limitations](PERFORMANCE.md).

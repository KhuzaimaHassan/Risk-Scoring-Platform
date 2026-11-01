# API behavior

[Project overview](../README.md)

Base path: `/api/v1`. The running service's [OpenAPI docs](https://risk-scoring-platform-sg.onrender.com/docs) provide full request/response schemas. UI and API share one origin in the demo.

## Routes

| Method and path | Authentication | Purpose |
| --- | --- | --- |
| `GET /live` | Public | Process liveness and deployed revision |
| `GET /health` | Public | Database/model readiness; 503 when degraded |
| `GET /model-info` | Public | Loaded model metadata, features, metrics, threshold |
| `GET /models` | Public | File-backed registry versions and stages |
| `POST /token` | Username/password form | Issue administrator JWT |
| `POST /predict` | Bearer JWT | Score one stored transaction and commit prediction log |
| `POST /predict/batch` | Bearer JWT | Score stored transactions, returning per-item errors |
| `GET /dashboard/stats` | Bearer JWT | Stored counts, distinct scored counts, flagged/confirmed fraud, scored USD volume |
| `GET /dashboard/chart-data` | Bearer JWT | Up to 200 predictions in the last 30 minutes |
| `GET /dashboard/recent-transactions` | Bearer JWT | Latest 10 prediction records and USD amounts |
| `GET /dashboard/transactions` | Bearer JWT | Stored UUID choices; `limit` defaults to 20, range 1–100 |
| `GET /dashboard/review-queue` | Bearer JWT | Up to 50 flagged, unreviewed latest predictions |
| `POST /dashboard/reviews/{log_id}` | Bearer JWT | Acknowledge inspection with administrator notes |

There is no seed/reset route. API documentation does not grant access to protected operations.

`POST /token` accepts `application/x-www-form-urlencoded` fields `username` and `password`, not JSON environment-variable names. Send the result as `Authorization: Bearer <access_token>`. Expiry defaults to 24 hours. There is one configured administrator, not a user-management/role system. Failed-login throttling is in-process rather than distributed.

## Single scoring

```json
{
  "transaction_id": "fbdd18de-ce99-4466-8a0f-ab5127071e88",
  "include_features": false
}
```

The UUID must exist in the configured database. One SELECT loads context and completed history from `[transaction_time - 30 days, transaction_time)`. Historical fraud features use labels only when `labelled_at` precedes the scored event. The service constructs features, runs its loaded model/threshold bundle, and commits the prediction before success.

| Response field | Meaning |
| --- | --- |
| `fraud_probability` | Model output, 0–1; not a guarantee of calibrated real-world probability |
| `risk_score` | Rounded probability × 100, on a 0–100 scale |
| `is_fraud` | Threshold classification, not independently confirmed fraud |
| `decision_threshold` | Serving model's threshold |
| `model_version` | Exact serving bundle used |
| `log_id` | Actual persisted prediction identifier |
| `feature_snapshot` | Optional features when requested |
| `risk_context` | Explanation and advisory action; no payment action is executed |
| `latency_ms` | Context, features, inference, and audit commit duration |

Audit failure cannot yield a fabricated log ID or successful score response. Repeated scoring creates separate logs; distinct dashboard KPIs use the latest prediction per transaction. Review acknowledgements and later reconciliation can update review/outcome fields.

## Batch scoring

```json
{
  "transaction_ids": [
    "fbdd18de-ce99-4466-8a0f-ab5127071e88",
    "00000000-0000-4000-8000-000000000000"
  ],
  "include_features": false
}
```

A valid batch can return HTTP 200 with both `results` and `errors`. Inspect `total_scored` and each error's status; HTTP 200 does not mean every item succeeded. The default maximum batch size is 500. A malformed UUID is a request-validation error, distinct from an absent but valid UUID.

## Timing headers

- `X-Process-Time-Ms`: total server request duration; distinct from client-network elapsed time.
- `X-App-Revision`: Render commit SHA, or `local` outside Render.
- `X-Request-ID`: correlation ID for logs.
- Single-scoring `Server-Timing`: `context_db`, `features`, `inference`, and `audit` in milliseconds. Audit includes commit.

Do not compare inference-only timing with an end-to-end requirement. Network and cold starts add client-visible delay. Warm scoring remains above the 50 ms target.

## Errors

| Status | Typical cause |
| --- | --- |
| 401 | Missing/expired/invalid JWT, or incorrect login credentials |
| 404 | Transaction or prediction not found |
| 422 | Invalid UUID, request body, or bounded input |
| 429 | Failed-login throttle reached |
| 503 | Missing model, degraded readiness, database unavailable, or unconfigured local sign-in |
| 500 | Unexpected failure; details stay in server logs |

Server errors are sanitized, with request IDs where available. Validation/not-found messages can still contain technical field/table names. Keep passwords and tokens out of reports/screenshots.

## Review semantics

`POST /dashboard/reviews/{log_id}` accepts a JSON `notes` string of at most 2,000 characters. It marks the prediction reviewed and attributes notes to the administrator. It does not label the transaction as fraud, reconcile outcomes, or send actions to a payment provider.

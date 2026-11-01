# Features

[Project overview](../README.md)

- JWT-protected single and batch scoring for stored transaction UUIDs.
- A shared 31-feature training/serving contract using completed transactions from the preceding 30 days. Historical aggregates exclude the current transaction, equal timestamps, future events, and labels unavailable at event time.
- A versioned Random Forest, validation-selected threshold, startup compatibility/quality checks, and an in-memory serving bundle.
- Durable prediction logging: success requires the audit commit to succeed. Records include the score, feature snapshot, model version, and latency. Review/outcome fields can subsequently be updated; this is not a tamper-proof audit system.
- Real dashboard data: distinct scored-transaction KPIs, normalized USD amounts, recent predictions, a 30-minute score chart, model registry, and review queue. Data refreshes every 30 seconds.
- Explicit sign-out, session-scoped browser tokens, visible validation errors, sanitized server failures, and request IDs.
- Readiness returns HTTP 503 when the database or serving model is unavailable.
- Optional ground-truth reconciliation and guarded retraining. **Background monitoring is disabled on the Singapore demo**; automated retraining is not an active hosted capability.

Scores and recommended actions are advisory. The app does not execute payments, block transactions, or confirm fraud by acknowledging a review.

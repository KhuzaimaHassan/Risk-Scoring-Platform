# Performance and known limitations

[Project overview](../README.md)

Live browser smoke checks on **2026-10-31** passed login, database/model readiness, scoring, persisted results, score/feed agreement, invalid-input handling, model registry, and the empty review queue. Six predictions were made for one existing transaction; this was not a load test.

| Measurement | Result |
| --- | --- |
| First scoring request after startup | 1,095 ms |
| Five subsequent scoring requests | 655–955 ms; mean 887 ms |
| Target | Under 50 ms — **not achieved** |
| Cold startup | Initial requests timed out while the Free instance woke up |

Displayed scoring latency includes context loading, feature computation, inference, and the durable audit commit. It excludes client-network time and dashboard refreshes. See [API timing headers](API.md).

The current model, `v20261029_production`, uses feature schema version 2 and threshold `0.15826714653647353`. Its stored chronological test evaluation is:

| Metric | Result |
| --- | --- |
| Test transactions / fraud examples | 815 / 20 |
| ROC AUC / PR AUC | 0.808 / 0.339 |
| Precision / recall / F1 | 0.250 / 0.500 / 0.333 |
| True positives / false negatives | 10 / 10 |
| False positives / true negatives | 30 / 765 |

These are limited demo results, not evidence of production readiness. Model quality needs improvement, the evaluation set is small, and the existing evaluation has informed development decisions. Stronger claims require additional representative data and a fresh independent evaluation. See [model metadata](../models/metadata_v20261029_production.json) and the [evaluation report](../reports/eval_v20261029_production.json).

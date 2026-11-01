# Training and model lifecycle

[Project overview](../README.md)

```powershell
python scripts/train_model.py --version v_local_candidate
```

The default training configuration uses chronological train/validation/test partitions, past-only features, and validation-based threshold selection. The command saves a candidate/report without promotion. Review the data, metrics, and quality gates before promoting any candidate.

The optional background path reconciles labels and evaluates recent resolved outcomes for the serving model. Sample/cooldown conditions govern retraining. With `RETRAINING_AUTO_PROMOTE=true`, promotion also requires quality gates and improvement against the incumbent on identical evaluation rows. The serving bundle is refreshed with registry rollback on load failure. This is a single-process mechanism, not a distributed rollout system.

Render Free storage is ephemeral: runtime-trained artifacts disappear on restart/redeploy unless saved externally and deliberately included in a release. Committed curated artifacts provide the demo's restart-safe baseline.

See [current model metrics and evaluation limitations](PERFORMANCE.md) and [deployment persistence](DEPLOYMENT.md).

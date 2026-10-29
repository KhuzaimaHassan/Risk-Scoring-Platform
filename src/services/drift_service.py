"""Ground-truth reconciliation and guarded, version-specific retraining."""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, func
from config.settings import get_settings
from src.database.session import AsyncSessionFactory
from src.database.models import PredictionLog, PredictionOutcome, FactTransaction
from src.training.train import train, load_train_config
from src.services.model_state import load_bundle

logger = logging.getLogger(__name__)

async def reconcile_ground_truth():
    now = datetime.now(timezone.utc)
    async with AsyncSessionFactory() as db:
        rows = (await db.execute(select(PredictionLog, FactTransaction.fraud_label).join(
            FactTransaction, PredictionLog.transaction_id == FactTransaction.transaction_id).where(
            PredictionLog.outcome == PredictionOutcome.UNKNOWN, FactTransaction.fraud_label.is_not(None),
            FactTransaction.labelled_at <= now).limit(1000))).all()
        for prediction, label in rows:
            prediction.outcome = {(1, 1): PredictionOutcome.TRUE_POSITIVE, (0, 0): PredictionOutcome.TRUE_NEGATIVE,
                                  (1, 0): PredictionOutcome.FALSE_POSITIVE, (0, 1): PredictionOutcome.FALSE_NEGATIVE}[(prediction.risk_label, label)]
            prediction.outcome_resolved_at = now
        await db.commit()
    return len(rows)

async def check_drift_and_retrain(app):
    settings = get_settings()
    bundle = getattr(app.state, "model_bundle", None)
    if bundle is None:
        return
    try:
        await reconcile_ground_truth()
        now = datetime.now(timezone.utc)
        last = app.state.last_retraining_attempt
        if last and now - last < timedelta(hours=settings.retraining_cooldown_hours):
            return
        async with AsyncSessionFactory() as db:
            counts = dict((await db.execute(select(PredictionLog.outcome, func.count()).where(
                PredictionLog.model_version == bundle.version,
                PredictionLog.created_at >= now - timedelta(hours=settings.monitoring_window_hours),
                PredictionLog.outcome != PredictionOutcome.UNKNOWN).group_by(PredictionLog.outcome))).all())
            tp = counts.get(PredictionOutcome.TRUE_POSITIVE, 0)
            fp = counts.get(PredictionOutcome.FALSE_POSITIVE, 0)
            fn = counts.get(PredictionOutcome.FALSE_NEGATIVE, 0)
            if tp + fp + fn < 30 or 2 * tp / max(2 * tp + fp + fn, 1) >= settings.performance_f1_min_threshold:
                return
            samples = await db.scalar(select(func.count()).select_from(FactTransaction).where(
                FactTransaction.fraud_label.is_not(None), FactTransaction.labelled_at < now))
            if samples < settings.retraining_min_samples:
                return
        app.state.last_retraining_attempt = now
        cfg = load_train_config()
        cfg.train_until = now
        cfg.auto_promote = False
        cfg.model_artifact_dir = settings.model_artifact_dir
        cfg.report_dir = settings.model_report_dir
        cfg.min_f1, cfg.min_recall = settings.model_min_f1, settings.model_min_recall
        result = await asyncio.to_thread(train, cfg)
        if not settings.retraining_auto_promote or not result["quality_gate_passed"]:
            return
        # Re-evaluate the incumbent on the candidate's identical holdout, never compare different datasets.
        from src.training.evaluate import compute_binary_metrics
        def incumbent_metrics():
            X_test, y_test = result["_evaluation_data"]
            return compute_binary_metrics(y_test.to_numpy(), bundle.model.predict_proba(X_test)[:, 1], bundle.threshold)
        incumbent = await asyncio.to_thread(incumbent_metrics)
        if result["metrics"]["f1"] <= incumbent["f1"] or result["metrics"]["recall"] < incumbent["recall"]:
            return
        from src.training.model_registry import ModelRegistry
        registry = ModelRegistry(settings.model_artifact_dir)
        registry.promote_model(result["version"], promoted_by="guarded-retraining")
        try:
            app.state.model_bundle = await asyncio.to_thread(load_bundle, settings.model_artifact_dir)
        except Exception:
            registry.promote_model(bundle.version, promoted_by="reload-rollback")
            raise
        logger.info("Serving retrained model %s", result["version"])
    except Exception:
        logger.exception("Monitoring/retraining failed; current serving model preserved")

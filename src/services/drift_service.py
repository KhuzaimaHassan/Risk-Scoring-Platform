"""
src/services/drift_service.py
------------------------------
Background service for reconciling ground truth, computing F1 scores over
recent predictions, and triggering automated retraining if performance drops.
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update, func, Float, case
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import AsyncSessionFactory
from src.database.models.prediction_log import PredictionLog, PredictionOutcome
from src.database.models.fact_transaction import FactTransaction
from src.training.train import train, TrainConfig

logger = logging.getLogger(__name__)

async def reconcile_ground_truth():
    """
    Match prediction logs with their ground-truth fraud labels from fact_transaction.
    If a transaction has been updated with a fraud_label, update the PredictionLog outcome.
    """
    try:
        async with AsyncSessionFactory() as db:
            # 1. Find all PredictionLogs where outcome is UNKNOWN
            stmt = select(PredictionLog, FactTransaction.fraud_label).join(
                FactTransaction, PredictionLog.transaction_id == FactTransaction.transaction_id
            ).where(
                PredictionLog.outcome == PredictionOutcome.UNKNOWN,
                FactTransaction.fraud_label.is_not(None)
            )
            
            result = await db.execute(stmt)
            rows = result.fetchall()
            
            if not rows:
                return 0

            updated_count = 0
            for pred_log, ground_truth in rows:
                pred_label = pred_log.risk_label
                if pred_label == 1 and ground_truth == 1:
                    outcome = PredictionOutcome.TRUE_POSITIVE
                elif pred_label == 0 and ground_truth == 0:
                    outcome = PredictionOutcome.TRUE_NEGATIVE
                elif pred_label == 1 and ground_truth == 0:
                    outcome = PredictionOutcome.FALSE_POSITIVE
                else:
                    outcome = PredictionOutcome.FALSE_NEGATIVE
                
                pred_log.outcome = outcome
                pred_log.outcome_resolved_at = datetime.now(timezone.utc)
                updated_count += 1
                
            await db.commit()
            if updated_count > 0:
                logger.info("Reconciled %d predictions with ground truth.", updated_count)
            return updated_count
    except Exception as e:
        logger.error("Failed to reconcile ground truth: %s", str(e))
        return 0

async def check_drift_and_retrain():
    """
    Calculate the F1 score for predictions over the last 24 hours.
    If the F1 score falls below 0.70 (or there are many False Positives/Negatives),
    trigger the model retraining pipeline.
    """
    try:
        # Step 1: Reconcile any missing ground truth first
        await reconcile_ground_truth()
        
        async with AsyncSessionFactory() as db:
            # Look at predictions from the last 24 hours
            since_time = datetime.now(timezone.utc) - timedelta(hours=24)
            
            stmt = select(
                func.sum(case((PredictionLog.outcome == PredictionOutcome.TRUE_POSITIVE, 1), else_=0)).label("tp"),
                func.sum(case((PredictionLog.outcome == PredictionOutcome.FALSE_POSITIVE, 1), else_=0)).label("fp"),
                func.sum(case((PredictionLog.outcome == PredictionOutcome.FALSE_NEGATIVE, 1), else_=0)).label("fn")
            ).where(
                PredictionLog.outcome != PredictionOutcome.UNKNOWN,
                PredictionLog.outcome_resolved_at >= since_time
            )
            
            result = await db.execute(stmt)
            row = result.fetchone()
            
            tp = row.tp or 0
            fp = row.fp or 0
            fn = row.fn or 0
            
            if tp == 0 and fp == 0 and fn == 0:
                return
                
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Threshold check
            if f1_score < 0.70 and (tp + fp + fn) > 10:  # Need at least 10 flagged transactions to be statistically relevant
                logger.warning("Data drift detected! F1 score (%.3f) fell below threshold (0.70). Initiating automated retraining...", f1_score)
                
                # We offload the blocking model training to a separate thread so we don't block the asyncio event loop
                cfg = TrainConfig(auto_promote=True, description="Auto-retrained due to F1 drift")
                loop = asyncio.get_running_loop()
                # Run the blocking training in a thread pool
                training_result = await loop.run_in_executor(None, train, cfg)
                
                logger.info("Automated retraining completed successfully! New model version %s deployed.", training_result['version'])

    except Exception as e:
        logger.error("Drift detection encountered an error: %s", str(e))

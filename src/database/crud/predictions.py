"""
src/database/crud/predictions.py
----------------------------------
CRUD operations for prediction_log and model_registry tables.

Prediction logs are append-only — there are no update/delete operations
for the core log rows. The only mutable fields are outcome resolution and
human review flags, which are set post-hoc.

Model registry supports full lifecycle: create, promote, archive.
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import and_, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ModelRegistry, ModelStage, PredictionLog, PredictionOutcome


# ===========================================================================
# PREDICTION LOG
# ===========================================================================

async def log_prediction(
    db: AsyncSession,
    *,
    request_id: uuid.UUID,
    transaction_id: uuid.UUID,
    model_name: str,
    model_version: str,
    fraud_score: float,
    risk_label: int,
    decision_threshold: float,
    feature_vector: dict | None = None,
    latency_ms: int = 0,
    batch_id: uuid.UUID | None = None,
) -> PredictionLog:
    """
    Append a single prediction result to the audit log.

    Args:
        request_id: Unique ID per API request (caller-supplied for idempotency).
        transaction_id: The transaction that was scored.
        model_name: Model family name.
        model_version: Exact version tag.
        fraud_score: Raw probability [0.0, 1.0].
        risk_label: Thresholded binary label.
        decision_threshold: Threshold that produced risk_label.
        feature_vector: Dict of feature name → value snapshot.
        latency_ms: Inference latency in milliseconds.
        batch_id: Optional batch grouping UUID.
    """
    log = PredictionLog(
        request_id=request_id,
        batch_id=batch_id,
        transaction_id=transaction_id,
        model_name=model_name,
        model_version=model_version,
        fraud_score=fraud_score,
        risk_label=risk_label,
        decision_threshold=decision_threshold,
        feature_vector=feature_vector,
        latency_ms=latency_ms,
        outcome=PredictionOutcome.UNKNOWN,
        is_reviewed=False,
    )
    db.add(log)
    await db.flush()
    await db.refresh(log)
    return log


async def get_prediction_log_by_id(
    db: AsyncSession,
    log_id: uuid.UUID,
) -> PredictionLog | None:
    """Fetch a single prediction log row by surrogate PK."""
    stmt = select(PredictionLog).where(PredictionLog.log_id == log_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_predictions_for_transaction(
    db: AsyncSession,
    transaction_id: uuid.UUID,
) -> Sequence[PredictionLog]:
    """
    Fetch all prediction log rows for a given transaction.
    A transaction may have been scored by multiple model versions.
    """
    stmt = (
        select(PredictionLog)
        .where(PredictionLog.transaction_id == transaction_id)
        .order_by(PredictionLog.created_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_recent_predictions(
    db: AsyncSession,
    *,
    model_version: str | None = None,
    since: datetime | None = None,
    limit: int = 1000,
) -> Sequence[PredictionLog]:
    """
    Fetch recent prediction logs, optionally filtered by model version and time window.
    Used by the monitoring layer for score distribution analysis.
    """
    conditions = []
    if model_version:
        conditions.append(PredictionLog.model_version == model_version)
    if since:
        conditions.append(PredictionLog.created_at >= since)

    stmt = (
        select(PredictionLog)
        .where(and_(*conditions) if conditions else True)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def resolve_prediction_outcome(
    db: AsyncSession,
    log_id: uuid.UUID,
    outcome: PredictionOutcome,
) -> PredictionLog | None:
    """
    Set the ground-truth resolved outcome once fraud_label is known.
    Called by a background reconciliation job.
    """
    stmt = (
        update(PredictionLog)
        .where(PredictionLog.log_id == log_id)
        .values(
            outcome=outcome,
            outcome_resolved_at=datetime.now(timezone.utc),
        )
        .returning(PredictionLog)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def mark_prediction_reviewed(
    db: AsyncSession,
    log_id: uuid.UUID,
    reviewer_notes: str | None = None,
) -> PredictionLog | None:
    """Mark a prediction as human-reviewed with optional notes."""
    stmt = (
        update(PredictionLog)
        .where(PredictionLog.log_id == log_id)
        .values(is_reviewed=True, reviewer_notes=reviewer_notes)
        .returning(PredictionLog)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


# ===========================================================================
# MODEL REGISTRY
# ===========================================================================

async def register_model(
    db: AsyncSession,
    *,
    model_name: str,
    model_version: str,
    artifact_path: str,
    framework: str,
    framework_version: str,
    python_version: str,
    training_rows: int,
    feature_names: list[str],
    hyperparameters: dict,
    metrics: dict,
    artifact_size_bytes: int | None = None,
    training_duration_s: float | None = None,
    description: str | None = None,
) -> ModelRegistry:
    """
    Register a newly trained model version in CANDIDATE stage.
    Promotion to STAGING or PRODUCTION is a separate explicit action.
    """
    entry = ModelRegistry(
        model_name=model_name,
        model_version=model_version,
        stage=ModelStage.CANDIDATE,
        artifact_path=artifact_path,
        artifact_size_bytes=artifact_size_bytes,
        framework=framework,
        framework_version=framework_version,
        python_version=python_version,
        training_rows=training_rows,
        training_duration_s=training_duration_s,
        feature_names=feature_names,
        hyperparameters=hyperparameters,
        metrics=metrics,
        is_active=False,
        description=description,
    )
    db.add(entry)
    await db.flush()
    await db.refresh(entry)
    return entry


async def get_active_model(
    db: AsyncSession,
    model_name: str,
) -> ModelRegistry | None:
    """
    Fetch the currently active (production) model for a given model family.
    Returns None if no model has been promoted yet.
    """
    stmt = (
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == model_name,
                ModelRegistry.is_active.is_(True),
            )
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_model_by_version(
    db: AsyncSession,
    model_name: str,
    model_version: str,
) -> ModelRegistry | None:
    """Fetch a specific model version entry."""
    stmt = select(ModelRegistry).where(
        and_(
            ModelRegistry.model_name == model_name,
            ModelRegistry.model_version == model_version,
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_model_versions(
    db: AsyncSession,
    model_name: str,
    *,
    stage: ModelStage | None = None,
    limit: int = 50,
) -> Sequence[ModelRegistry]:
    """
    List all versions for a model family, newest first.
    Optionally filter by lifecycle stage.
    """
    conditions = [ModelRegistry.model_name == model_name]
    if stage:
        conditions.append(ModelRegistry.stage == stage)

    stmt = (
        select(ModelRegistry)
        .where(and_(*conditions))
        .order_by(desc(ModelRegistry.created_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def promote_model(
    db: AsyncSession,
    model_name: str,
    model_version: str,
    new_stage: ModelStage,
    promoted_by: str = "system",
) -> ModelRegistry | None:
    """
    Promote a model version to a new stage.

    If promoting to PRODUCTION:
    - Demotes the currently active model to ARCHIVED.
    - Sets is_active = True on the new version.

    Returns the promoted registry entry, or None if version not found.
    """
    # Find the target model
    target = await get_model_by_version(db, model_name, model_version)
    if not target:
        return None

    if new_stage == ModelStage.PRODUCTION:
        # Demote existing active model to ARCHIVED
        await db.execute(
            update(ModelRegistry)
            .where(
                and_(
                    ModelRegistry.model_name == model_name,
                    ModelRegistry.is_active.is_(True),
                )
            )
            .values(stage=ModelStage.ARCHIVED, is_active=False)
        )
        # Promote target
        await db.execute(
            update(ModelRegistry)
            .where(ModelRegistry.registry_id == target.registry_id)
            .values(
                stage=new_stage,
                is_active=True,
                promoted_at=datetime.now(timezone.utc),
                promoted_by=promoted_by,
            )
        )
    else:
        await db.execute(
            update(ModelRegistry)
            .where(ModelRegistry.registry_id == target.registry_id)
            .values(
                stage=new_stage,
                promoted_at=datetime.now(timezone.utc),
                promoted_by=promoted_by,
            )
        )

    await db.flush()
    await db.refresh(target)
    return target

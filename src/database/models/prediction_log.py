"""
src/database/models/prediction_log.py
---------------------------------------
Audit log for every inference call made by the risk scoring API.

Purpose:
- Complete traceability: which model version scored which transaction.
- Source of truth for model performance monitoring over time.
- Ground-truth join: link prediction_score → fraud_label from fact_transaction.
- Immutable after insert — predictions are never edited, only appended.

Design notes:
- One row per individual scoring request (not per batch).
- Batch requests explode into N rows with the same batch_id.
- model_version is denormalised here (not FK) to retain history even if
  the registry row is deleted.
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base, TimestampMixin


class PredictionOutcome(str, PyEnum):
    """
    Resolved outcome once ground truth becomes available.
    Set by a background reconciliation job.
    """
    TRUE_POSITIVE = "true_positive"    # Predicted fraud, was fraud
    TRUE_NEGATIVE = "true_negative"    # Predicted legit, was legit
    FALSE_POSITIVE = "false_positive"  # Predicted fraud, was legit
    FALSE_NEGATIVE = "false_negative"  # Predicted legit, was fraud
    UNKNOWN = "unknown"                # Ground truth not yet available


class PredictionLog(Base, TimestampMixin):
    """
    Log Table: Model Predictions

    Columns
    -------
    log_id              — Surrogate PK (UUID v4)
    request_id          — Idempotency key passed by API caller
    batch_id            — Groups batch prediction rows together (nullable for single)
    transaction_id      — FK → fact_transaction.transaction_id
    model_name          — Name of the model family (e.g. "fraud_xgboost")
    model_version       — Exact version tag (e.g. "v3_20260301")
    fraud_score         — Raw probability output [0.0 – 1.0]
    risk_label          — Thresholded binary decision (0=legit, 1=fraud)
    decision_threshold  — Threshold used to convert score → label
    feature_vector      — JSONB snapshot of features used for this prediction
    latency_ms          — End-to-end inference latency in milliseconds
    outcome             — Resolved outcome once ground truth is known
    outcome_resolved_at — Timestamp when outcome was set
    is_reviewed         — True if a human analyst has reviewed this prediction
    reviewer_notes      — Free-text notes from reviewer
    """

    __tablename__ = "prediction_log"
    __table_args__ = (
        # Primary queries: all predictions for a given transaction
        Index("ix_pred_log_transaction_id", "transaction_id"),
        # Time-range queries for monitoring windows
        Index("ix_pred_log_created_at", "created_at"),
        # Per-model-version queries for performance reports
        Index("ix_pred_log_model_version_time", "model_version", "created_at"),
        # Outcome resolution queries
        Index("ix_pred_log_outcome", "outcome"),
        # Idempotency check
        Index("ix_pred_log_request_id", "request_id", unique=True),
        # Batch grouping
        Index("ix_pred_log_batch_id", "batch_id"),
        # Check: score must be a valid probability
        CheckConstraint(
            "fraud_score >= 0.0 AND fraud_score <= 1.0",
            name="ck_pred_log_score_range",
        ),
        CheckConstraint(
            "risk_label IN (0, 1)",
            name="ck_pred_log_risk_label_binary",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="ck_pred_log_latency_positive",
        ),
        {"comment": "Immutable audit log of all model predictions"},
    )

    # --- Primary Key ---
    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Surrogate primary key (UUID v4)",
    )

    # --- Request Identity ---
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        default=uuid.uuid4,
        comment="Per-request idempotency key from API caller",
    )
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Groups all rows from a single batch predict call",
    )

    # --- Transaction Reference ---
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fact_transaction.transaction_id", ondelete="RESTRICT"),
        nullable=False,
        comment="FK → fact_transaction",
    )

    # --- Model Identity (denormalised for history preservation) ---
    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Model family name (e.g. fraud_xgboost)",
    )
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Exact version tag (e.g. v3_20260301_143022)",
    )

    # --- Prediction Output ---
    fraud_score: Mapped[float] = mapped_column(
        Numeric(6, 5),
        nullable=False,
        comment="Fraud probability [0.00000 – 1.00000]",
    )
    risk_label: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Thresholded binary label: 0=legit, 1=fraud",
    )
    decision_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
        comment="Decision threshold applied to convert score to label",
    )

    # --- Feature Snapshot (for debugging & drift analysis) ---
    feature_vector: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="JSONB snapshot of input feature values at prediction time",
    )

    # --- Performance ---
    latency_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="End-to-end inference latency in milliseconds",
    )

    # --- Ground Truth Resolution ---
    outcome: Mapped[PredictionOutcome] = mapped_column(
        Enum(PredictionOutcome, name="prediction_outcome_enum", create_type=True),
        nullable=False,
        default=PredictionOutcome.UNKNOWN,
        comment="Resolved outcome once ground truth fraud_label is known",
    )
    outcome_resolved_at: Mapped[uuid.UUID | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when ground truth was reconciled",
    )

    # --- Human Review ---
    is_reviewed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if an analyst has reviewed this prediction",
    )
    reviewer_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Free-text analyst notes post-review",
    )

    # --- Relationships ---
    transaction: Mapped["FactTransaction"] = relationship(  # noqa: F821
        "FactTransaction",
        back_populates="prediction_logs",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<PredictionLog(log_id={self.log_id!s}, "
            f"transaction_id={self.transaction_id!s}, "
            f"fraud_score={self.fraud_score:.4f}, "
            f"model_version={self.model_version!r})>"
        )

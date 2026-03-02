"""
src/database/models/model_registry.py
---------------------------------------
Stores metadata for every trained model version.

Purpose:
- Version-controlled model inventory — every training run creates a row.
- Promotion workflow: only one model can be "active" at a time per name.
- Stores evaluation metrics as JSONB so new metrics can be added without
  schema migrations.
- Links to the serialised artifact on disk (or object storage in production).
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, TimestampMixin


class ModelStage(str, PyEnum):
    """
    Lifecycle stage for a model version.
    Mirrors MLflow stage conventions for future compatibility.
    """
    CANDIDATE = "candidate"    # Freshly trained, under evaluation
    STAGING = "staging"        # Passed offline eval, awaiting A/B or shadow test
    PRODUCTION = "production"  # Live — serving real predictions
    ARCHIVED = "archived"      # Superseded by a newer version
    FAILED = "failed"          # Training or evaluation failed


class ModelRegistry(Base, TimestampMixin):
    """
    Model Registry Table

    Columns
    -------
    registry_id         — Surrogate PK (UUID v4)
    model_name          — Model family name (e.g. "fraud_xgboost")
    model_version       — Semantic version tag (e.g. "v3_20260301_143022")
    stage               — Lifecycle stage
    artifact_path       — Absolute path to .pkl / .joblib file on shared volume
    artifact_size_bytes — File size for storage monitoring
    framework           — ML framework name (e.g. "xgboost", "sklearn")
    framework_version   — pinned version (e.g. "2.0.3")
    python_version      — Python version used at training time
    training_rows       — Number of training samples
    training_duration_s — Wall-clock training time in seconds
    feature_names       — JSONB list of feature names in order used for training
    hyperparameters     — JSONB of hyperparameter values
    metrics             — JSONB of evaluation metrics (AUC, F1, etc.)
    is_active           — True for the single currently-serving version per name
    promoted_at         — Timestamp of last stage promotion
    promoted_by         — Identifier of the user/job that promoted
    description         — Free-text description / release notes
    """

    __tablename__ = "model_registry"
    __table_args__ = (
        # Only one active model per model_name at a time
        UniqueConstraint(
            "model_name", "is_active",
            name="uq_model_registry_one_active",
        ),
        # Unique version within a model family
        UniqueConstraint(
            "model_name", "model_version",
            name="uq_model_registry_version",
        ),
        Index("ix_model_registry_name_stage", "model_name", "stage"),
        Index("ix_model_registry_active", "model_name", "is_active"),
        CheckConstraint(
            "training_rows > 0",
            name="ck_model_registry_training_rows_positive",
        ),
        {"comment": "Versioned model registry with promotion workflow"},
    )

    # --- Primary Key ---
    registry_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Surrogate PK",
    )

    # --- Identity ---
    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Model family name",
    )
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Unique version tag within this model family",
    )
    stage: Mapped[ModelStage] = mapped_column(
        Enum(ModelStage, name="model_stage_enum", create_type=True),
        nullable=False,
        default=ModelStage.CANDIDATE,
        comment="Current lifecycle stage",
    )

    # --- Artifact Location ---
    artifact_path: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Absolute path to serialised model file",
    )
    artifact_size_bytes: Mapped[int | None] = mapped_column(
        Numeric(18, 0),
        nullable=True,
        comment="File size in bytes",
    )

    # --- Environment Metadata ---
    framework: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="ML framework (e.g. xgboost, sklearn)",
    )
    framework_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Framework pinned version string",
    )
    python_version: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="Python version at training time",
    )

    # --- Training Metadata ---
    training_rows: Mapped[int] = mapped_column(
        Numeric(18, 0),
        nullable=False,
        comment="Number of rows in training dataset",
    )
    training_duration_s: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Wall-clock training time in seconds",
    )

    # --- Feature / Hyperparameter Snapshots ---
    feature_names: Mapped[list | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Ordered list of feature names used during training",
    )
    hyperparameters: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="JSONB hyperparameter dict",
    )

    # --- Evaluation Metrics ---
    metrics: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="JSONB evaluation metrics: {auc_roc, auc_pr, f1, precision, recall, ...}",
    )

    # --- Promotion State ---
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True = currently serving production traffic",
    )
    promoted_at: Mapped[uuid.UUID | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last promotion timestamp",
    )
    promoted_by: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="User or service that triggered promotion",
    )

    # --- Notes ---
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Release notes / model description",
    )

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry(model_name={self.model_name!r}, "
            f"version={self.model_version!r}, stage={self.stage}, "
            f"is_active={self.is_active})>"
        )

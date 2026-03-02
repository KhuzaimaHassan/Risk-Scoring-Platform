"""
src/database/models/__init__.py
---------------------------------
Centralised model registry for SQLAlchemy.

Importing this module guarantees that all ORM models are registered
with the shared Base metadata — required for:
  1. Alembic autogenerate to detect all tables.
  2. Base.metadata.create_all() in test fixtures.
  3. Relationship resolution across module boundaries.

Always import from here, never from individual model files directly,
to avoid circular import issues.
"""

from src.database.base import Base, TimestampMixin
from src.database.models.dim_merchant import DimMerchant, MerchantCategory, MerchantRiskLevel
from src.database.models.dim_user import DimUser, RiskTier
from src.database.models.fact_transaction import (
    FactTransaction,
    PaymentMethod,
    TransactionChannel,
    TransactionStatus,
)
from src.database.models.model_registry import ModelRegistry, ModelStage
from src.database.models.prediction_log import PredictionLog, PredictionOutcome

__all__ = [
    # Base
    "Base",
    "TimestampMixin",
    # Dimensions
    "DimUser",
    "RiskTier",
    "DimMerchant",
    "MerchantCategory",
    "MerchantRiskLevel",
    # Fact
    "FactTransaction",
    "TransactionStatus",
    "TransactionChannel",
    "PaymentMethod",
    # Prediction Audit
    "PredictionLog",
    "PredictionOutcome",
    # Model Registry
    "ModelRegistry",
    "ModelStage",
]

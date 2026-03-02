"""
src/api/schemas.py
-------------------
Pydantic request and response schemas for the inference API.

All schemas are versioned via the API prefix (e.g. /api/v1/predict).
Input schemas use strict validation; output schemas expose only safe fields.

Design principles:
    - Fail fast: invalid payloads are rejected at the schema layer, before
      any DB or model code executes.
    - Explicit over implicit: every field has a type, description, and
      example value.
    - No ORM objects in responses — all output is serialised from pure
      Pydantic models to keep the API layer decoupled from SQLAlchemy.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enum mirrors (avoid importing SQLAlchemy enums from ORM into the API layer)
# ---------------------------------------------------------------------------

class ChannelEnum(str, Enum):
    WEB    = "web"
    MOBILE = "mobile"
    POS    = "pos"
    ATM    = "atm"
    API    = "api"


class PaymentMethodEnum(str, Enum):
    CREDIT_CARD   = "credit_card"
    DEBIT_CARD    = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO        = "crypto"
    WALLET        = "wallet"
    BNPL          = "bnpl"


# ---------------------------------------------------------------------------
# Prediction Request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Payload for POST /predict.

    The transaction MUST already exist in fact_transaction (inserted by the
    upstream payment service before calling the risk API). The scoring engine
    fetches all necessary context (user history, merchant data) from the
    database using the provided transaction_id.

    Alternatively, when `include_features` is True, the response will also
    include the raw feature vector — useful for debugging and shadow-mode
    monitoring.
    """

    transaction_id: UUID = Field(
        ...,
        description="UUID of the transaction in fact_transaction to score.",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    include_features: bool = Field(
        default=False,
        description="If True, include the feature vector snapshot in the response.",
    )
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "transaction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "include_features": False,
        }
    })


class PredictBatchRequest(BaseModel):
    """
    Payload for POST /predict/batch — scores multiple transactions in one call.
    Maximum 200 transactions per batch to prevent timeout.
    """
    transaction_ids: list[UUID] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of transaction UUIDs to score.",
    )
    include_features: bool = Field(default=False)

    @field_validator("transaction_ids")
    @classmethod
    def deduplicate(cls, v: list[UUID]) -> list[UUID]:
        """Silently deduplicate IDs to avoid double-billing logic."""
        seen: set[UUID] = set()
        return [x for x in v if not (x in seen or seen.add(x))]  # type: ignore


# ---------------------------------------------------------------------------
# Prediction Response
# ---------------------------------------------------------------------------

class ConfusionContext(BaseModel):
    """Human-readable explanation attached to each prediction."""
    risk_band: str = Field(..., description="low | medium | high | critical")
    recommended_action: str = Field(..., description="allow | review | block")
    confidence: str = Field(..., description="low | moderate | high")


class PredictResponse(BaseModel):
    """
    Response for POST /predict.

    All probability and threshold values are rounded to 6 decimal places
    for deterministic serialisation.
    """

    transaction_id: UUID
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw fraud probability from the model [0.0, 1.0].",
    )
    is_fraud: bool = Field(
        ...,
        description="Binary fraud classification (threshold applied).",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Fraud probability scaled to [0, 100] for readability.",
    )
    decision_threshold: float = Field(
        ...,
        description="The threshold used to derive is_fraud from fraud_probability.",
    )
    model_version: str = Field(..., description="Version tag of the model that scored this.")
    risk_context: ConfusionContext
    log_id: UUID = Field(..., description="UUID of the prediction_log row for this event.")
    latency_ms: int = Field(..., description="End-to-end inference latency in milliseconds.")
    scored_at: datetime

    # Optional — only present when request.include_features=True
    feature_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Feature vector used for this prediction (debug mode only).",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "transaction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "fraud_probability": 0.83,
            "is_fraud": True,
            "risk_score": 83.0,
            "decision_threshold": 0.50,
            "model_version": "v20260301_143022",
            "risk_context": {
                "risk_band": "critical",
                "recommended_action": "block",
                "confidence": "high",
            },
            "log_id": "9b2e5a3f-1234-5678-abcd-ef0123456789",
            "latency_ms": 47,
            "scored_at": "2026-03-01T14:30:22Z",
            "feature_snapshot": None,
        }
    })


class PredictBatchResponse(BaseModel):
    """Response for POST /predict/batch."""
    results: list[PredictResponse]
    total_requested: int
    total_scored: int
    total_fraud: int
    batch_latency_ms: int


# ---------------------------------------------------------------------------
# Health & Model Info
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    model_version: str | None
    db_connected: bool
    uptime_seconds: float
    timestamp: datetime


class ModelMetrics(BaseModel):
    """Metrics sub-object for ModelInfoResponse."""
    roc_auc: float | None = None
    pr_auc: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    threshold_used: float | None = None
    training_rows: int | None = None
    test_samples: int | None = None


class ModelInfoResponse(BaseModel):
    """Response for GET /model-info."""
    model_name: str
    model_version: str
    stage: str
    trained_at: str | None
    n_features: int
    feature_columns: list[str]
    hyperparameters: dict[str, Any]
    metrics: ModelMetrics
    artifact_path: str


class ModelListItem(BaseModel):
    """Single item in GET /models list."""
    version: str
    stage: str
    trained_at: str | None
    roc_auc: float | None
    f1: float | None
    is_production: bool


class ModelListResponse(BaseModel):
    """Response for GET /models."""
    current_production: str | None
    total_versions: int
    versions: list[ModelListItem]


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    """Standard error response body."""
    error: str
    detail: str
    request_id: str | None = None


class ValidationErrorResponse(BaseModel):
    """422 Unprocessable Entity body."""
    error: str = "validation_error"
    detail: list[dict[str, Any]]

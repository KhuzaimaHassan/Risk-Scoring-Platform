"""
src/database/models/fact_transaction.py
-----------------------------------------
Central fact table for the star schema — the primary source of truth
for all transactional events in the risk platform.

Design notes:
- One row per payment event (immutable after insert).
- Partitioned by txn_timestamp in production (range partitioning by month).
- FKs to dim_user and dim_merchant for dimensional enrichment.
- fraud_label is set post-review (nullable until confirmed by ops team).
- All monetary values stored as Numeric(18,4) to avoid floating-point drift.
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
)
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base, TimestampMixin


class TransactionStatus(str, PyEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    DECLINED = "declined"
    REVERSED = "reversed"
    FLAGGED = "flagged"


class TransactionChannel(str, PyEnum):
    WEB = "web"
    MOBILE = "mobile"
    POS = "pos"            # Point-of-sale terminal
    ATM = "atm"
    API = "api"            # Programmatic / B2B


class PaymentMethod(str, PyEnum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"
    WALLET = "wallet"
    BUY_NOW_PAY_LATER = "bnpl"


class FactTransaction(Base, TimestampMixin):
    """
    Fact Table: Transactions

    Immutability contract: rows are NEVER updated after status reaches
    COMPLETED or DECLINED. Corrections are recorded as new reversal rows.

    Columns
    -------
    transaction_id      — Surrogate PK (UUID v4)
    external_txn_id     — Idempotency key from upstream payment service
    user_id             — FK → dim_user.user_id
    merchant_id         — FK → dim_merchant.merchant_id
    txn_timestamp       — Exact event timestamp (UTC, with timezone)
    amount              — Transaction amount in original currency
    currency            — ISO 4217 (e.g. "USD", "EUR")
    amount_usd          — Normalised USD equivalent at time of transaction
    status              — Current processing status
    channel             — Originating channel
    payment_method      — Payment instrument type
    ip_address          — Hashed client IP (PII compliance)
    device_fingerprint  — Hashed device ID
    is_international    — True if user country ≠ merchant country
    fraud_label         — Ground truth: 1=fraud, 0=legit, null=unknown
    labelled_at         — Timestamp when fraud_label was set
    review_notes        — Free-text from ops / analyst review
    """

    __tablename__ = "fact_transaction"
    __table_args__ = (
        # Composite index for fast user-based feature window queries
        Index("ix_fact_txn_user_time", "user_id", "txn_timestamp"),
        # Composite index for merchant-based feature window queries
        Index("ix_fact_txn_merchant_time", "merchant_id", "txn_timestamp"),
        # Index for fraud label queries (training dataset extraction)
        Index("ix_fact_txn_fraud_label", "fraud_label", "txn_timestamp"),
        # Index for status filtering
        Index("ix_fact_txn_status", "status"),
        # Index for idempotency checks at ingest
        Index("ix_fact_txn_external_id", "external_txn_id", unique=True),
        # Check: amount must be positive
        CheckConstraint("amount > 0", name="ck_fact_txn_amount_positive"),
        CheckConstraint("amount_usd >= 0", name="ck_fact_txn_amount_usd_non_negative"),
        CheckConstraint(
            "fraud_label IS NULL OR fraud_label IN (0, 1)",
            name="ck_fact_txn_fraud_label_binary",
        ),
        {"comment": "Core transaction fact table — star schema"},
    )

    # --- Primary Key ---
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Surrogate primary key (UUID v4)",
    )

    # --- Idempotency ---
    external_txn_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Upstream idempotency key / payment reference",
    )

    # --- Dimension Foreign Keys ---
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dim_user.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="FK → dim_user",
    )
    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dim_merchant.merchant_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="FK → dim_merchant",
    )

    # --- Timing ---
    txn_timestamp: Mapped[uuid.UUID] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Exact transaction event timestamp (UTC)",
    )

    # --- Value ---
    amount: Mapped[float] = mapped_column(
        Numeric(18, 4),
        nullable=False,
        comment="Original transaction amount in source currency",
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="USD",
        comment="ISO 4217 currency code",
    )
    amount_usd: Mapped[float] = mapped_column(
        Numeric(18, 4),
        nullable=False,
        comment="FX-normalised amount in USD at transaction time",
    )

    # --- Classification ---
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus, name="txn_status_enum", create_type=True),
        nullable=False,
        default=TransactionStatus.PENDING,
        comment="Current processing status",
    )
    channel: Mapped[TransactionChannel] = mapped_column(
        Enum(TransactionChannel, name="txn_channel_enum", create_type=True),
        nullable=False,
        comment="Originating transaction channel",
    )
    payment_method: Mapped[PaymentMethod] = mapped_column(
        Enum(PaymentMethod, name="payment_method_enum", create_type=True),
        nullable=False,
        comment="Payment instrument type",
    )

    # --- Device & Network (PII-safe hashed) ---
    ip_address_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="SHA-256 of client IP address",
    )
    device_fingerprint_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Hashed device fingerprint",
    )

    # --- Geo Risk Signal ---
    is_international: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if user country ≠ merchant country",
    )

    # --- Ground Truth Label ---
    fraud_label: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Confirmed fraud label: 1=fraud, 0=legit, NULL=unlabelled",
    )
    labelled_at: Mapped[uuid.UUID | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when fraud_label was assigned",
    )
    review_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Analyst review notes",
    )

    # --- Relationships ---
    user: Mapped["DimUser"] = relationship(  # noqa: F821
        "DimUser",
        back_populates="transactions",
        lazy="noload",
    )
    merchant: Mapped["DimMerchant"] = relationship(  # noqa: F821
        "DimMerchant",
        back_populates="transactions",
        lazy="noload",
    )
    prediction_logs: Mapped[list["PredictionLog"]] = relationship(  # noqa: F821
        "PredictionLog",
        back_populates="transaction",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<FactTransaction(transaction_id={self.transaction_id!s}, "
            f"amount_usd={self.amount_usd}, status={self.status}, "
            f"fraud_label={self.fraud_label})>"
        )

"""
src/database/models/dim_user.py
--------------------------------
Dimension table for customers / users who initiate transactions.

Star-schema role: slowly-changing dimension (Type 1 — overwrite in place).
This table holds the current snapshot of user attributes that enrich
fact_transactions for analytics and feature engineering.
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import BigInteger, Boolean, Enum, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base, TimestampMixin


class RiskTier(str, PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class DimUser(Base, TimestampMixin):
    """
    Dimension: Users / Customers

    Columns
    -------
    user_id         — Surrogate PK (UUID), used in fact table FK
    external_id     — Business key from upstream system (e.g. CRM ID)
    full_name       — Display name (masked in non-prod environments)
    email           — Hashed or tokenised for PII compliance
    country_code    — ISO 3166-1 alpha-2 (e.g. "US", "GB")
    account_age_days— Derived at ingestion time; int for fast filtering
    risk_tier       — Current risk classification from rule engine
    is_active       — Soft-delete flag; inactive users cannot transact
    kyc_verified    — Know-Your-Customer flag; drives rule eligibility
    credit_score    — Optional bureau score [300-850]; null if unavailable
    lifetime_txn_count — Denormalised counter; updated by trigger/ETL
    lifetime_txn_volume — Denormalised total spend; decimal(18,4)
    """

    __tablename__ = "dim_user"
    __table_args__ = (
        Index("ix_dim_user_external_id", "external_id", unique=True),
        Index("ix_dim_user_country_risk", "country_code", "risk_tier"),
        Index("ix_dim_user_kyc_active", "kyc_verified", "is_active"),
        {"comment": "Customer dimension — star schema"},
    )

    # --- Primary Key ---
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Surrogate primary key (UUID v4)",
    )

    # --- Business / Natural Key ---
    external_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Upstream business key from source system",
    )

    # --- Identity ---
    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="User display name — mask in non-prod",
    )
    email_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="SHA-256 hash of normalised email for PII safety",
    )
    country_code: Mapped[str] = mapped_column(
        String(2),
        nullable=False,
        comment="ISO 3166-1 alpha-2 country code",
    )

    # --- Account Attributes ---
    account_age_days: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="Age of account in days at last update",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-delete flag; False = deactivated user",
    )
    kyc_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether full KYC has been completed",
    )

    # --- Risk Profile ---
    risk_tier: Mapped[RiskTier] = mapped_column(
        Enum(RiskTier, name="risk_tier_enum", create_type=True),
        nullable=False,
        default=RiskTier.MEDIUM,
        comment="Current assigned risk tier",
    )
    credit_score: Mapped[float | None] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="Bureau credit score [300.00 – 850.00]; nullable",
    )

    # --- Denormalised Aggregates (updated by ETL/trigger) ---
    lifetime_txn_count: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="Total historical transaction count",
    )
    lifetime_txn_volume: Mapped[float] = mapped_column(
        Numeric(18, 4),
        nullable=False,
        default=0,
        comment="Total historical transaction volume in USD",
    )

    # --- Notes ---
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Free-form internal notes / flags",
    )

    # --- Relationships ---
    transactions: Mapped[list["FactTransaction"]] = relationship(  # noqa: F821
        "FactTransaction",
        back_populates="user",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return f"<DimUser(user_id={self.user_id!s}, external_id={self.external_id!r}, risk_tier={self.risk_tier})>"

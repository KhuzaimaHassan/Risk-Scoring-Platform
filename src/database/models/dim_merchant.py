"""
src/database/models/dim_merchant.py
-------------------------------------
Dimension table for merchants who receive transactions.

Star-schema role: slowly-changing dimension (Type 1).
Merchant risk attributes feed directly into feature engineering
(e.g. historical fraud rate per merchant category).
"""

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Enum, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base, TimestampMixin


class MerchantCategory(str, PyEnum):
    """
    Simplified MCC (Merchant Category Code) groupings.
    Extend with real MCCs as needed.
    """
    RETAIL = "retail"
    FOOD_BEVERAGE = "food_beverage"
    TRAVEL = "travel"
    ELECTRONICS = "electronics"
    GAMBLING = "gambling"
    CRYPTO = "crypto"
    HEALTHCARE = "healthcare"
    UTILITIES = "utilities"
    OTHER = "other"


class MerchantRiskLevel(str, PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DimMerchant(Base, TimestampMixin):
    """
    Dimension: Merchants

    Columns
    -------
    merchant_id         — Surrogate PK (UUID)
    external_id         — Business key from payment processor / MCC system
    merchant_name       — Display name
    category            — Simplified MCC grouping
    country_code        — ISO 3166-1 alpha-2 merchant registration country
    city                — Merchant city (for geo-risk features)
    is_online_only      — True for e-commerce / CNP merchants (higher fraud risk)
    is_active           — Soft-delete flag
    risk_level          — Internal risk level classification
    historical_fraud_rate — Rolling 90-day fraud rate [0.0 – 1.0]; updated by ETL
    avg_transaction_amount — Rolling 30-day average; used as baseline for z-score
    total_txn_count     — Denormalised lifetime count
    is_high_risk_category — Precomputed flag (gambling/crypto = True)
    """

    __tablename__ = "dim_merchant"
    __table_args__ = (
        Index("ix_dim_merchant_external_id", "external_id", unique=True),
        Index("ix_dim_merchant_category_risk", "category", "risk_level"),
        Index("ix_dim_merchant_country", "country_code"),
        Index("ix_dim_merchant_fraud_rate", "historical_fraud_rate"),
        {"comment": "Merchant dimension — star schema"},
    )

    # --- Primary Key ---
    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Surrogate primary key (UUID v4)",
    )

    # --- Business Key ---
    external_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Merchant ID from payment processor",
    )

    # --- Identity ---
    merchant_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Merchant display name",
    )
    category: Mapped[MerchantCategory] = mapped_column(
        Enum(MerchantCategory, name="merchant_category_enum", create_type=True),
        nullable=False,
        default=MerchantCategory.OTHER,
        comment="Simplified MCC category grouping",
    )
    country_code: Mapped[str] = mapped_column(
        String(2),
        nullable=False,
        comment="ISO 3166-1 alpha-2 country of registration",
    )
    city: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Merchant city for geo-risk features",
    )

    # --- Channel ---
    is_online_only: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True = card-not-present / e-commerce merchant",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-delete flag",
    )

    # --- Risk Profile ---
    risk_level: Mapped[MerchantRiskLevel] = mapped_column(
        Enum(MerchantRiskLevel, name="merchant_risk_level_enum", create_type=True),
        nullable=False,
        default=MerchantRiskLevel.MEDIUM,
        comment="Internal risk classification",
    )
    is_high_risk_category: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Precomputed flag: True for gambling/crypto categories",
    )

    # --- Denormalised Statistics (updated by ETL) ---
    historical_fraud_rate: Mapped[float] = mapped_column(
        Numeric(6, 5),
        nullable=False,
        default=0.0,
        comment="Rolling 90-day fraud rate [0.00000 – 1.00000]",
    )
    avg_transaction_amount: Mapped[float] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        default=0.0,
        comment="Rolling 30-day average transaction amount (USD)",
    )
    total_txn_count: Mapped[int] = mapped_column(
        Numeric(18, 0),
        nullable=False,
        default=0,
        comment="Lifetime transaction count",
    )

    # --- Notes ---
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Free-form risk notes",
    )

    # --- Relationships ---
    transactions: Mapped[list["FactTransaction"]] = relationship(  # noqa: F821
        "FactTransaction",
        back_populates="merchant",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return (
            f"<DimMerchant(merchant_id={self.merchant_id!s}, "
            f"name={self.merchant_name!r}, category={self.category})>"
        )

"""
src/database/crud/transactions.py
-----------------------------------
CRUD operations for the fact_transaction table.

Design principles:
- All functions accept an AsyncSession as first arg (dependency injected).
- No business logic — pure data access.
- Returns ORM objects; serialisation happens at the API schema layer.
- Uses select() with explicit column loading to avoid N+1 problems.
- All writes are committed by the caller (session lifespan managed externally).
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import FactTransaction, TransactionStatus


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------

async def create_transaction(
    db: AsyncSession,
    *,
    external_txn_id: str,
    user_id: uuid.UUID,
    merchant_id: uuid.UUID,
    txn_timestamp: datetime,
    amount: float,
    currency: str,
    amount_usd: float,
    status: TransactionStatus,
    channel: str,
    payment_method: str,
    is_international: bool = False,
    ip_address_hash: str | None = None,
    device_fingerprint_hash: str | None = None,
) -> FactTransaction:
    """
    Insert a new transaction row.

    Returns the persisted ORM object (with server-assigned created_at).
    The session is NOT committed here — the caller controls the transaction.
    """
    transaction = FactTransaction(
        external_txn_id=external_txn_id,
        user_id=user_id,
        merchant_id=merchant_id,
        txn_timestamp=txn_timestamp,
        amount=amount,
        currency=currency,
        amount_usd=amount_usd,
        status=status,
        channel=channel,
        payment_method=payment_method,
        is_international=is_international,
        ip_address_hash=ip_address_hash,
        device_fingerprint_hash=device_fingerprint_hash,
    )
    db.add(transaction)
    await db.flush()  # Assign DB-generated defaults without committing
    await db.refresh(transaction)
    return transaction


# ---------------------------------------------------------------------------
# READ — single row
# ---------------------------------------------------------------------------

async def get_transaction_by_id(
    db: AsyncSession,
    transaction_id: uuid.UUID,
) -> FactTransaction | None:
    """Fetch a single transaction by surrogate PK. Returns None if not found."""
    stmt = select(FactTransaction).where(
        FactTransaction.transaction_id == transaction_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_transaction_by_external_id(
    db: AsyncSession,
    external_txn_id: str,
) -> FactTransaction | None:
    """Fetch by upstream idempotency key. Used for deduplication at ingest."""
    stmt = select(FactTransaction).where(
        FactTransaction.external_txn_id == external_txn_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# READ — collections
# ---------------------------------------------------------------------------

async def get_transactions_by_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    since: datetime | None = None,
    limit: int = 500,
) -> Sequence[FactTransaction]:
    """
    Fetch transactions for a user, ordered newest-first.
    `since` filters to a rolling time window for feature engineering.
    """
    conditions = [FactTransaction.user_id == user_id]
    if since:
        conditions.append(FactTransaction.txn_timestamp >= since)

    stmt = (
        select(FactTransaction)
        .where(and_(*conditions))
        .order_by(FactTransaction.txn_timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_transactions_by_merchant(
    db: AsyncSession,
    merchant_id: uuid.UUID,
    *,
    since: datetime | None = None,
    limit: int = 500,
) -> Sequence[FactTransaction]:
    """
    Fetch transactions for a merchant within an optional time window.
    Used for merchant-level feature aggregations.
    """
    conditions = [FactTransaction.merchant_id == merchant_id]
    if since:
        conditions.append(FactTransaction.txn_timestamp >= since)

    stmt = (
        select(FactTransaction)
        .where(and_(*conditions))
        .order_by(FactTransaction.txn_timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_unlabelled_transactions(
    db: AsyncSession,
    *,
    limit: int = 1000,
) -> Sequence[FactTransaction]:
    """
    Fetch completed transactions that still have fraud_label = NULL.
    Used by the ground-truth reconciliation job.
    """
    stmt = (
        select(FactTransaction)
        .where(
            and_(
                FactTransaction.fraud_label.is_(None),
                FactTransaction.status == TransactionStatus.COMPLETED,
            )
        )
        .order_by(FactTransaction.txn_timestamp.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_labelled_transactions_for_training(
    db: AsyncSession,
    *,
    since: datetime | None = None,
    limit: int = 200_000,
) -> Sequence[FactTransaction]:
    """
    Fetch labelled transactions for model training dataset extraction.
    Optionally scoped to a time window for recency-weighted retraining.
    """
    conditions = [FactTransaction.fraud_label.is_not(None)]
    if since:
        conditions.append(FactTransaction.txn_timestamp >= since)

    stmt = (
        select(FactTransaction)
        .where(and_(*conditions))
        .order_by(FactTransaction.txn_timestamp.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------

async def update_transaction_status(
    db: AsyncSession,
    transaction_id: uuid.UUID,
    new_status: TransactionStatus,
) -> FactTransaction | None:
    """
    Update status field of an existing transaction.
    Returns the updated object, or None if not found.
    """
    stmt = (
        update(FactTransaction)
        .where(FactTransaction.transaction_id == transaction_id)
        .values(status=new_status)
        .returning(FactTransaction)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def set_fraud_label(
    db: AsyncSession,
    transaction_id: uuid.UUID,
    fraud_label: int,
    review_notes: str | None = None,
) -> FactTransaction | None:
    """
    Set the ground-truth fraud label on a transaction (0=legit, 1=fraud).
    Also records the labelling timestamp.
    Only call this once per transaction — fraud labels are immutable after set.
    """
    stmt = (
        update(FactTransaction)
        .where(FactTransaction.transaction_id == transaction_id)
        .values(
            fraud_label=fraud_label,
            labelled_at=datetime.now(timezone.utc),
            review_notes=review_notes,
        )
        .returning(FactTransaction)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# AGGREGATE — used for monitoring and stats endpoints
# ---------------------------------------------------------------------------

async def count_transactions_in_window(
    db: AsyncSession,
    *,
    since: datetime,
    user_id: uuid.UUID | None = None,
    merchant_id: uuid.UUID | None = None,
) -> int:
    """
    Count transactions within a time window, optionally filtered by user/merchant.
    Used for velocity features and rate-limit checks.
    """
    conditions = [FactTransaction.txn_timestamp >= since]
    if user_id:
        conditions.append(FactTransaction.user_id == user_id)
    if merchant_id:
        conditions.append(FactTransaction.merchant_id == merchant_id)

    stmt = select(func.count()).select_from(FactTransaction).where(and_(*conditions))
    result = await db.execute(stmt)
    return result.scalar_one()


async def get_fraud_rate_summary(
    db: AsyncSession,
    *,
    since: datetime,
) -> dict:
    """
    Returns total count, fraud count, and fraud rate for labelled transactions
    in a given time window. Used by the monitoring performance endpoint.
    """
    stmt = (
        select(
            func.count().label("total"),
            func.sum(FactTransaction.fraud_label).label("fraud_count"),
        )
        .where(
            and_(
                FactTransaction.txn_timestamp >= since,
                FactTransaction.fraud_label.is_not(None),
            )
        )
    )
    result = await db.execute(stmt)
    row = result.one()
    total = row.total or 0
    fraud_count = int(row.fraud_count or 0)
    fraud_rate = fraud_count / total if total > 0 else 0.0
    return {"total": total, "fraud_count": fraud_count, "fraud_rate": fraud_rate}

"""Authenticated dashboard data; empty databases return empty results."""
from datetime import datetime, timedelta, timezone
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from src.api.dependencies import AsyncDBSession
from src.database.models import FactTransaction, DimUser, PredictionLog
from src.api.auth import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"], dependencies=[Depends(get_current_user)])

def latest_predictions():
    return select(PredictionLog.log_id, PredictionLog.transaction_id, PredictionLog.risk_label,
        func.row_number().over(partition_by=PredictionLog.transaction_id,
            order_by=(PredictionLog.created_at.desc(), PredictionLog.log_id.desc())).label("rank")).subquery()

@router.get("/stats")
async def get_dashboard_stats(db: AsyncDBSession):
    latest = latest_predictions()
    scored = select(latest.c.transaction_id, latest.c.risk_label).where(latest.c.rank == 1).subquery()
    result = (await db.execute(select(
        select(func.count()).select_from(DimUser).scalar_subquery().label("total_users"),
        select(func.count()).select_from(FactTransaction).scalar_subquery().label("stored_transactions"),
        select(func.count()).select_from(scored).scalar_subquery().label("total_transactions"),
        select(func.count()).select_from(scored).where(scored.c.risk_label == 1).scalar_subquery().label("flagged_transactions"),
        select(func.count()).select_from(scored).join(FactTransaction, FactTransaction.transaction_id == scored.c.transaction_id)
            .where(scored.c.risk_label == 1, FactTransaction.fraud_label == 1,
                   FactTransaction.labelled_at <= datetime.now(timezone.utc)).scalar_subquery().label("total_fraud_caught"),
        select(func.coalesce(func.sum(FactTransaction.amount_usd), 0)).select_from(FactTransaction)
            .join(scored, scored.c.transaction_id == FactTransaction.transaction_id).scalar_subquery().label("total_volume_usd")
    ))).mappings().one()
    return {**dict(result), "total_volume_usd": float(result["total_volume_usd"])}

@router.get("/chart-data")
async def get_chart_data(db: AsyncDBSession):
    since = datetime.now(timezone.utc) - timedelta(minutes=30)
    preds = (await db.scalars(select(PredictionLog).where(PredictionLog.created_at >= since)
        .order_by(PredictionLog.created_at.desc()).limit(200))).all()
    return [{"time": p.created_at.isoformat(), "score": round(float(p.fraud_score) * 100, 2)} for p in reversed(preds)]

@router.get("/recent-transactions")
async def get_recent_transactions(db: AsyncDBSession):
    rows = (await db.execute(select(PredictionLog, FactTransaction.amount_usd)
        .join(FactTransaction, PredictionLog.transaction_id == FactTransaction.transaction_id)
        .order_by(PredictionLog.created_at.desc()).limit(10))).all()
    return [{"id": str(p.transaction_id), "log_id": str(p.log_id), "score": round(float(p.fraud_score)*100, 2),
             "status": "Flagged" if p.risk_label else "Approved", "time": p.created_at.isoformat(),
             "amount_usd": float(amount), "currency": "USD"} for p, amount in rows]

@router.get("/transactions")
async def available_transactions(db: AsyncDBSession, limit: int = Query(20, ge=1, le=100)):
    rows = (await db.execute(select(FactTransaction.transaction_id, FactTransaction.amount_usd, FactTransaction.txn_timestamp)
        .order_by(FactTransaction.txn_timestamp.desc()).limit(limit))).all()
    return [{"id": str(tid), "amount_usd": float(amount), "time": ts.isoformat()} for tid, amount, ts in rows]

@router.get("/review-queue")
async def review_queue(db: AsyncDBSession):
    latest = latest_predictions()
    rows = (await db.execute(select(PredictionLog, FactTransaction.amount_usd).join(latest,
        latest.c.log_id == PredictionLog.log_id).join(FactTransaction,
        FactTransaction.transaction_id == PredictionLog.transaction_id).where(latest.c.rank == 1,
        PredictionLog.risk_label == 1, PredictionLog.is_reviewed.is_(False))
        .order_by(PredictionLog.created_at.desc()).limit(50))).all()
    return [{"id": str(p.transaction_id), "log_id": str(p.log_id), "score": float(p.fraud_score)*100,
             "time": p.created_at.isoformat(), "amount_usd": float(amount)} for p, amount in rows]

class ReviewRequest(BaseModel):
    notes: str = Field(default="", max_length=2000)

@router.post("/reviews/{log_id}")
async def mark_reviewed(log_id: UUID, payload: ReviewRequest, db: AsyncDBSession, user: str = Depends(get_current_user)):
    prediction = await db.get(PredictionLog, log_id)
    if prediction is None:
        raise HTTPException(404, "Prediction not found")
    prediction.is_reviewed = True
    prediction.reviewer_notes = f"{user}: {payload.notes}"
    await db.commit()
    return {"log_id": str(log_id), "is_reviewed": True}

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from src.database.session import get_db
from src.database.models import FactTransaction, DimUser, PredictionLog
from src.api.routes.auth import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db), _: str = Depends(get_current_user)):
    total_users = db.scalar(select(func.count(DimUser.user_id))) or 0
    total_txns = db.scalar(select(func.count(FactTransaction.transaction_id))) or 0
    fraud_txns = db.scalar(select(func.count(PredictionLog.log_id)).where(PredictionLog.risk_label == 1)) or 0
    
    total_volume = db.scalar(select(func.sum(FactTransaction.amount_usd))) or 0.0
    
    return {
        "total_users": total_users,
        "total_transactions": total_txns,
        "total_fraud_caught": fraud_txns,
        "total_volume_usd": float(total_volume)
    }

@router.get("/chart-data")
def get_chart_data(db: Session = Depends(get_db), _: str = Depends(get_current_user)):
    # Get last 20 predictions
    preds = db.scalars(
        select(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(20)
    ).all()
    
    chart_data = []
    # Reverse to show chronological left-to-right on chart
    for p in reversed(preds):
        time_str = p.created_at.strftime("%H:%M:%S")
        chart_data.append({
            "time": time_str,
            "score": round(p.fraud_score * 100, 2)
        })
        
    # If no data, return some fallback mock data
    if not chart_data:
        chart_data = [
            {"time": "10:00", "score": 12},
            {"time": "10:05", "score": 25},
            {"time": "10:10", "score": 15},
            {"time": "10:15", "score": 45},
            {"time": "10:20", "score": 85},
            {"time": "10:25", "score": 20},
            {"time": "10:30", "score": 35},
        ]
        
    return chart_data

@router.get("/recent-transactions")
def get_recent_transactions(db: Session = Depends(get_db), _: str = Depends(get_current_user)):
    # Get last 10 predictions joined with their fact_transaction
    query = (
        select(PredictionLog, FactTransaction.amount)
        .join(FactTransaction, PredictionLog.transaction_id == FactTransaction.transaction_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(10)
    )
    
    results = db.execute(query).all()
    
    recent = []
    for pred, amount in results:
        recent.append({
            "id": str(pred.transaction_id)[:8] + "...",
            "score": round(pred.fraud_score * 100, 2),
            "status": "Flagged" if pred.risk_label == 1 else "Approved",
            "time": pred.created_at.strftime("%H:%M:%S"),
            "amount": f"${float(amount):.2f}"
        })
        
    # If no data, return mock data
    if not recent:
        recent = [
            { "id": 'txn-1', "score": 85, "status": 'Flagged', "time": '10:20:15', "amount": '$1,200.00' },
            { "id": 'txn-2', "score": 12, "status": 'Approved', "time": '10:20:12', "amount": '$45.00' },
            { "id": 'txn-3', "score": 25, "status": 'Approved', "time": '10:19:45', "amount": '$89.99' },
            { "id": 'txn-4', "score": 45, "status": 'Review', "time": '10:18:22', "amount": '$350.00' },
        ]
        
    return recent

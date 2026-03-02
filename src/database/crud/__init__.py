"""
src/database/crud/__init__.py
Exposes all CRUD functions at the crud package level for clean imports.
"""

from src.database.crud.predictions import (
    get_active_model,
    get_model_by_version,
    get_prediction_log_by_id,
    get_predictions_for_transaction,
    get_recent_predictions,
    list_model_versions,
    log_prediction,
    mark_prediction_reviewed,
    promote_model,
    register_model,
    resolve_prediction_outcome,
)
from src.database.crud.transactions import (
    count_transactions_in_window,
    create_transaction,
    get_fraud_rate_summary,
    get_labelled_transactions_for_training,
    get_transaction_by_external_id,
    get_transaction_by_id,
    get_transactions_by_merchant,
    get_transactions_by_user,
    get_unlabelled_transactions,
    set_fraud_label,
    update_transaction_status,
)

__all__ = [
    # Transactions
    "create_transaction",
    "get_transaction_by_id",
    "get_transaction_by_external_id",
    "get_transactions_by_user",
    "get_transactions_by_merchant",
    "get_unlabelled_transactions",
    "get_labelled_transactions_for_training",
    "update_transaction_status",
    "set_fraud_label",
    "count_transactions_in_window",
    "get_fraud_rate_summary",
    # Predictions
    "log_prediction",
    "get_prediction_log_by_id",
    "get_predictions_for_transaction",
    "get_recent_predictions",
    "resolve_prediction_outcome",
    "mark_prediction_reviewed",
    # Model Registry
    "register_model",
    "get_active_model",
    "get_model_by_version",
    "list_model_versions",
    "promote_model",
]

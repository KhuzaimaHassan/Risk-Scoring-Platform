"""
src/features/__init__.py
-------------------------
Public API for the feature engineering layer.

External consumers (training, inference, tests) should import from here
rather than from individual submodules, to maintain a stable interface.
"""

from src.features.feature_aggregation import (
    AGGREGATED_FEATURE_NAMES,
    compute_all_aggregated_features,
    compute_behavioral_features,
    compute_merchant_rolling_features,
    compute_user_rolling_features,
)
from src.features.feature_extraction import (
    RAW_FEATURE_NAMES,
    extract_merchant_features,
    extract_raw_features,
    extract_temporal_features,
    extract_transaction_features,
    extract_user_features,
)
from src.features.feature_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FeaturePipeline,
)

__all__ = [
    # Pipeline (primary interface)
    "FeaturePipeline",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    # Extraction
    "extract_raw_features",
    "extract_transaction_features",
    "extract_user_features",
    "extract_merchant_features",
    "extract_temporal_features",
    "RAW_FEATURE_NAMES",
    # Aggregation
    "compute_all_aggregated_features",
    "compute_user_rolling_features",
    "compute_merchant_rolling_features",
    "compute_behavioral_features",
    "AGGREGATED_FEATURE_NAMES",
]

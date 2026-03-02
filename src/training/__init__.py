"""
src/training/__init__.py
Exposes the primary training API.
"""

from src.training.evaluate import (
    compute_binary_metrics,
    evaluate_model,
    find_optimal_threshold,
    save_evaluation_report,
)
from src.training.model_registry import ModelRegistry
from src.training.save_model import (
    generate_version_tag,
    load_model,
    load_model_with_metadata,
    load_metadata,
    save_model,
)
from src.training.train import TrainConfig, load_train_config, train

__all__ = [
    # Training orchestration
    "train",
    "TrainConfig",
    "load_train_config",
    # Evaluation
    "evaluate_model",
    "compute_binary_metrics",
    "find_optimal_threshold",
    "save_evaluation_report",
    # Model persistence
    "save_model",
    "load_model",
    "load_metadata",
    "load_model_with_metadata",
    "generate_version_tag",
    # Registry
    "ModelRegistry",
]

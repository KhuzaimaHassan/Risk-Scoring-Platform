"""
scripts/train_model.py
-----------------------
CLI entrypoint for training the fraud detection model.

Usage:
    # Default config (reads config/training.yaml):
    python scripts/train_model.py

    # With custom settings:
    python scripts/train_model.py \\
        --n-estimators 500 \\
        --test-size 0.20 \\
        --split time \\
        --auto-promote

    # With a custom version tag:
    python scripts/train_model.py --version v20260301_manual

    # Inside Docker:
    docker-compose exec api python scripts/train_model.py --auto-promote

After training:
    1. Model artifact saved to:   models/model_{version}.pkl
    2. Metadata saved to:         models/metadata_{version}.json
    3. Eval report saved to:      reports/eval_{version}.json
    4. Registry updated at:       models/registry.json

To promote a trained model manually:
    python -c "
    from pathlib import Path
    from src.training.model_registry import ModelRegistry
    r = ModelRegistry(Path('models/'))
    r.promote_model('{version}', stage='production')
    print(r.summary())
    "
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

from src.training.train import TrainConfig, load_train_config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fraud detection risk scoring model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config override flags
    parser.add_argument(
        "--config", type=Path, default=Path("config/training.yaml"),
        help="Path to training YAML config file",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Custom version tag (default: auto-generated timestamp)",
    )

    # Hyperparameter overrides
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument(
        "--split", type=str, choices=["time", "random"], default=None,
        help="Train/test split strategy",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold for binary fraud label [0.0, 1.0]",
    )
    parser.add_argument(
        "--optimise-for", type=str, choices=["f1", "recall", "precision"], default=None,
        help="Metric to optimise decision threshold for",
    )

    # Behaviour
    parser.add_argument(
        "--auto-promote", action="store_true", default=False,
        help="Automatically promote new model to production after training",
    )
    parser.add_argument(
        "--description", type=str, default="",
        help="Optional release notes / description for this training run",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports"),
        help="Directory to save evaluation reports",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load base config from YAML
    cfg = load_train_config(args.config)

    # Apply CLI overrides (only if explicitly provided)
    if args.n_estimators is not None:  cfg.n_estimators = args.n_estimators
    if args.max_depth    is not None:  cfg.max_depth    = args.max_depth
    if args.test_size    is not None:  cfg.test_size    = args.test_size
    if args.split        is not None:  cfg.split_strategy = args.split
    if args.threshold    is not None:  cfg.decision_threshold = args.threshold
    if args.optimise_for is not None:  cfg.optimise_threshold_for = args.optimise_for

    cfg.auto_promote       = args.auto_promote
    cfg.description        = args.description
    cfg.model_artifact_dir = args.model_dir
    cfg.report_dir         = args.report_dir

    # Run training
    start = time.perf_counter()
    result = train(cfg=cfg, version=args.version)
    elapsed = time.perf_counter() - start

    # Print summary to stdout
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Version      : {result['version']}")
    print(f"  AUC-ROC      : {result['metrics'].get('roc_auc', 0):.4f}")
    print(f"  AUC-PR       : {result['metrics'].get('pr_auc', 0):.4f}")
    print(f"  F1           : {result['metrics'].get('f1', 0):.4f}")
    print(f"  Recall       : {result['metrics'].get('recall', 0):.4f}")
    print(f"  Precision    : {result['metrics'].get('precision', 0):.4f}")
    print(f"  Elapsed      : {elapsed:.1f}s")
    print(f"  Artifact     : {result['artifact_path']}")
    print(f"  Eval report  : {result['eval_report_path']}")
    print(f"  Promoted     : {'[YES - production]' if result['promoted'] else '[NO  - candidate]'}")
    print("=" * 55)

    if not result["promoted"]:
        print(f"\n  To promote this model to production, run:")
        print(f"  python -c \"")
        print(f"  from pathlib import Path; from src.training.model_registry import ModelRegistry")
        print(f"  ModelRegistry(Path('models/')).promote_model('{result['version']}')\"")
    print()


if __name__ == "__main__":
    main()

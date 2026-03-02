"""
scripts/seed_db.py
-------------------
CLI entrypoint for seeding the Risk Scoring Platform database
with synthetic transaction data.

Usage:
    # Default (1,000 users / 100 merchants / 50,000 txns / 3% fraud):
    python scripts/seed_db.py

    # Custom scale:
    python scripts/seed_db.py --users 500 --transactions 10000 --fraud-ratio 0.04

    # Inside Docker:
    docker-compose exec api python scripts/seed_db.py
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so imports work whether run
# via `python scripts/seed_db.py` or `docker-compose exec api python scripts/seed_db.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import GeneratorConfig, seed_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the Risk Scoring Platform database with synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--users", type=int, default=1_000,
        help="Number of synthetic users to generate",
    )
    parser.add_argument(
        "--merchants", type=int, default=100,
        help="Number of synthetic merchants to generate",
    )
    parser.add_argument(
        "--transactions", type=int, default=50_000,
        help="Number of transactions to generate",
    )
    parser.add_argument(
        "--fraud-ratio", type=float, default=0.03,
        help="Target fraud ratio [0.0–1.0]",
    )
    parser.add_argument(
        "--label-coverage", type=float, default=0.85,
        help="Fraction of transactions to receive a fraud label (rest are NULL)",
    )
    parser.add_argument(
        "--simulation-days", type=int, default=180,
        help="Number of days to span transactions across",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1_000,
        help="DB insert batch size (tune for memory vs. speed)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (use -1 for non-deterministic)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = GeneratorConfig(
        n_users=args.users,
        n_merchants=args.merchants,
        n_transactions=args.transactions,
        fraud_ratio=args.fraud_ratio,
        label_coverage=args.label_coverage,
        simulation_days=args.simulation_days,
        batch_size=args.batch_size,
        random_seed=None if args.seed == -1 else args.seed,
    )

    start = time.perf_counter()
    seed_database(cfg)
    elapsed = time.perf_counter() - start
    print(f"\n✅ Database seeded in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

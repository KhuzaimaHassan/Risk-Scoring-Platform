"""
src/data/__init__.py
Exposes the primary generator entry points for external consumers.
"""

from src.data.synthetic_generator import (
    GeneratorConfig,
    MerchantProfile,
    UserProfile,
    generate_merchants,
    generate_transactions,
    generate_users,
    inject_fraud_patterns,
    seed_database,
)

__all__ = [
    "GeneratorConfig",
    "UserProfile",
    "MerchantProfile",
    "generate_users",
    "generate_merchants",
    "generate_transactions",
    "inject_fraud_patterns",
    "seed_database",
]

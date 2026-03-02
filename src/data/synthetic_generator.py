"""
src/data/synthetic_generator.py
---------------------------------
Production-grade synthetic transaction data generator.

Generates semi-realistic behavioural data for the Risk Scoring Platform:
  - Users with distinct spending profiles (low / medium / high tier)
  - Merchants with category-specific amount distributions
  - 50,000 transactions spanning 6 months with realistic temporal patterns
  - Fraud injection (~3% by default) using 4 distinct fraud archetypes

Design principles:
  - No random.uniform() scatter-gun logic — every number comes from a
    profile-driven distribution (lognormal, gamma, Poisson, etc.)
  - Fraud patterns mirror real-world attack archetypes documented in
    industry fraud reports (CNP, account takeover, synthetic identity)
  - Bulk-inserts via SQLAlchemy Core (not ORM) for performance at 50K rows
  - Fully configurable via GeneratorConfig dataclass

Usage:
    # Seed entire database
    python scripts/seed_db.py

    # Or import and call directly
    from src.data.synthetic_generator import seed_database, GeneratorConfig
    cfg = GeneratorConfig(n_users=500, n_transactions=10_000, fraud_ratio=0.04)
    seed_database(cfg)
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from faker import Faker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """
    Central configuration for the synthetic data generator.
    Change these values to scale up/down the dataset or adjust fraud prevalence.
    """
    # Dataset scale
    n_users: int = 1_000
    n_merchants: int = 100
    n_transactions: int = 50_000

    # Fraud
    fraud_ratio: float = 0.03         # Target proportion of fraudulent rows
    label_coverage: float = 0.85      # Fraction of txns that have a fraud_label (rest stay NULL)

    # Time window  (6 months ending at generation time)
    simulation_days: int = 180

    # Behavioural mixing — proportions must sum to 1.0
    user_tier_weights: dict[str, float] = field(default_factory=lambda: {
        "low": 0.50,     # Infrequent, small-spend users
        "medium": 0.35,  # Regular shoppers
        "high": 0.15,    # Power users / high-value customers
    })
    txn_freq_weights: dict[str, float] = field(default_factory=lambda: {
        "rare": 0.40,      # ~1–5 txns / month
        "moderate": 0.40,  # ~5–20 txns / month
        "frequent": 0.20,  # ~20–60 txns / month
    })

    # Database batch commit size (rows per flush)
    batch_size: int = 1_000

    # Random seed for reproducibility (set None for non-deterministic)
    random_seed: int | None = 42


# ---------------------------------------------------------------------------
# Internal profile definitions (not exported — used by generator functions)
# ---------------------------------------------------------------------------

#: Spending profiles per user tier
#  Each profile drives a lognormal distribution: amount ~ lognormal(mu, sigma)
#  mu / sigma chosen so that e^(mu + sigma²/2) ≈ the listed mean_usd.
_USER_SPEND_PROFILES: dict[str, dict[str, Any]] = {
    "low":    {"mean_usd": 35.0,   "sigma": 0.55, "max_usd": 500.0},
    "medium": {"mean_usd": 180.0,  "sigma": 0.70, "max_usd": 3_000.0},
    "high":   {"mean_usd": 850.0,  "sigma": 0.85, "max_usd": 25_000.0},
}

_USER_FREQ_PROFILES: dict[str, dict[str, float]] = {
    "rare":     {"lambda": 2.5},   # Poisson λ = expected txns/month
    "moderate": {"lambda": 12.0},
    "frequent": {"lambda": 38.0},
}

#: Merchant category → typical transaction amount distribution (lognormal)
_MERCHANT_CATEGORY_PROFILES: dict[str, dict[str, Any]] = {
    "retail":        {"mean_usd": 65.0,    "sigma": 0.60, "is_high_risk": False},
    "food_beverage": {"mean_usd": 28.0,    "sigma": 0.45, "is_high_risk": False},
    "travel":        {"mean_usd": 520.0,   "sigma": 0.80, "is_high_risk": False},
    "electronics":   {"mean_usd": 380.0,   "sigma": 0.70, "is_high_risk": False},
    "healthcare":    {"mean_usd": 210.0,   "sigma": 0.65, "is_high_risk": False},
    "utilities":     {"mean_usd": 120.0,   "sigma": 0.35, "is_high_risk": False},
    "gambling":      {"mean_usd": 175.0,   "sigma": 1.05, "is_high_risk": True},
    "crypto":        {"mean_usd": 950.0,   "sigma": 1.20, "is_high_risk": True},
    "other":         {"mean_usd": 55.0,    "sigma": 0.55, "is_high_risk": False},
}

#: Merchant category distribution (realistic mix skewed toward everyday categories)
_MERCHANT_CATEGORY_DIST: list[tuple[str, float]] = [
    ("retail",        0.28),
    ("food_beverage", 0.25),
    ("travel",        0.10),
    ("electronics",   0.10),
    ("healthcare",    0.08),
    ("utilities",     0.07),
    ("other",         0.07),
    ("gambling",      0.03),
    ("crypto",        0.02),
]

#: Channel / payment method distributions (cumulative probability)
_CHANNEL_WEIGHTS = ["web", "mobile", "pos", "atm", "api"]
_CHANNEL_PROBS   = [0.30,  0.35,    0.25,  0.05,  0.05]

_PAYMENT_WEIGHTS = ["credit_card", "debit_card", "bank_transfer", "crypto", "wallet", "bnpl"]
_PAYMENT_PROBS   = [0.35,          0.30,         0.15,            0.03,     0.12,     0.05]

#: Countries represented in the simulation (ISO 3166-1 alpha-2)
_COUNTRIES = ["US", "GB", "DE", "FR", "CA", "AU", "IN", "SG", "BR", "MX"]
_COUNTRY_WEIGHTS = [0.35, 0.12, 0.10, 0.08, 0.10, 0.06, 0.06, 0.04, 0.05, 0.04]

#: Fraud archetypes — each has a weight and a generation strategy key
_FRAUD_ARCHETYPES: list[dict[str, Any]] = [
    {"name": "high_amount",         "weight": 0.35},  # Amount > 3× user's average
    {"name": "rapid_velocity",      "weight": 0.25},  # 3+ txns within 5 minutes
    {"name": "foreign_country",     "weight": 0.20},  # Merchant country ≠ user country
    {"name": "high_risk_merchant",  "weight": 0.20},  # Gambling/crypto + elevated amount
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sha256(value: str) -> str:
    """Return lowercase hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _lognormal_amount(mean_usd: float, sigma: float, max_usd: float, rng: np.random.Generator) -> float:
    """
    Draw a transaction amount from a lognormal distribution.
    Converts the desired mean into lognormal mu parameter, then clips to max_usd.
    """
    mu = math.log(mean_usd) - 0.5 * sigma ** 2
    raw = rng.lognormal(mean=mu, sigma=sigma)
    return float(min(round(raw, 2), max_usd))


def _weighted_choice(choices: list[str], weights: list[float], rng: np.random.Generator) -> str:
    """Pick one item from choices according to probability weights."""
    cumulative = np.cumsum(weights)
    r = rng.random()
    for choice, cum_w in zip(choices, cumulative):
        if r <= cum_w:
            return choice
    return choices[-1]


def _random_timestamp(start: datetime, end: datetime, rng: np.random.Generator) -> datetime:
    """Return a uniformly random timezone-aware UTC datetime within [start, end]."""
    delta_seconds = int((end - start).total_seconds())
    offset = int(rng.integers(0, max(delta_seconds, 1)))
    return start + timedelta(seconds=offset)


def _apply_time_of_day_bias(ts: datetime, rng: np.random.Generator) -> datetime:
    """
    Bias transaction times toward realistic human activity patterns:
    - Peak hours: 10:00–13:00 and 18:00–22:00 local time
    - Low activity: 01:00–06:00
    Implemented by occasionally replacing a random timestamp with one
    sampled from a business-hours Gaussian.
    """
    if rng.random() < 0.70:  # 70% of txns follow business-hour pattern
        # Sample hour from bimodal Gaussian (lunch peak ~11, evening peak ~20)
        peak = rng.choice([11.0, 20.0])
        hour_float = rng.normal(loc=peak, scale=1.5)
        hour_float = float(np.clip(hour_float, 0, 23.99))
        hour = int(hour_float)
        minute = int((hour_float - hour) * 60)
        return ts.replace(hour=hour, minute=minute, second=int(rng.integers(0, 60)))
    return ts


# ---------------------------------------------------------------------------
# User profiles (in-memory data structures before DB insert)
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """In-memory representation of a synthetic user before DB insert."""
    user_id: uuid.UUID
    external_id: str
    full_name: str
    email_hash: str
    country_code: str
    account_age_days: int
    is_active: bool
    kyc_verified: bool
    risk_tier: str
    credit_score: float | None
    spend_tier: str          # "low" | "medium" | "high"  (drives amounts)
    freq_tier: str           # "rare" | "moderate" | "frequent"
    avg_monthly_txns: float


@dataclass
class MerchantProfile:
    """In-memory representation of a synthetic merchant before DB insert."""
    merchant_id: uuid.UUID
    external_id: str
    merchant_name: str
    category: str
    country_code: str
    city: str
    is_online_only: bool
    is_active: bool
    risk_level: str
    is_high_risk_category: bool
    avg_transaction_amount: float


# ---------------------------------------------------------------------------
# Core Generator Functions
# ---------------------------------------------------------------------------

def generate_users(
    cfg: GeneratorConfig,
    fake: Faker,
    rng: np.random.Generator,
) -> list[UserProfile]:
    """
    Generate `cfg.n_users` user profiles with stratified spending and
    frequency tiers.

    Distribution:
    - 50% low-spend / infrequent
    - 35% medium-spend / moderate
    - 15% high-spend / frequent

    Risk tier correlated with spend tier:
    - Low spenders → mostly LOW risk
    - High spenders → mostly MEDIUM/HIGH risk (more attractive fraud targets)
    """
    logger.info("Generating %d user profiles …", cfg.n_users)

    spend_tiers = list(cfg.user_tier_weights.keys())
    spend_probs = list(cfg.user_tier_weights.values())
    freq_tiers = list(cfg.txn_freq_weights.keys())
    freq_probs = list(cfg.txn_freq_weights.values())

    # Risk tier conditional on spend tier
    _risk_by_spend: dict[str, tuple[list[str], list[float]]] = {
        "low":    (["low", "medium", "high", "blocked"], [0.65, 0.28, 0.05, 0.02]),
        "medium": (["low", "medium", "high", "blocked"], [0.35, 0.48, 0.14, 0.03]),
        "high":   (["low", "medium", "high", "blocked"], [0.20, 0.42, 0.33, 0.05]),
    }

    users: list[UserProfile] = []
    for i in range(cfg.n_users):
        spend_tier = _weighted_choice(spend_tiers, spend_probs, rng)
        freq_tier = _weighted_choice(freq_tiers, freq_probs, rng)
        country = _weighted_choice(_COUNTRIES, _COUNTRY_WEIGHTS, rng)

        risk_choices, risk_probs = _risk_by_spend[spend_tier]
        risk_tier = _weighted_choice(risk_choices, risk_probs, rng)

        # Credit score: lognormal-ish, correlated with spend tier
        has_credit = rng.random() > 0.15  # 85% have a credit score
        if has_credit:
            mean_score: float = {"low": 620.0, "medium": 700.0, "high": 760.0}[spend_tier]
            raw_score = rng.normal(loc=mean_score, scale=55.0)
            credit_score: float | None = float(np.clip(round(raw_score, 2), 300.0, 850.0))
        else:
            credit_score = None

        # Account age: older accounts skew toward higher-spend tiers
        mean_age: float = {"low": 365.0, "medium": 900.0, "high": 1800.0}[spend_tier]
        account_age = int(max(1, rng.normal(loc=mean_age, scale=mean_age * 0.3)))

        # KYC: correlated with account age and risk tier
        kyc_prob = 0.95 if account_age > 730 else 0.70
        if risk_tier in ("high", "blocked"):
            kyc_prob = min(kyc_prob, 0.55)
        kyc_verified = rng.random() < kyc_prob

        profile = _USER_FREQ_PROFILES[freq_tier]
        avg_monthly_txns = float(rng.poisson(profile["lambda"]))

        full_name = fake.name()
        email = fake.email()

        users.append(UserProfile(
            user_id=uuid.uuid4(),
            external_id=f"USR-{i+1:06d}",
            full_name=full_name,
            email_hash=_sha256(email.lower().strip()),
            country_code=country,
            account_age_days=account_age,
            is_active=rng.random() > 0.03,   # 3% inactive
            kyc_verified=kyc_verified,
            risk_tier=risk_tier,
            credit_score=credit_score,
            spend_tier=spend_tier,
            freq_tier=freq_tier,
            avg_monthly_txns=max(avg_monthly_txns, 0.5),
        ))

    logger.info("  → %d users generated.", len(users))
    return users


def generate_merchants(
    cfg: GeneratorConfig,
    fake: Faker,
    rng: np.random.Generator,
) -> list[MerchantProfile]:
    """
    Generate `cfg.n_merchants` merchant profiles.

    Merchant category distribution follows `_MERCHANT_CATEGORY_DIST`.
    High-risk merchants (gambling/crypto) receive elevated risk_level.
    """
    logger.info("Generating %d merchant profiles …", cfg.n_merchants)

    categories = [c for c, _ in _MERCHANT_CATEGORY_DIST]
    cat_probs = [p for _, p in _MERCHANT_CATEGORY_DIST]

    merchants: list[MerchantProfile] = []
    for i in range(cfg.n_merchants):
        category = _weighted_choice(categories, cat_probs, rng)
        cat_profile = _MERCHANT_CATEGORY_PROFILES[category]
        is_high_risk = cat_profile["is_high_risk"]
        country = _weighted_choice(_COUNTRIES, _COUNTRY_WEIGHTS, rng)

        # Risk level: high-risk categories are always HIGH; others follow normal dist
        if is_high_risk:
            risk_level = _weighted_choice(["medium", "high"], [0.3, 0.7], rng)
        else:
            risk_level = _weighted_choice(["low", "medium", "high"], [0.45, 0.45, 0.10], rng)

        # Online-only flag: e-commerce / high-value categories trend online
        online_prob = {"gambling": 0.95, "crypto": 1.00, "travel": 0.70,
                       "electronics": 0.50}.get(category, 0.25)
        is_online_only = rng.random() < online_prob

        avg_amount = _lognormal_amount(
            cat_profile["mean_usd"], cat_profile["sigma"], 50_000.0, rng
        )

        merchants.append(MerchantProfile(
            merchant_id=uuid.uuid4(),
            external_id=f"MER-{i+1:04d}",
            merchant_name=fake.company(),
            category=category,
            country_code=country,
            city=fake.city(),
            is_online_only=is_online_only,
            is_active=rng.random() > 0.02,
            risk_level=risk_level,
            is_high_risk_category=is_high_risk,
            avg_transaction_amount=avg_amount,
        ))

    logger.info("  → %d merchants generated.", len(merchants))
    return merchants


def generate_transactions(
    cfg: GeneratorConfig,
    users: list[UserProfile],
    merchants: list[MerchantProfile],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """
    Generate `cfg.n_transactions` raw transaction dicts (not yet labelled).

    Allocation strategy:
    1. Compute each user's expected share of total transactions based on
       their `avg_monthly_txns` (Poisson-weighted).
    2. For each user's transactions, sample merchants with a preference
       for their home region (80% same-country, 20% foreign).
    3. Transaction amount drawn from user's spend-profile lognormal,
       modulated by merchant category.
    4. Timestamps distributed across the simulation window with
       business-hour bias.

    Returns a list of dicts ready for bulk insert.
    """
    logger.info("Generating %d transactions …", cfg.n_transactions)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=cfg.simulation_days)

    # --- Step 1: Compute per-user transaction allocations ---
    raw_weights = np.array([u.avg_monthly_txns for u in users], dtype=float)
    raw_weights /= raw_weights.sum()
    # Draw allocation counts that sum to n_transactions
    counts = rng.multinomial(cfg.n_transactions, raw_weights)

    # --- Step 2: Build a same-country merchant index for faster sampling ---
    country_merchant_map: dict[str, list[MerchantProfile]] = {}
    for m in merchants:
        country_merchant_map.setdefault(m.country_code, []).append(m)

    # --- Step 3: Generate transactions per user ---
    transactions: list[dict[str, Any]] = []
    for user, n_txns in zip(users, counts):
        if n_txns == 0:
            continue

        spend_profile = _USER_SPEND_PROFILES[user.spend_tier]

        # Pre-generate timestamps for this user in sorted order
        raw_timestamps = sorted([
            _apply_time_of_day_bias(
                _random_timestamp(start_dt, end_dt, rng), rng
            )
            for _ in range(n_txns)
        ])

        for ts in raw_timestamps:
            # Choose merchant with same-country preference
            same_country = country_merchant_map.get(user.country_code, [])
            use_same_country = same_country and rng.random() < 0.80
            merchant: MerchantProfile = (
                rng.choice(same_country) if use_same_country  # type: ignore[arg-type]
                else rng.choice(merchants)  # type: ignore[arg-type]
            )

            is_international = merchant.country_code != user.country_code

            # Amount: blend user spend profile with merchant category profile
            cat_profile = _MERCHANT_CATEGORY_PROFILES[merchant.category]
            blended_mean = (spend_profile["mean_usd"] * 0.6 + cat_profile["mean_usd"] * 0.4)
            blended_sigma = (spend_profile["sigma"] * 0.6 + cat_profile["sigma"] * 0.4)
            blended_max = max(spend_profile["max_usd"], cat_profile.get("mean_usd", 100) * 5)
            amount = _lognormal_amount(blended_mean, blended_sigma, blended_max, rng)

            # Channel / payment method correlated with merchant type
            if merchant.is_online_only:
                channel = _weighted_choice(["web", "mobile", "api"], [0.45, 0.40, 0.15], rng)
                payment = _weighted_choice(
                    ["credit_card", "debit_card", "wallet", "bnpl"],
                    [0.40, 0.25, 0.22, 0.13], rng
                )
            else:
                channel = _weighted_choice(_CHANNEL_WEIGHTS, _CHANNEL_PROBS, rng)
                payment = _weighted_choice(_PAYMENT_WEIGHTS, _PAYMENT_PROBS, rng)

            # Pick a currency based on merchant country (simplified mapping)
            currency_map = {"US": "USD", "GB": "GBP", "DE": "EUR", "FR": "EUR",
                            "CA": "CAD", "AU": "AUD", "IN": "INR", "SG": "SGD",
                            "BR": "BRL", "MX": "MXN"}
            currency = currency_map.get(merchant.country_code, "USD")

            # Simplified FX (fixed rates for simulation — not real market rates)
            fx_to_usd = {"USD": 1.0, "GBP": 1.27, "EUR": 1.10, "CAD": 0.74,
                         "AUD": 0.66, "INR": 0.012, "SGD": 0.74, "BRL": 0.20, "MXN": 0.058}
            amount_usd = round(amount * fx_to_usd.get(currency, 1.0), 4)

            # Status: most complete; small fraction pending/declined
            status_rand = rng.random()
            if status_rand < 0.01:
                status = "pending"
            elif status_rand < 0.04:
                status = "declined"
            else:
                status = "completed"

            # Generate hashed device / IP tokens
            ip_hash = _sha256(f"{user.external_id}-{rng.integers(0, 1000)}")
            device_hash = _sha256(f"{user.external_id}-device-{rng.integers(0, 20)}")

            transactions.append({
                "transaction_id": uuid.uuid4(),
                "external_txn_id": f"TXN-{uuid.uuid4().hex[:16].upper()}",
                "user_id": user.user_id,
                "merchant_id": merchant.merchant_id,
                "txn_timestamp": ts,
                "amount": amount,
                "currency": currency,
                "amount_usd": amount_usd,
                "status": status,
                "channel": channel,
                "payment_method": payment,
                "ip_address_hash": ip_hash,
                "device_fingerprint_hash": device_hash,
                "is_international": is_international,
                "fraud_label": None,     # Set later by inject_fraud_patterns()
                "labelled_at": None,
                "review_notes": None,
                "_user_profile": user,   # Temporary reference — stripped before insert
                "_merchant_profile": merchant,
            })

    logger.info("  → %d transactions generated (pre-labelling).", len(transactions))
    return transactions


def inject_fraud_patterns(
    transactions: list[dict[str, Any]],
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """
    Label a subset of transactions as fraudulent using 4 realistic archetypes.

    Fraud archetypes:
    ─────────────────
    1. HIGH_AMOUNT      — Amount ≥ 3× the user's typical average. Represents
                          Account Takeover (ATO) where a fraudster maxes out a card.

    2. RAPID_VELOCITY   — Injects a cluster of 3–6 transactions within a
                          5-minute window for the same user. Represents card
                          testing or automated bot transactions.

    3. FOREIGN_COUNTRY  — Marks an international transaction as fraud with elevated
                          probability if the user has never transacted internationally
                          before (simulated by checking their profile).

    4. HIGH_RISK_MERCHANT — Gambling/crypto transaction at elevated amount (>2× user avg).
                            Represents money laundering or synthetic identity fraud.

    Post-labelling:
    - `cfg.label_coverage` fraction of all transactions receive a label
      (0 or 1). The rest remain NULL to simulate delayed ops review.
    - Legitimate transactions are labelled 0; fraudulent ones labelled 1.
    - A `labelled_at` timestamp is set within 1–72 hours of the transaction.
    """
    logger.info("Injecting fraud patterns (target ratio: %.1f%%) …", cfg.fraud_ratio * 100)

    target_fraud_count = int(len(transactions) * cfg.fraud_ratio)

    # --- Build user → transactions index for velocity checks ---
    user_txn_index: dict[uuid.UUID, list[int]] = {}
    for idx, txn in enumerate(transactions):
        uid = txn["_user_profile"].user_id
        user_txn_index.setdefault(uid, []).append(idx)

    fraud_indices: set[int] = set()
    archetype_names = [a["name"] for a in _FRAUD_ARCHETYPES]
    archetype_weights = [a["weight"] for a in _FRAUD_ARCHETYPES]
    archetype_cum_weights = list(np.cumsum(archetype_weights))

    attempts = 0
    max_attempts = target_fraud_count * 20  # Safety limit

    while len(fraud_indices) < target_fraud_count and attempts < max_attempts:
        attempts += 1
        archetype = _weighted_choice(archetype_names, archetype_weights, rng)

        if archetype == "high_amount":
            # Pick a random non-fraud transaction and inflate the amount
            idx = int(rng.integers(0, len(transactions)))
            if idx in fraud_indices:
                continue
            txn = transactions[idx]
            user: UserProfile = txn["_user_profile"]
            spend_profile = _USER_SPEND_PROFILES[user.spend_tier]
            # Fraudulent amount: 3× to 8× the user's typical mean
            fraud_multiplier = float(rng.uniform(3.0, 8.0))
            new_amount = round(spend_profile["mean_usd"] * fraud_multiplier, 2)
            new_amount = max(new_amount, txn["amount"] * 2.0)  # Always higher than original
            # Clamp to a plausible max
            new_amount = min(new_amount, spend_profile["max_usd"])
            new_amount_usd = round(new_amount * (txn["amount_usd"] / txn["amount"]), 4)
            transactions[idx] = {**txn, "amount": new_amount, "amount_usd": new_amount_usd}
            fraud_indices.add(idx)

        elif archetype == "rapid_velocity":
            # Pick a user with multiple transactions, inject a tight cluster
            uid = rng.choice(list(user_txn_index.keys()))  # type: ignore
            user_indices = user_txn_index[uid]
            if len(user_indices) < 3:
                continue
            # Pick a base index and force the next 2–5 txns to be within 5 minutes
            base_pos = int(rng.integers(0, max(1, len(user_indices) - 5)))
            cluster_size = int(rng.integers(3, min(6, len(user_indices) - base_pos + 1)))
            base_idx = user_indices[base_pos]
            base_ts: datetime = transactions[base_idx]["txn_timestamp"]
            for offset in range(cluster_size):
                target_idx = user_indices[base_pos + offset]
                if target_idx in fraud_indices:
                    continue
                jitter = timedelta(seconds=int(rng.integers(10, 300)))  # within 5 min
                transactions[target_idx] = {
                    **transactions[target_idx],
                    "txn_timestamp": base_ts + jitter,
                }
                fraud_indices.add(target_idx)

        elif archetype == "foreign_country":
            # Pick a predominantly domestic user, mark an international txn as fraud
            idx = int(rng.integers(0, len(transactions)))
            if idx in fraud_indices:
                continue
            txn = transactions[idx]
            merchant: MerchantProfile = txn["_merchant_profile"]
            user = txn["_user_profile"]
            if merchant.country_code == user.country_code:
                # Create a foreign merchant for this txn
                all_countries = [c for c in _COUNTRIES if c != user.country_code]
                foreign_country = rng.choice(all_countries)  # type: ignore
                merchant_copy = MerchantProfile(
                    **{**merchant.__dict__, "country_code": foreign_country}
                )
                transactions[idx] = {
                    **txn,
                    "is_international": True,
                    "_merchant_profile": merchant_copy,
                }
            fraud_indices.add(idx)

        elif archetype == "high_risk_merchant":
            # Find transactions at high-risk merchants and inflate amounts
            high_risk_indices = [
                i for i, t in enumerate(transactions)
                if t["_merchant_profile"].is_high_risk_category and i not in fraud_indices
            ]
            if not high_risk_indices:
                continue
            idx = int(rng.choice(high_risk_indices))  # type: ignore
            txn = transactions[idx]
            user = txn["_user_profile"]
            spend_profile = _USER_SPEND_PROFILES[user.spend_tier]
            fraud_multiplier = float(rng.uniform(2.0, 5.0))
            new_amount = round(spend_profile["mean_usd"] * fraud_multiplier, 2)
            new_amount_usd = round(new_amount * (txn["amount_usd"] / txn["amount"]), 4)
            transactions[idx] = {**txn, "amount": new_amount, "amount_usd": new_amount_usd}
            fraud_indices.add(idx)

    actual_fraud = len(fraud_indices)
    logger.info("  → %d fraud transactions injected (%.2f%% of total).",
                actual_fraud, actual_fraud / len(transactions) * 100)

    # --- Apply labels with coverage ---
    now = datetime.now(timezone.utc)
    labelled_indices = set(
        rng.choice(len(transactions), size=int(len(transactions) * cfg.label_coverage), replace=False)
    )

    for idx, txn in enumerate(transactions):
        if idx in labelled_indices:
            is_fraud = idx in fraud_indices
            label = 1 if is_fraud else 0
            label_delay = timedelta(hours=float(rng.uniform(1.0, 72.0)))
            review = "Flagged by automated rule engine." if is_fraud else None
            transactions[idx] = {
                **txn,
                "fraud_label": label,
                "labelled_at": txn["txn_timestamp"] + label_delay,
                "review_notes": review,
            }

    return transactions


# ---------------------------------------------------------------------------
# Database Insertion
# ---------------------------------------------------------------------------

def _strip_internal_keys(txn: dict[str, Any]) -> dict[str, Any]:
    """Remove generator-internal keys (prefixed with _) before DB insert."""
    return {k: v for k, v in txn.items() if not k.startswith("_")}


def seed_database(cfg: GeneratorConfig | None = None) -> None:
    """
    Main entry point — generates all synthetic data and bulk-inserts into PostgreSQL.

    Execution steps:
    1. Validate config and initialise RNG / Faker
    2. Generate user profiles → insert to dim_user
    3. Generate merchant profiles → insert to dim_merchant
    4. Generate raw transactions → inject fraud patterns → insert to fact_transaction
    5. Update denormalised aggregates on dim_user and dim_merchant

    Uses sync SQLAlchemy session (not async) since this is a script context.
    Inserts in configurable batches to avoid OOM and long transactions.
    """
    if cfg is None:
        cfg = GeneratorConfig()

    logger.info("=" * 60)
    logger.info("Risk Scoring Platform — Synthetic Data Generator")
    logger.info("Config: users=%d, merchants=%d, transactions=%d, fraud=%.1f%%",
                cfg.n_users, cfg.n_merchants, cfg.n_transactions, cfg.fraud_ratio * 100)
    logger.info("=" * 60)

    # --- Initialise randomness ---
    if cfg.random_seed is not None:
        logger.info("Using fixed random seed: %d (fully reproducible run).", cfg.random_seed)
        random.seed(cfg.random_seed)
    rng = np.random.default_rng(cfg.random_seed)
    fake = Faker()
    if cfg.random_seed is not None:
        Faker.seed(cfg.random_seed)

    # --- Import DB dependencies here to avoid circular imports at module level ---
    from sqlalchemy import text
    from src.database.session import sync_db_context
    from src.database.models import (  # noqa: F401 — registers models with Base
        DimUser, DimMerchant, FactTransaction,
        RiskTier, MerchantCategory, MerchantRiskLevel,
        TransactionStatus, TransactionChannel, PaymentMethod,
    )

    # --- Generate in-memory data ---
    users = generate_users(cfg, fake, rng)
    merchants = generate_merchants(cfg, fake, rng)
    transactions = generate_transactions(cfg, users, merchants, rng)
    transactions = inject_fraud_patterns(transactions, cfg, rng)

    # --- Insert to DB ---
    with sync_db_context() as db:
        # Use Core INSERT via text() — bypasses ORM enum coercion which sends
        # Python enum member names (e.g. "MEDIUM") instead of values ("medium").

        # 1. Insert users
        logger.info("Inserting %d users …", len(users))
        for i in range(0, len(users), cfg.batch_size):
            batch = users[i:i + cfg.batch_size]
            db.execute(
                text("""
                    INSERT INTO dim_user (
                        user_id, external_id, full_name, email_hash,
                        country_code, account_age_days, is_active, kyc_verified,
                        risk_tier, credit_score, lifetime_txn_count, lifetime_txn_volume
                    ) VALUES (
                        :user_id, :external_id, :full_name, :email_hash,
                        :country_code, :account_age_days, :is_active, :kyc_verified,
                        CAST(:risk_tier AS risk_tier_enum), :credit_score, :lifetime_txn_count, :lifetime_txn_volume
                    )
                """),
                [
                    {
                        "user_id": str(u.user_id),
                        "external_id": u.external_id,
                        "full_name": u.full_name,
                        "email_hash": u.email_hash,
                        "country_code": u.country_code,
                        "account_age_days": u.account_age_days,
                        "is_active": u.is_active,
                        "kyc_verified": u.kyc_verified,
                        "risk_tier": u.risk_tier,   # already lowercase string
                        "credit_score": u.credit_score,
                        "lifetime_txn_count": 0,
                        "lifetime_txn_volume": 0,
                    }
                    for u in batch
                ]
            )
            db.flush()
        logger.info("  ✓ Users inserted.")

        # 2. Insert merchants
        logger.info("Inserting %d merchants …", len(merchants))
        for i in range(0, len(merchants), cfg.batch_size):
            batch = merchants[i:i + cfg.batch_size]
            db.execute(
                text("""
                    INSERT INTO dim_merchant (
                        merchant_id, external_id, merchant_name, category,
                        country_code, city, is_online_only, is_active,
                        risk_level, is_high_risk_category,
                        historical_fraud_rate, avg_transaction_amount, total_txn_count
                    ) VALUES (
                        :merchant_id, :external_id, :merchant_name, CAST(:category AS merchant_category_enum),
                        :country_code, :city, :is_online_only, :is_active,
                        CAST(:risk_level AS merchant_risk_level_enum), :is_high_risk_category,
                        :historical_fraud_rate, :avg_transaction_amount, :total_txn_count
                    )
                """),
                [
                    {
                        "merchant_id": str(m.merchant_id),
                        "external_id": m.external_id,
                        "merchant_name": m.merchant_name,
                        "category": m.category,
                        "country_code": m.country_code,
                        "city": m.city,
                        "is_online_only": m.is_online_only,
                        "is_active": m.is_active,
                        "risk_level": m.risk_level,
                        "is_high_risk_category": m.is_high_risk_category,
                        "historical_fraud_rate": 0.0,
                        "avg_transaction_amount": m.avg_transaction_amount,
                        "total_txn_count": 0,
                    }
                    for m in batch
                ]
            )
            db.flush()
        logger.info("  ✓ Merchants inserted.")

        # 3. Insert transactions in batches
        logger.info("Inserting %d transactions in batches of %d …",
                    len(transactions), cfg.batch_size)
        for i in range(0, len(transactions), cfg.batch_size):
            batch = [_strip_internal_keys(t) for t in transactions[i:i + cfg.batch_size]]
            db.execute(
                text("""
                    INSERT INTO fact_transaction (
                        transaction_id, external_txn_id, user_id, merchant_id,
                        txn_timestamp, amount, currency, amount_usd,
                        status, channel, payment_method,
                        ip_address_hash, device_fingerprint_hash,
                        is_international, fraud_label, labelled_at, review_notes
                    ) VALUES (
                        :transaction_id, :external_txn_id, :user_id, :merchant_id,
                        :txn_timestamp, :amount, :currency, :amount_usd,
                        CAST(:status AS txn_status_enum), CAST(:channel AS txn_channel_enum),
                        CAST(:payment_method AS payment_method_enum),
                        :ip_address_hash, :device_fingerprint_hash,
                        :is_international, :fraud_label, :labelled_at, :review_notes
                    )
                """),
                [
                    {
                        "transaction_id": str(t["transaction_id"]),
                        "external_txn_id": t["external_txn_id"],
                        "user_id": str(t["user_id"]),
                        "merchant_id": str(t["merchant_id"]),
                        "txn_timestamp": t["txn_timestamp"],
                        "amount": t["amount"],
                        "currency": t["currency"],
                        "amount_usd": t["amount_usd"],
                        "status": t["status"],
                        "channel": t["channel"],
                        "payment_method": t["payment_method"],
                        "ip_address_hash": t["ip_address_hash"],
                        "device_fingerprint_hash": t["device_fingerprint_hash"],
                        "is_international": t["is_international"],
                        "fraud_label": t.get("fraud_label"),
                        "labelled_at": t.get("labelled_at"),
                        "review_notes": t.get("review_notes"),
                    }
                    for t in batch
                ]
            )
            db.flush()
            if (i // cfg.batch_size + 1) % 10 == 0:
                logger.info("  … %d / %d transactions inserted.", i + len(batch), len(transactions))
        logger.info("  ✓ Transactions inserted.")


        # 4. Update denormalised aggregates on dim_user
        logger.info("Updating denormalised aggregates on dim_user …")
        db.execute(text("""
            UPDATE dim_user u
            SET
                lifetime_txn_count  = agg.cnt,
                lifetime_txn_volume = agg.vol
            FROM (
                SELECT user_id, COUNT(*) AS cnt, SUM(amount_usd) AS vol
                FROM   fact_transaction
                GROUP  BY user_id
            ) agg
            WHERE u.user_id = agg.user_id
        """))

        # 5. Update denormalised aggregates on dim_merchant
        logger.info("Updating denormalised aggregates on dim_merchant …")
        db.execute(text("""
            UPDATE dim_merchant m
            SET
                total_txn_count        = agg.cnt,
                avg_transaction_amount = agg.avg_amt,
                historical_fraud_rate  = agg.fraud_rate
            FROM (
                SELECT
                    merchant_id,
                    COUNT(*)                                        AS cnt,
                    AVG(amount_usd)                                 AS avg_amt,
                    COALESCE(AVG(fraud_label::FLOAT), 0.0)          AS fraud_rate
                FROM   fact_transaction
                GROUP  BY merchant_id
            ) agg
            WHERE m.merchant_id = agg.merchant_id
        """))
        logger.info("  ✓ Aggregates updated.")

    # --- Final summary ---
    fraud_count = sum(1 for t in transactions if t.get("fraud_label") == 1)
    legitimate_count = sum(1 for t in transactions if t.get("fraud_label") == 0)
    unlabelled_count = sum(1 for t in transactions if t.get("fraud_label") is None)

    logger.info("=" * 60)
    logger.info("Seeding complete!")
    logger.info("  Users inserted      : %d", len(users))
    logger.info("  Merchants inserted  : %d", len(merchants))
    logger.info("  Transactions total  : %d", len(transactions))
    logger.info("    ├── Fraud (label=1)   : %d  (%.2f%%)",
                fraud_count, fraud_count / len(transactions) * 100)
    logger.info("    ├── Legit (label=0)   : %d  (%.2f%%)",
                legitimate_count, legitimate_count / len(transactions) * 100)
    logger.info("    └── Unlabelled (NULL) : %d  (%.2f%%)",
                unlabelled_count, unlabelled_count / len(transactions) * 100)
    logger.info("=" * 60)

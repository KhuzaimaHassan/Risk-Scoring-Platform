"""
config/settings.py
-------------------
Centralised application configuration using Pydantic BaseSettings.

All settings are read from environment variables (or .env file).
A cached singleton is returned via get_settings() for zero-overhead
access across the application.

Naming convention:
  - All env vars are SCREAMING_SNAKE_CASE.
  - All Python attributes are snake_case.
  - Sensitive values (passwords, keys) use SecretStr.
"""

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Usage:
        from config.settings import get_settings
        settings = get_settings()

    Anywhere in FastAPI:
        from config.settings import get_settings
        from fastapi import Depends

        def my_endpoint(settings = Depends(get_settings)):
            ...
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",            # Ignore unknown env vars — don't raise
    )

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------
    app_name: str = Field(
        default="Risk Scoring Platform",
        description="Human-readable application name shown in API docs",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Semantic version of the application",
    )
    environment: str = Field(
        default="development",
        description="Runtime environment: development | staging | production",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (never True in production)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )

    # -----------------------------------------------------------------------
    # API Server
    # -----------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="Bind host for uvicorn")
    api_port: int = Field(default=8000, description="Bind port for uvicorn")
    api_workers: int = Field(default=1, description="Number of uvicorn worker processes")
    api_reload: bool = Field(default=False, description="Enable auto-reload (dev only)")
    api_prefix: str = Field(default="/api/v1", description="Global API route prefix")

    # -----------------------------------------------------------------------
    # PostgreSQL (sync + async)
    # -----------------------------------------------------------------------
    db_host: str = Field(default="localhost", description="PostgreSQL host")
    db_port: int = Field(default=5432, description="PostgreSQL port")
    db_name: str = Field(default="risk_scoring", description="Database name")
    db_user: str = Field(default="risk_user", description="Database user")
    db_password: SecretStr = Field(
        default=SecretStr(""),
        description="Database password — use SecretStr to prevent log leakage",
    )
    db_pool_size: int = Field(
        default=10,
        description="SQLAlchemy connection pool size",
    )
    db_max_overflow: int = Field(
        default=20,
        description="Max connections beyond pool_size",
    )
    db_echo: bool = Field(
        default=False,
        description="Echo SQL queries to log (dev only, never in prod)",
    )

    # -----------------------------------------------------------------------
    # Model Registry & Artifact Storage
    # -----------------------------------------------------------------------
    model_artifact_dir: Path = Field(
        default=Path("models"),
        description="Directory for serialised model artifacts",
    )
    model_report_dir: Path = Field(
        default=Path("reports"),
        description="Directory for evaluation reports (JSON, plots)",
    )
    active_model_name: str = Field(
        default="fraud_classifier",
        description="Model family name used for production scoring",
    )

    # -----------------------------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------------------------
    feature_window_1h_minutes: int = Field(
        default=60,
        description="Rolling window for 1-hour aggregations (minutes)",
    )
    feature_window_24h_minutes: int = Field(
        default=1440,
        description="Rolling window for 24-hour aggregations (minutes)",
    )
    feature_window_7d_minutes: int = Field(
        default=10080,
        description="Rolling window for 7-day aggregations (minutes)",
    )

    # -----------------------------------------------------------------------
    # Prediction / Inference
    # -----------------------------------------------------------------------
    default_decision_threshold: float = Field(
        default=0.5,
        description="Default score threshold for binary fraud label",
        ge=0.0,
        le=1.0,
    )
    max_batch_size: int = Field(
        default=500,
        description="Maximum transactions per batch prediction request",
    )

    # -----------------------------------------------------------------------
    # Monitoring & Drift Detection
    # -----------------------------------------------------------------------
    drift_psi_threshold_warning: float = Field(
        default=0.1,
        description="PSI threshold for WARNING-level drift alert",
        ge=0.0,
    )
    drift_psi_threshold_critical: float = Field(
        default=0.2,
        description="PSI threshold for CRITICAL-level drift alert",
        ge=0.0,
    )
    drift_ks_p_value_threshold: float = Field(
        default=0.05,
        description="KS test p-value below which drift is flagged",
        ge=0.0,
        le=1.0,
    )
    monitoring_window_hours: int = Field(
        default=24,
        description="Hours of recent predictions used for monitoring checks",
    )
    performance_f1_min_threshold: float = Field(
        default=0.70,
        description="Minimum F1 below which retraining is triggered",
        ge=0.0,
        le=1.0,
    )

    # -----------------------------------------------------------------------
    # Retraining
    # -----------------------------------------------------------------------
    retraining_min_samples: int = Field(
        default=10_000,
        description="Minimum labelled samples required before a retrain",
    )
    retraining_auto_promote: bool = Field(
        default=False,
        description="Auto-promote retrained model to production if metrics improve",
    )

    # -----------------------------------------------------------------------
    # Security
    # -----------------------------------------------------------------------
    secret_key: SecretStr = Field(
        default=SecretStr("change-this-in-production"),
        description="Secret key for signing tokens / CSRF protection",
    )
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="CORS allowed origins list",
    )

    # -----------------------------------------------------------------------
    # Computed properties (not injected from env)
    # -----------------------------------------------------------------------

    @property
    def database_url(self) -> str:
        """Synchronous DSN for Alembic and scripts (psycopg2 driver)."""
        pwd = self.db_password.get_secret_value()
        auth = f"{self.db_user}:{pwd}" if pwd else self.db_user
        return f"postgresql+psycopg2://{auth}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def async_database_url(self) -> str:
        """Async DSN for FastAPI runtime (asyncpg driver)."""
        pwd = self.db_password.get_secret_value()
        auth = f"{self.db_user}:{pwd}" if pwd else self.db_user
        return f"postgresql+asyncpg://{auth}@{self.db_host}:{self.db_port}/{self.db_name}"

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @model_validator(mode="after")
    def enforce_prod_safety(self) -> "Settings":
        """Prevent known-insecure defaults from reaching production."""
        if self.environment == "production":
            if self.debug:
                raise ValueError("debug=True is not allowed in production")
            if self.db_echo:
                raise ValueError("db_echo=True is not allowed in production")
            if self.secret_key.get_secret_value() == "change-this-in-production":
                raise ValueError("Default secret_key must not be used in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings singleton.
    Cache is invalidated on process restart — safe for production.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()

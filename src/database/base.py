"""
src/database/base.py
--------------------
Shared SQLAlchemy declarative base and abstract mixin classes.
All ORM models must import from here — never redeclare Base elsewhere.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Project-wide declarative base. All ORM models inherit from this."""
    pass


class TimestampMixin:
    """
    Reusable mixin that adds audit timestamp columns to any model.
    created_at  — set once at INSERT time (server default)
    updated_at  — automatically updated on every UPDATE
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Row creation timestamp (UTC)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Row last-update timestamp (UTC)",
    )

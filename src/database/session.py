"""
src/database/session.py
------------------------
Database engine construction and session management.

Design decisions:
- Uses SQLAlchemy 2.x async engine for non-blocking I/O.
- Sync engine is provided for Alembic migrations and scripts.
- Sessions are scoped per-request via FastAPI dependency injection.
- Connection pool tuned for a production containerised environment.
"""

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from config.settings import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Async engine  (used by FastAPI endpoints at runtime)
# ---------------------------------------------------------------------------
async_engine = create_async_engine(
    settings.async_database_url,          # postgresql+asyncpg://...
    pool_size=settings.db_pool_size,       # default 10
    max_overflow=settings.db_max_overflow, # default 20
    pool_pre_ping=True,                    # recycle stale connections
    pool_recycle=3600,                     # recycle after 1 hour
    echo=settings.db_echo,                 # SQL query logging (only in dev)
    future=True,
)

AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,   # avoid implicit lazy-loads after commit
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# Sync engine  (used by Alembic, scripts, and background jobs)
# ---------------------------------------------------------------------------
sync_engine = create_engine(
    settings.database_url,                 # postgresql+psycopg2://...
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.db_echo,
    future=True,
)

SyncSessionFactory: sessionmaker[Session] = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency — yields an async session per HTTP request
# ---------------------------------------------------------------------------
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI route handlers.

    Usage:
        @router.get("/example")
        async def handler(db: AsyncSession = Depends(get_async_db)):
            ...
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Context manager — for scripts and background tasks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def async_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for use outside of FastAPI (CLI scripts, jobs).

    Usage:
        async with async_db_context() as db:
            result = await db.execute(...)
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@contextmanager
def sync_db_context() -> Generator[Session, None, None]:
    """
    Sync context manager for Alembic env.py and plain Python scripts.

    Usage:
        with sync_db_context() as db:
            db.execute(text("SELECT 1"))
    """
    session = SyncSessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Health-check helper
# ---------------------------------------------------------------------------
async def check_db_connectivity() -> bool:
    """
    Fires a lightweight query to verify the database is reachable.
    Used by the /health endpoint.
    """
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

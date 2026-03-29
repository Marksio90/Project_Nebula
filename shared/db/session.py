"""
shared/db/session.py
─────────────────────────────────────────────────────────────────────────────
Async SQLAlchemy engine + session factory.

Usage (inside async code):
    async with get_db() as session:
        result = await session.execute(select(Mix).where(Mix.id == mix_id))
        mix = result.scalar_one_or_none()

Usage (inside Celery tasks — sync context):
    with get_sync_db() as session:
        session.add(mix)
        session.commit()
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from shared.config import get_settings

log = logging.getLogger("nebula.db.session")
settings = get_settings()

# ── Async engine (FastAPI, async code paths) ──────────────────────────────────

_async_engine = create_async_engine(
    str(settings.postgres_dsn),
    echo=settings.environment == "development",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,          # Discard stale connections automatically
    pool_recycle=3600,           # Recycle connections older than 1 hour
)

_AsyncSessionLocal = async_sessionmaker(
    bind=_async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a transactional AsyncSession."""
    async with _AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Sync engine (Celery task context — no event loop available) ───────────────
# Uses psycopg2 driver (not asyncpg) — swap scheme for sync compatibility.

_sync_dsn = str(settings.postgres_dsn).replace(
    "postgresql+asyncpg://", "postgresql+psycopg2://"
)

_sync_engine = create_engine(
    _sync_dsn,
    echo=settings.environment == "development",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

_SyncSessionLocal = sessionmaker(
    bind=_sync_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """
    Sync context manager for Celery task code paths.
    Yields a transactional Session; commits on success, rolls back on error.
    """
    session: Session = _SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

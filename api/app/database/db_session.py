from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import asyncio

import os
from typing import Any

# Database configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "airflow")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")
POSTGRES_DB = os.getenv("POSTGRES_DB", "airflow")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
DATABASE_STARTUP_RETRIES = int(os.getenv("DATABASE_STARTUP_RETRIES", "12"))
DATABASE_STARTUP_RETRY_DELAY_SECONDS = float(os.getenv("DATABASE_STARTUP_RETRY_DELAY_SECONDS", "5"))

# Build database URL with proper password interpolation
POSTGRES_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

Base = declarative_base()

engine: Any = None
AsyncSessionLocal: Any = None
_engine_initialization_error: Exception | None = None


def _configure_engine() -> None:
    global engine, AsyncSessionLocal, _engine_initialization_error

    if engine is not None and AsyncSessionLocal is not None:
        return

    try:
        engine = create_async_engine(POSTGRES_URL, echo=False, future=True, pool_pre_ping=True)
        AsyncSessionLocal = sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        _engine_initialization_error = None
    except ModuleNotFoundError as exc:
        engine = None
        AsyncSessionLocal = None
        _engine_initialization_error = exc


def _require_engine():
    if engine is None:
        _configure_engine()

    if engine is None:
        detail = str(_engine_initialization_error) if _engine_initialization_error else "unknown error"
        raise RuntimeError(
            "Database engine is not available. Install the PostgreSQL async driver "
            f"from api/requirements.txt and retry. Original error: {detail}"
        )

    return engine


def _require_sessionmaker():
    if AsyncSessionLocal is None:
        _configure_engine()

    if AsyncSessionLocal is None:
        detail = str(_engine_initialization_error) if _engine_initialization_error else "unknown error"
        raise RuntimeError(
            "Database session factory is not available. Install the PostgreSQL async driver "
            f"from api/requirements.txt and retry. Original error: {detail}"
        )

    return AsyncSessionLocal


_configure_engine()


async def wait_for_database(
    retries: int | None = None,
    delay_seconds: float | None = None,
) -> None:
    attempts = retries if retries is not None else DATABASE_STARTUP_RETRIES
    delay = delay_seconds if delay_seconds is not None else DATABASE_STARTUP_RETRY_DELAY_SECONDS

    if attempts < 1:
        attempts = 1

    for attempt in range(1, attempts + 1):
        try:
            async with _require_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
            return
        except Exception:
            if attempt >= attempts:
                raise
            await asyncio.sleep(delay)

async def init_models():
    async with _require_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.commit()

async def get_db():
    async with _require_sessionmaker()() as session:
        yield session
    

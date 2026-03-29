"""
migrations/env.py — Alembic environment configuration.
Supports both offline (SQL dump) and online (live DB) migration modes.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

from shared.config import get_settings

# Alembic Config object
config = context.config
settings = get_settings()

# Override sqlchemy.url from our Settings (avoids duplicating it in alembic.ini)
config.set_main_option("sqlalchemy.url", str(settings.postgres_dsn))

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so Alembic can detect changes
from shared.db.models import Base  # noqa: E402  (must be after Base import)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

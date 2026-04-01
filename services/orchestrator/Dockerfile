##############################################################################
# Orchestrator — CrewAI + GPT-4o multi-agent crew + Celery worker
##############################################################################
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/orchestrator/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY shared /app/shared
COPY services/orchestrator /app/services/orchestrator
COPY tests /app/tests
# Alembic migration files — required by the db_migrate service
COPY alembic.ini /app/alembic.ini
COPY migrations /app/migrations

ENV PYTHONPATH=/app

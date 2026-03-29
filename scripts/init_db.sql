-- Project Nebula — PostgreSQL bootstrap
-- Runs once on first container start via docker-entrypoint-initdb.d

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";     -- used by subgenre fuzzy search

-- The application's Alembic migrations handle table creation.
-- This file only ensures the schema/extensions are available.

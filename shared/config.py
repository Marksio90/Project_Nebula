"""
shared/config.py
─────────────────────────────────────────────────────────────────────────────
Centralised Pydantic-Settings configuration.
Loaded once per process; consumed by every service and shared module.
─────────────────────────────────────────────────────────────────────────────
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ───────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Database ──────────────────────────────────────────────────────────
    postgres_dsn: PostgresDsn = Field(
        ...,
        alias="POSTGRES_DSN",
        description="Full asyncpg DSN: postgresql+asyncpg://user:pass@host/db",
    )

    # ── Broker ────────────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(..., alias="REDIS_URL")

    # ── AI APIs ───────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")

    # ── Social / Distribution ─────────────────────────────────────────────
    # Step 1 — downloaded from Google Cloud Console (OAuth 2.0 client ID)
    youtube_client_secrets_file: str = Field(
        default="/secrets/youtube_client_secrets.json",
        alias="YOUTUBE_CLIENT_SECRETS_FILE",
    )
    # Step 2 — generated once by running: python scripts/youtube_auth.py
    # Contains refresh_token; used for all subsequent autonomous uploads.
    youtube_token_file: str = Field(
        default="/secrets/youtube_token.json",
        alias="YOUTUBE_TOKEN_FILE",
    )
    tiktok_api_key: str = Field(default="", alias="TIKTOK_API_KEY")

    # ── Storage Paths ─────────────────────────────────────────────────────
    stems_dir: str = Field(default="/mnt/stems", alias="STEMS_DIR")
    mixes_dir: str = Field(default="/mnt/mixes", alias="MIXES_DIR")
    visuals_dir: str = Field(default="/mnt/visuals", alias="VISUALS_DIR")
    exports_dir: str = Field(default="/mnt/exports", alias="EXPORTS_DIR")

    # ── FFmpeg ────────────────────────────────────────────────────────────────
    ffmpeg_hwaccel: str = Field(
        default="auto",
        alias="FFMPEG_HWACCEL",
        description="Hardware acceleration: auto | nvenc | vaapi | none",
    )

    # ── Celery Tuning ─────────────────────────────────────────────────────
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_task_track_started: bool = True
    celery_task_acks_late: bool = True          # Re-queue on worker crash
    celery_worker_prefetch_multiplier: int = 1  # One task at a time per DSP slot

    # ── CrewAI ────────────────────────────────────────────────────────────
    crewai_verbose: bool = Field(default=False, alias="CREWAI_VERBOSE")

    # ── Gemini / Generation Limits ────────────────────────────────────────
    lyria_stem_duration_seconds: int = 30       # Gemini Lyria 3 output length
    max_stems_per_mix: int = 96                 # 96 × 30 s = 48 min max
    target_lufs: float = -14.0                  # Streaming loudness target
    true_peak_dbfs: float = -1.0                # True peak ceiling

    @field_validator("postgres_dsn", mode="before")
    @classmethod
    def coerce_asyncpg_scheme(cls, v: str) -> str:
        """Ensure the DSN uses the asyncpg driver scheme."""
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()

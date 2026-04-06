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

    # Replicate — used by ReplicateMusicGenProvider (default audio backend)
    replicate_api_token: str = Field(default="", alias="REPLICATE_API_TOKEN")
    # Replicate model ID for music generation.
    # Set this if the default model has been deprecated or moved.
    # Find current models at: https://replicate.com/explore?category=audio
    # Common options (verify on Replicate dashboard):
    #   meta/musicgen  — newest namespace
    #   facebook/musicgen-stereo-large  — original (may be archived)
    replicate_model: str = Field(
        default="meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
        alias="REPLICATE_MODEL",
    )

    # ── Media Provider Selection ──────────────────────────────────────────
    # Production stack — change only if adding a new provider implementation.
    #   audio_provider : replicate  → MusicGen via Replicate (~$0.008/stem)
    #   image_provider : dalle3     → DALL-E 3 via OpenAI    (~$0.080/image)
    #   video_provider : ffmpeg     → Ken Burns animation     (FREE, local)
    audio_provider: str = Field(default="replicate", alias="AUDIO_PROVIDER")
    image_provider: str = Field(default="dalle3",    alias="IMAGE_PROVIDER")
    video_provider: str = Field(default="ffmpeg",    alias="VIDEO_PROVIDER")

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

    # LLM model for CrewAI agents.
    # gpt-4o-mini is ~16x cheaper than gpt-4o and handles all orchestration
    # tasks (strategy, prompts, SEO) very well.
    # Swap to gpt-4o or claude-sonnet-4-6 for higher-quality output.
    # Any LiteLLM-compatible model string works:
    #   gpt-4o-mini           (OpenAI, cheapest good option)
    #   gpt-4o                (OpenAI, expensive but top quality)
    #   claude-haiku-4-5-20251001  (Anthropic, ultra-cheap)
    #   claude-sonnet-4-6          (Anthropic, balanced)
    # ── LLM model assignment strategy ─────────────────────────────────────
    # Three tiers, each controllable independently via env vars:
    #
    #   llm_model        → Audio PE, Visual PE (English prompts, formulaic structure)
    #                      gpt-4o-mini is sufficient — tasks are repetitive and constrained
    #
    #   llm_precise_model → CSO strategy + all QA agents (structured JSON only)
    #                       gpt-4o-mini is sufficient — no creative writing, pure logic
    #
    #   llm_seo_model     → Polish SEO agent ONLY (user-facing viral copy on YouTube/TikTok)
    #                       Defaults to gpt-4o — Polish language quality and viral hook
    #                       writing are directly tied to channel growth and revenue.
    #                       This is the only LLM call the audience ever sees.
    #                       Swap to gpt-4o-mini to save ~$0.01/mix if quality is acceptable.
    #
    # Cost per mix at defaults (~150 stems):
    #   Audio PE (8 batches)  : ~32k tokens  → mini : $0.018
    #   CSO + QA agents       : ~2k tokens   → mini : $0.001
    #   Visual PE             : ~1.5k tokens → mini : $0.0005
    #   Polish SEO + QA       : ~2.5k tokens → 4o   : $0.025
    #   ─────────────────────────────────────────────────────
    #   Total LLM per mix     :                      ≈ $0.045
    #   Replicate MusicGen    :                      ≈ $1.20   (dominates)
    #   DALL-E 3 images       :                      ≈ $0.64
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_precise_model: str = Field(default="gpt-4o-mini", alias="LLM_PRECISE_MODEL")
    llm_seo_model: str = Field(default="gpt-4o", alias="LLM_SEO_MODEL")

    # ── Generation Limits ─────────────────────────────────────────────────
    stem_duration_seconds: int = 30              # Audio stem length in seconds
    max_stems_per_mix: int = 150                 # 150 × 30 s = 75 min max
    target_lufs: float = -14.0                  # Streaming loudness target
    true_peak_dbfs: float = -1.0                # True peak ceiling

    # ── Concurrent stem generation ─────────────────────────────────────────
    # How many Replicate predictions to poll simultaneously.
    # Free tier: 4–6 concurrent; paid tier: 50+.
    # Increase STEM_CONCURRENCY if your Replicate plan allows more.
    stem_concurrency: int = Field(default=8, alias="STEM_CONCURRENCY")
    # Delay (seconds) between consecutive prediction *submissions*.
    # 0.5 s is safe even on the free tier — submissions are cheap HTTP
    # POSTs; the heavy lifting (generation) happens asynchronously.
    replicate_submit_delay_s: float = Field(default=0.5, alias="REPLICATE_SUBMIT_DELAY_S")

    # ── Concurrent prompt generation ───────────────────────────────────────
    # Stems per LLM batch call.
    # Token budget (gpt-4o-mini): 16 384 max output tokens.
    # Per stem (worst case): ~150 tokens (80-word prompt + JSON keys).
    # 40 stems × 150 = 6 000 tokens → 37% of limit, safe headroom.
    # Code hard-caps this at 81 regardless of env setting
    # (= floor(16384 × 0.75 / 150)) — cannot exceed token limit.
    audio_prompt_batch_size: int = Field(default=40, alias="AUDIO_PROMPT_BATCH_SIZE")
    # How many LLM batch calls to fire simultaneously.
    # OpenAI tier-1+: 500 RPM — 10 concurrent batches is trivially safe.
    prompt_concurrency: int = Field(default=10, alias="PROMPT_CONCURRENCY")

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

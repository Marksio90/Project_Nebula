"""
shared/utils/retry.py
─────────────────────────────────────────────────────────────────────────────
Tenacity retry strategies for Project Nebula API calls.

Usage:
    @retry_gemini_api
    def call_lyria(prompt: str) -> bytes: ...

    @retry_openai_api
    def call_gpt4o(messages: list) -> str: ...
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import random

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

log = logging.getLogger("nebula.retry")


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception()
    log.warning(
        "Retry attempt %d/%d for %s — error: %s",
        retry_state.attempt_number,
        retry_state.retry_object.statistics.get("attempt_number", "?"),
        retry_state.fn.__name__ if retry_state.fn else "unknown",
        exc,
    )


# ── Gemini API (Lyria 3 / Nano Banana 2 / Veo) ────────────────────────────────
# Gemini uses HTTP 429 (quota) and 503 (overload) — both warrant exponential backoff

retry_gemini_api = retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(multiplier=2, min=4, max=120),
    before_sleep=_log_retry_attempt,
)

# ── OpenAI API (GPT-4o via CrewAI) ────────────────────────────────────────────
retry_openai_api = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1.5, min=2, max=60),
    before_sleep=_log_retry_attempt,
)

# ── YouTube Data API v3 ────────────────────────────────────────────────────────
# Quota resets daily — longer backoff on exhaustion
retry_youtube_api = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=300),
    before_sleep=_log_retry_attempt,
)

# ── TikTok API ─────────────────────────────────────────────────────────────────
retry_tiktok_api = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    before_sleep=_log_retry_attempt,
)

# ── Generic HTTP (any requests-based call) ────────────────────────────────────
retry_http = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, min=1, max=30),
    before_sleep=_log_retry_attempt,
)

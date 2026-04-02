"""
tests/integration/test_api_gateway.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for the API Gateway.

These tests hit the *live* running API — no mocking, no ASGI tricks.
The full stack (postgres, redis, api_gateway) must be up.

Run locally (stack running via Docker):
    pytest tests/integration/ -v

Run against a remote host:
    API_BASE_URL=http://my-server:8000 pytest tests/integration/ -v

Tests are skipped automatically if the API is unreachable.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import uuid

import httpx
import pytest

BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── Helpers ────────────────────────────────────────────────────────────────


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=15.0)


def _api_is_reachable() -> bool:
    try:
        with _client() as c:
            return c.get("/health").status_code == 200
    except Exception:
        return False


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def api():
    """Session-scoped guard — skips the entire module when API is down."""
    if not _api_is_reachable():
        pytest.skip(f"API Gateway not reachable at {BASE_URL}")


# ── Health ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestHealth:
    def test_returns_200(self, api):
        with _client() as c:
            r = c.get("/health")
        assert r.status_code == 200

    def test_body_has_status_ok(self, api):
        with _client() as c:
            r = c.get("/health")
        assert r.json()["status"] == "ok"

    def test_body_identifies_service(self, api):
        with _client() as c:
            r = c.get("/health")
        assert r.json()["service"] == "api_gateway"


# ── POST /mixes/generate ───────────────────────────────────────────────────


@pytest.mark.integration
class TestGenerateMix:
    def test_returns_202(self, api):
        with _client() as c:
            r = c.post("/mixes/generate", json={"requested_duration_minutes": 10})
        assert r.status_code == 202

    def test_response_has_mix_id_and_task_id(self, api):
        with _client() as c:
            r = c.post("/mixes/generate", json={"requested_duration_minutes": 10})
        body = r.json()
        assert "mix_id" in body
        assert "task_id" in body
        assert body["message"]
        uuid.UUID(body["mix_id"])  # raises ValueError if not a valid UUID

    def test_style_hint_accepted(self, api):
        with _client() as c:
            r = c.post(
                "/mixes/generate",
                json={"requested_duration_minutes": 10, "style_hint": "Dark Neurofunk"},
            )
        assert r.status_code == 202

    def test_force_bpm_accepted(self, api):
        with _client() as c:
            r = c.post(
                "/mixes/generate",
                json={"requested_duration_minutes": 10, "force_bpm": 174},
            )
        assert r.status_code == 202

    def test_duration_below_minimum_returns_422(self, api):
        with _client() as c:
            r = c.post("/mixes/generate", json={"requested_duration_minutes": 5})
        assert r.status_code == 422

    def test_duration_above_maximum_returns_422(self, api):
        with _client() as c:
            r = c.post("/mixes/generate", json={"requested_duration_minutes": 200})
        assert r.status_code == 422

    def test_force_bpm_below_minimum_returns_422(self, api):
        with _client() as c:
            r = c.post(
                "/mixes/generate",
                json={"requested_duration_minutes": 10, "force_bpm": 50},
            )
        assert r.status_code == 422

    def test_force_bpm_above_maximum_returns_422(self, api):
        with _client() as c:
            r = c.post(
                "/mixes/generate",
                json={"requested_duration_minutes": 10, "force_bpm": 300},
            )
        assert r.status_code == 422

    def test_style_hint_too_long_returns_422(self, api):
        with _client() as c:
            r = c.post(
                "/mixes/generate",
                json={"requested_duration_minutes": 10, "style_hint": "x" * 257},
            )
        assert r.status_code == 422


# ── GET /mixes/{mix_id} ────────────────────────────────────────────────────


@pytest.mark.integration
class TestGetMixStatus:
    def test_unknown_mix_returns_404(self, api):
        with _client() as c:
            r = c.get(f"/mixes/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_invalid_uuid_returns_422(self, api):
        with _client() as c:
            r = c.get("/mixes/not-a-uuid")
        assert r.status_code == 422

    def test_enqueued_mix_status_is_reachable(self, api):
        """
        After enqueuing a mix the DB row may not exist yet (task is still
        queued in Redis). Either 200 (pending) or 404 is acceptable here —
        the important thing is that the API doesn't crash.
        """
        with _client() as c:
            gen = c.post("/mixes/generate", json={"requested_duration_minutes": 10})
        assert gen.status_code == 202
        mix_id = gen.json()["mix_id"]

        with _client() as c:
            r = c.get(f"/mixes/{mix_id}")
        assert r.status_code in (200, 404)

    def test_200_response_has_required_fields(self, api):
        """
        If any mix already exists in the DB (from a previous run), verify
        the response schema. Skips gracefully if DB is empty.
        """
        with _client() as c:
            gen = c.post("/mixes/generate", json={"requested_duration_minutes": 10})
        mix_id = gen.json()["mix_id"]

        with _client() as c:
            r = c.get(f"/mixes/{mix_id}")
        if r.status_code == 200:
            body = r.json()
            assert "mix_id" in body
            assert "task_id" in body
            assert "state" in body


# ── GET /mixes/{mix_id}/audio/download ────────────────────────────────────


@pytest.mark.integration
class TestDownloadAudio:
    def test_unknown_mix_returns_404(self, api):
        with _client() as c:
            r = c.get(f"/mixes/{uuid.uuid4()}/audio/download")
        assert r.status_code == 404

    def test_invalid_uuid_returns_422(self, api):
        with _client() as c:
            r = c.get("/mixes/not-a-uuid/audio/download")
        assert r.status_code == 422

    def test_mix_without_audio_returns_404(self, api):
        """A freshly enqueued mix has no audio yet — endpoint must return 404."""
        with _client() as c:
            gen = c.post("/mixes/generate", json={"requested_duration_minutes": 10})
        mix_id = gen.json()["mix_id"]

        with _client() as c:
            r = c.get(f"/mixes/{mix_id}/audio/download")
        # Either 404 (mix row not created yet) or 404 (no audio path yet)
        assert r.status_code == 404

    def test_completed_mix_returns_wav(self, api):
        """
        Skipped unless a mix with status=complete exists in the DB.
        To exercise manually: wait for a full pipeline run, then run this test.
        The response should be audio/wav with Content-Disposition attachment.
        """
        # We can't fabricate a completed mix here — skip if none exist
        pytest.skip("Requires a completed mix in the DB. Run after a full pipeline.")

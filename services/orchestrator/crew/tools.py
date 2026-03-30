"""
services/orchestrator/crew/tools.py
─────────────────────────────────────────────────────────────────────────────
Custom CrewAI tools for Project Nebula agents.

Tools:
  BpmRegistryTool     — Queries PostgreSQL BpmSubgenreRegistry to surface
                        all previously used (bpm, subgenre, key) combos so
                        the CSO agent can guarantee novelty.
  YoutubeAnalyticsTool — Fetches view/CTR stats from the channel to inform
                         the CSO's strategic decision.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import select, text

from shared.db.models import BpmSubgenreRegistry, Mix, MixStatus
from shared.db.session import get_sync_db

log = logging.getLogger("nebula.crew.tools")


# ── BPM Registry Tool ─────────────────────────────────────────────────────────

class BpmRegistryInput(BaseModel):
    subgenre_filter: str | None = Field(
        default=None,
        description="Optional subgenre to filter by (e.g. 'Neurofunk'). Returns all if None.",
    )
    bpm_min: float = Field(default=60.0,  description="Minimum BPM to include in results.")
    bpm_max: float = Field(default=220.0, description="Maximum BPM to include in results.")
    limit: int     = Field(default=200,   description="Maximum number of registry entries to return.")


class BpmRegistryTool(BaseTool):
    """
    Query the BPM/Subgenre deduplication registry.

    Returns a JSON list of all previously used (bpm, subgenre, key_signature)
    combinations so the CSO can select a genuinely novel combination for the
    next mix. The CSO MUST NOT repeat any combination in this list.
    """
    name: str = "bpm_subgenre_registry"
    description: str = (
        "Query the music label's deduplication database. Returns all previously used "
        "(bpm, subgenre, key_signature) combinations. The CSO MUST select parameters "
        "that do NOT appear in this list to guarantee 0% content repetition."
    )
    args_schema: Type[BaseModel] = BpmRegistryInput

    def _run(
        self,
        subgenre_filter: str | None = None,
        bpm_min: float = 60.0,
        bpm_max: float = 220.0,
        limit: int = 200,
    ) -> str:
        try:
            with get_sync_db() as db:
                query = (
                    select(
                        BpmSubgenreRegistry.bpm,
                        BpmSubgenreRegistry.subgenre,
                        BpmSubgenreRegistry.key_signature,
                        BpmSubgenreRegistry.used_at,
                    )
                    .where(
                        BpmSubgenreRegistry.bpm >= bpm_min,
                        BpmSubgenreRegistry.bpm <= bpm_max,
                    )
                    .order_by(BpmSubgenreRegistry.used_at.desc())
                    .limit(limit)
                )
                if subgenre_filter:
                    query = query.where(
                        BpmSubgenreRegistry.subgenre.ilike(f"%{subgenre_filter}%")
                    )
                rows = db.execute(query).fetchall()

            used = [
                {
                    "bpm": r.bpm,
                    "subgenre": r.subgenre,
                    "key_signature": r.key_signature,
                    "used_at": r.used_at.isoformat() if r.used_at else None,
                }
                for r in rows
            ]
            log.info("BpmRegistryTool: returned %d used combinations", len(used))
            return json.dumps({"used_combinations": used, "total_count": len(used)})

        except Exception as exc:
            log.error("BpmRegistryTool error: %s", exc)
            return json.dumps({"error": str(exc), "used_combinations": []})


# ── YouTube Analytics Tool ────────────────────────────────────────────────────

class YoutubeAnalyticsInput(BaseModel):
    metric: str = Field(
        default="top_performing_subgenres",
        description=(
            "Metric to retrieve. One of: 'top_performing_subgenres', "
            "'recent_mix_performance', 'avg_view_duration_by_bpm'."
        ),
    )
    days_back: int = Field(default=90, description="Number of days of history to analyse.")


class YoutubeAnalyticsTool(BaseTool):
    """
    Retrieve YouTube Analytics data for the Nebula label channel.

    Returns performance metrics (views, CTR, avg watch time) broken down by
    subgenre and BPM range to inform the CSO's next mix strategy.
    """
    name: str = "youtube_analytics"
    description: str = (
        "Retrieve YouTube Analytics for the label channel. Use this to understand "
        "which subgenres and BPM ranges are currently performing best — and which "
        "are oversaturated. Informs CSO strategic decisions about mix direction."
    )
    args_schema: Type[BaseModel] = YoutubeAnalyticsInput

    def _run(self, metric: str = "top_performing_subgenres", days_back: int = 90) -> str:
        """
        Queries the platform_uploads table for upload performance as a proxy
        for analytics data. In production, wire to the YouTube Analytics API v2.
        """
        try:
            with get_sync_db() as db:
                # Proxy: look at our own upload history for recently completed mixes
                # Production: replace with YouTube Analytics API v2 call
                rows = db.execute(
                    text("""
                        SELECT
                            m.subgenre,
                            m.bpm,
                            COUNT(pu.id)             AS upload_count,
                            MAX(m.created_at)        AS last_used,
                            AVG(m.actual_duration_seconds) AS avg_duration_s
                        FROM mixes m
                        LEFT JOIN platform_uploads pu ON pu.mix_id = m.id
                            AND pu.platform = 'youtube'
                            AND pu.upload_status = 'uploaded'
                        WHERE m.status = 'complete'
                          AND m.created_at >= NOW() - INTERVAL :days
                        GROUP BY m.subgenre, m.bpm
                        ORDER BY upload_count DESC
                        LIMIT 20
                    """),
                    {"days": f"{days_back} days"},
                ).fetchall()

            analytics = [
                {
                    "subgenre": r.subgenre,
                    "bpm": r.bpm,
                    "upload_count": r.upload_count,
                    "last_used": r.last_used.isoformat() if r.last_used else None,
                    "avg_duration_minutes": round((r.avg_duration_s or 0) / 60, 1),
                }
                for r in rows
            ]
            log.info("YoutubeAnalyticsTool: %d subgenre/BPM records", len(analytics))
            return json.dumps({"metric": metric, "days_back": days_back, "data": analytics})

        except Exception as exc:
            log.error("YoutubeAnalyticsTool error: %s", exc)
            return json.dumps({"error": str(exc), "data": []})

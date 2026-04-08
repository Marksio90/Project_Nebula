"""
shared/tasks/celery_app.py
─────────────────────────────────────────────────────────────────────────────
Celery application factory — single source of truth for broker config,
serialisation, and queue routing.

Queue architecture (strict isolation):
  orchestration  → orchestrator service  (CrewAI coordination, light I/O)
  dsp            → dsp_worker service    (CPU-bound audio DSP)
  video          → video_worker service  (GPU/CPU-bound FFmpeg rendering)
  upload         → orchestrator service  (platform API calls)
─────────────────────────────────────────────────────────────────────────────
"""

from celery import Celery
from kombu import Exchange, Queue

from shared.config import get_settings
from shared.tasks import definitions as _task_defs

settings = get_settings()

# ── Exchanges ─────────────────────────────────────────────────────────────────
nebula_exchange = Exchange("nebula", type="direct", durable=True)

# ── Queue definitions ─────────────────────────────────────────────────────────
TASK_QUEUES = (
    Queue("orchestration", nebula_exchange, routing_key="orchestration", durable=True),
    Queue("dsp",           nebula_exchange, routing_key="dsp",           durable=True),
    Queue("video",         nebula_exchange, routing_key="video",         durable=True),
    Queue("upload",        nebula_exchange, routing_key="upload",        durable=True),
)

# ── Task → Queue routing ──────────────────────────────────────────────────────
# Task names are derived dynamically from the imported function objects so that
# IDE renames and refactors keep routing correct without silent failures.
def _route(task_fn, queue: str) -> tuple[str, dict]:
    return task_fn.name, {"queue": queue}


TASK_ROUTES = dict([
    # Orchestration tasks
    _route(_task_defs.orchestrate_mix_pipeline,  "orchestration"),
    _route(_task_defs.run_cso_strategy,          "orchestration"),
    _route(_task_defs.generate_audio_prompts,    "orchestration"),
    _route(_task_defs.generate_visual_prompts,   "orchestration"),
    _route(_task_defs.fetch_audio_stems,         "orchestration"),
    _route(_task_defs.fetch_visual_assets,       "orchestration"),
    _route(_task_defs.upload_to_youtube,         "upload"),
    _route(_task_defs.upload_to_tiktok,          "upload"),
    # DSP tasks
    _route(_task_defs.stitch_and_master_audio,   "dsp"),
    _route(_task_defs.run_qa_audio_check,        "dsp"),
    # Video tasks
    _route(_task_defs.render_full_video,         "video"),
    _route(_task_defs.slice_viral_shorts,        "video"),
    _route(_task_defs.run_qa_video_check,        "video"),
])

# ── Factory ───────────────────────────────────────────────────────────────────

def create_celery_app() -> Celery:
    app = Celery("nebula")

    app.conf.update(
        # Broker & backend
        broker_url=str(settings.redis_url),
        result_backend=str(settings.redis_url),

        # Serialisation
        task_serializer=settings.celery_task_serializer,
        result_serializer=settings.celery_result_serializer,
        accept_content=["json"],

        # Reliability
        task_acks_late=settings.celery_task_acks_late,
        task_reject_on_worker_lost=True,
        task_track_started=settings.celery_task_track_started,
        worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,

        # Timezone
        timezone="UTC",
        enable_utc=True,

        # Queues & routing
        task_queues=TASK_QUEUES,
        task_routes=TASK_ROUTES,
        task_default_queue="orchestration",
        task_default_exchange="nebula",
        task_default_routing_key="orchestration",

        # Suppress CPendingDeprecationWarning in Celery 5.x (default in 6.0)
        broker_connection_retry_on_startup=True,

        # Result expiry — 48 hours is enough for pipeline state
        result_expires=172_800,

        # Auto-discover tasks in all registered modules
        imports=["shared.tasks.definitions"],
    )

    return app


celery_app: Celery = create_celery_app()

"""
app/api/routes/health.py

Health check endpoints. These are:
  - Called by load balancers, Kubernetes liveness/readiness probes
  - Used by the frontend to verify the ML service is reachable before submitting jobs
  - Unauthenticated (no API key required) — they reveal no sensitive data

Endpoints:
  GET /health/         — Lightweight liveness check (is the process alive?)
  GET /health/ready    — Readiness check (can it accept work? tests Supabase conn.)
  GET /health/detail   — Full diagnostic (service info + dependency statuses)
"""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Track service start time for uptime calculation
_SERVICE_START_TIME = time.time()


@router.get(
    "/",
    summary="Liveness check",
    description=(
        "Returns 200 if the FastAPI process is running. "
        "Does NOT test external dependencies. "
        "Use /health/ready for a deeper check."
    ),
    response_description="Service is alive",
)
async def liveness() -> dict:
    """
    Minimal liveness probe.
    If this returns anything other than 200, the process is dead and should restart.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/ready",
    summary="Readiness check",
    description=(
        "Tests whether the service is ready to accept training jobs. "
        "Verifies Supabase connectivity by executing a lightweight DB query."
    ),
)
async def readiness() -> JSONResponse:
    """
    Readiness probe — checks that all critical dependencies are reachable.
    Returns 200 if ready, 503 if not (so load balancers stop routing to it).
    """
    checks: dict[str, dict] = {}
    all_healthy = True

    # ── Check 1: Supabase Database connectivity ───────────────────────────
    t0 = time.monotonic()
    try:
        supabase = get_supabase_client()
        # Lightweight query — just count projects (doesn't load data)
        result = (
            supabase
            .table("projects")
            .select("id", count="exact")
            .limit(1)
            .execute()
        )
        db_latency_ms = int((time.monotonic() - t0) * 1000)
        checks["supabase_db"] = {
            "status": "healthy",
            "latency_ms": db_latency_ms,
        }
    except Exception as e:
        db_latency_ms = int((time.monotonic() - t0) * 1000)
        logger.warning(f"Readiness check: Supabase DB unhealthy — {e}")
        checks["supabase_db"] = {
            "status": "unhealthy",
            "error": str(e),
            "latency_ms": db_latency_ms,
        }
        all_healthy = False

    # ── Check 2: Supabase Storage connectivity ───────────────────────────
    t0 = time.monotonic()
    try:
        supabase = get_supabase_client()
        supabase.storage.from_(settings.DATASETS_BUCKET).list(path="", options={"limit": 1})
        storage_latency_ms = int((time.monotonic() - t0) * 1000)
        checks["supabase_storage"] = {
            "status": "healthy",
            "latency_ms": storage_latency_ms,
        }
    except Exception as e:
        storage_latency_ms = int((time.monotonic() - t0) * 1000)
        logger.warning(f"Readiness check: Supabase Storage unhealthy — {e}")
        checks["supabase_storage"] = {
            "status": "unhealthy",
            "error": str(e),
            "latency_ms": storage_latency_ms,
        }
        all_healthy = False

    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        },
    )


@router.get(
    "/detail",
    summary="Detailed diagnostic",
    description=(
        "Full service diagnostic including version, uptime, environment, "
        "and dependency statuses. Intended for internal monitoring."
    ),
)
async def detail(request: Request) -> JSONResponse:
    """
    Comprehensive health diagnostic. Reveals environment metadata.
    Consider adding internal-network-only access control in production.
    """
    uptime_seconds = int(time.time() - _SERVICE_START_TIME)
    uptime_human = _format_uptime(uptime_seconds)

    # Run the same Supabase checks as readiness
    db_status = "unknown"
    storage_status = "unknown"

    try:
        supabase = get_supabase_client()
        supabase.table("projects").select("id", count="exact").limit(1).execute()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:80]}"

    try:
        supabase = get_supabase_client()
        supabase.storage.from_(settings.DATASETS_BUCKET).list(path="", options={"limit": 1})
        storage_status = "connected"
    except Exception as e:
        storage_status = f"error: {str(e)[:80]}"

    return JSONResponse(
        status_code=200,
        content={
            "service": "AutoML ML Microservice",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "running",
            "uptime": {
                "seconds": uptime_seconds,
                "human": uptime_human,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dependencies": {
                "supabase_db": db_status,
                "supabase_storage": storage_status,
            },
            "configuration": {
                "supabase_url": settings.SUPABASE_URL[:40] + "...",
                "datasets_bucket": settings.DATASETS_BUCKET,
                "models_bucket": settings.MODELS_BUCKET,
                "model_cache_size": settings.MODEL_CACHE_SIZE,
                "test_split_ratio": settings.TEST_SPLIT_RATIO,
                "training_timeout_sec": settings.TRAINING_TIMEOUT_SEC,
            },
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_uptime(seconds: int) -> str:
    """Converts uptime seconds to a human-readable string."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)

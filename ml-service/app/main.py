"""
AutoML SaaS Platform — FastAPI ML Microservice
Entry point: main.py

Architecture note:
  This service is a PURE ML WORKER. It:
    - Receives job payloads from the frontend (project_id, file_path, target_column)
    - Uses the Supabase SERVICE ROLE key to download datasets and upload models
    - Updates project status in the Supabase PostgreSQL database
    - NEVER handles user authentication or frontend sessions

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Production:
    gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.supabase_client import get_supabase_client, close_supabase_client
from app.api.routes import train, predict, health

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application Lifespan (replaces deprecated @app.on_event)
# Handles startup/shutdown logic: Supabase client init, resource cleanup.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  AutoML ML Microservice starting up...")
    logger.info(f"  Environment : {settings.ENVIRONMENT}")
    logger.info(f"  Supabase URL: {settings.SUPABASE_URL[:40]}...")
    logger.info("=" * 60)

    # Initialize and cache the Supabase client on app state
    # so all route handlers can access it without re-creating it
    app.state.supabase = get_supabase_client()
    logger.info("✓ Supabase client initialized")

    yield  # Application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("AutoML ML Microservice shutting down...")
    close_supabase_client()
    logger.info("✓ Supabase client closed")


# ---------------------------------------------------------------------------
# FastAPI Application Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AutoML SaaS — ML Microservice",
    description=(
        "Internal ML worker for the AutoML SaaS platform. "
        "Handles dataset ingestion, automated model training, "
        "serialization, and real-time predictions. "
        "All storage and database operations go through Supabase."
    ),
    version="1.0.0",
    docs_url="/docs",          # Swagger UI at /docs
    redoc_url="/redoc",        # ReDoc UI at /redoc
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Configured per environment
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global Exception Handler
# Catches any unhandled exception and returns a clean JSON error
# instead of leaking stack traces to clients.
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal server error occurred.",
            "detail": str(exc) if settings.ENVIRONMENT == "development" else None,
        },
    )


# ---------------------------------------------------------------------------
# Router Registration
# Each router is isolated in its own module for clean separation of concerns.
# ---------------------------------------------------------------------------

# /health — Service health check (no auth required)
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

# /train — Triggers the AutoML pipeline for a given project
app.include_router(
    train.router,
    prefix="/train",
    tags=["Training"],
)

# /predict — Dynamic prediction endpoint for deployed models
app.include_router(
    predict.router,
    prefix="/predict",
    tags=["Prediction"],
)


# ---------------------------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "service": "AutoML ML Microservice",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }

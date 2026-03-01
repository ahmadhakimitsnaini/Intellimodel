"""
app/api/routes/train.py

POST /train/ — Triggers the AutoML training pipeline.

Flow:
  1. Validate the request payload (Pydantic)
  2. Immediately return HTTP 202 with status="training"
  3. The actual pipeline runs as a FastAPI BackgroundTask (non-blocking)
     in a thread pool — the HTTP response is sent BEFORE training starts.
  4. The frontend polls the Supabase `projects` table to watch status transitions:
       pending → training → completed | failed

Why BackgroundTasks (not Celery/RQ)?
  - For a single-server deployment, FastAPI's built-in BackgroundTasks are
    sufficient and introduce zero infrastructure complexity.
  - Each training job runs in a threadpool worker, so the event loop stays free.
  - For multi-server horizontal scaling, swap background_tasks.add_task() for
    a Celery task with Redis broker — the AutoMLPipeline.run_pipeline() signature
    is identical either way.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, status
from supabase import Client

from app.core.supabase_client import get_supabase_client_dependency
from app.models.schemas import ProjectStatus, TrainRequest, TrainResponse
from app.services.automl import AutoMLPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger AutoML training pipeline",
    description=(
        "Validates the payload, marks the project as 'training' in the DB, "
        "and immediately returns HTTP 202. The full AutoML pipeline "
        "(download -> preprocess -> train -> evaluate -> upload -> update DB) "
        "runs asynchronously in the background. "
        "Poll the Supabase `projects` table for live status updates."
    ),
    responses={
        202: {"description": "Training job accepted and queued"},
        400: {"description": "Invalid payload (bad file_path, missing column, etc.)"},
        422: {"description": "Request body validation error"},
        500: {"description": "Internal server error"},
    },
)
async def trigger_training(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
    supabase: Client = Depends(get_supabase_client_dependency),
) -> TrainResponse:
    """
    Accepts a training job and queues it as a background task.

    The response is sent immediately — do NOT wait for this endpoint to return
    a 'completed' status. Monitor the `projects` table in Supabase instead.
    """
    logger.info(
        f"[TRAIN] Request received | "
        f"project={payload.project_id} | "
        f"file={payload.file_path} | "
        f"target='{payload.target_column}'"
    )

    # Instantiate a fresh pipeline orchestrator for this job.
    # Each job gets its own instance so there is no shared state between runs.
    pipeline = AutoMLPipeline(supabase=supabase)

    # Register the pipeline as a background task.
    # FastAPI will execute this after sending the HTTP 202 response.
    background_tasks.add_task(
        pipeline.run_pipeline,
        project_id=payload.project_id,
        file_path=payload.file_path,
        target_column=payload.target_column,
    )

    logger.info(
        f"[TRAIN] Background task queued | project={payload.project_id}"
    )

    return TrainResponse(
        success=True,
        project_id=payload.project_id,
        status=ProjectStatus.TRAINING,
        message=(
            f"Training job accepted. The AutoML pipeline is running in the background. "
            f"Target column: '{payload.target_column}'. "
            f"Poll your project status in Supabase for updates."
        ),
    )

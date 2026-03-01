"""
app/api/routes/predict.py

POST /predict/{project_id} — Serve predictions from a trained model.

Flow:
  1. Load project metadata from Supabase DB (task_type, model_path, feature_columns)
  2. Check the in-memory LRU cache for the loaded Pipeline
  3. On cache miss: download the .joblib from Supabase Storage, cache it
  4. Validate input features against the expected feature set
  5. Run model.predict() (and predict_proba() for classification)
  6. Write an audit row to prediction_logs (non-blocking, failure-tolerant)
  7. Return the prediction with latency measurement

GET /predict/{project_id}/schema — Returns expected input schema for a model.
  Useful for the frontend to auto-build the prediction form.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, status
from supabase import Client

from app.core.supabase_client import get_supabase_client_dependency
from app.models.schemas import (
    PredictRequest,
    PredictResponse,
    ProjectStatus,
    TaskType,
)
from app.services.storage import StorageService
from app.utils.model_cache import model_cache

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/{project_id}",
    response_model=PredictResponse,
    summary="Get a prediction from a trained model",
    description=(
        "Loads the trained model for `project_id` from cache or Supabase Storage "
        "and returns a prediction for the provided input features. "
        "For classification: returns the predicted class label and confidence scores. "
        "For regression: returns the predicted numeric value."
    ),
    responses={
        200: {"description": "Prediction returned successfully"},
        404: {"description": "Project not found or model not yet trained"},
        409: {"description": "Model training is still in progress"},
        422: {"description": "Input features do not match trained model schema"},
        500: {"description": "Inference error"},
    },
)
async def predict(
    project_id: str = Path(
        ...,
        description="UUID of the project whose model should serve the prediction",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    ),
    payload: PredictRequest = ...,
    supabase: Client = Depends(get_supabase_client_dependency),
) -> PredictResponse:
    """
    Full prediction flow with model caching and prediction logging.
    """
    request_start = time.monotonic()
    storage = StorageService(supabase)

    # ── Step 1: Load project metadata ────────────────────────────────────────
    try:
        project = storage.get_project(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    # ── Step 2: Guard on project status ──────────────────────────────────────
    if project.status == ProjectStatus.TRAINING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model is still training. Please try again in a few moments.",
        )
    if project.status == ProjectStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Training failed for this project: {project.error_message}. "
                "Please re-trigger training with a corrected dataset."
            ),
        )
    if project.status != ProjectStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Model not available: project status is '{project.status.value}'. "
                "Training must complete before predictions can be made."
            ),
        )

    # Validate fields that must exist on a completed project
    if not project.model_path or not project.feature_columns or not project.task_type:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Project metadata is incomplete. Please contact support.",
        )

    # ── Step 3: Load pipeline (cache → Storage) ───────────────────────────────
    pipeline = model_cache.get(project_id)
    if pipeline is None:
        logger.info(f"[PREDICT] Cache miss — loading from Storage | project={project_id}")
        try:
            pipeline = storage.download_model(project.model_path)
            model_cache.put(project_id, pipeline)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {exc}",
            )

    # ── Step 4: Validate input features ──────────────────────────────────────
    expected_features = project.feature_columns
    provided_features = set(payload.features.keys())
    missing_features  = set(expected_features) - provided_features
    extra_features    = provided_features - set(expected_features)

    if missing_features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Missing required feature columns",
                "missing": sorted(missing_features),
                "expected": expected_features,
            },
        )
    # Extra features are silently dropped (model pipeline will ignore them)

    # ── Step 5: Build input DataFrame ────────────────────────────────────────
    # Construct a single-row DataFrame in the exact column order the pipeline expects.
    try:
        input_df = pd.DataFrame([payload.features])[expected_features]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to build input DataFrame: {exc}",
        )

    # ── Step 6: Run inference ─────────────────────────────────────────────────
    try:
        raw_prediction = pipeline.predict(input_df)
    except Exception as exc:
        logger.error(f"[PREDICT] Inference error | project={project_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {exc}",
        )

    task_type = project.task_type  # TaskType enum

    # ── Step 7: Build prediction output ──────────────────────────────────────
    prediction_value: object
    prediction_label: str | None       = None
    probability:      float | None     = None
    all_probabilities: dict | None     = None

    if task_type == TaskType.REGRESSION:
        prediction_value = float(raw_prediction[0])

    else:  # CLASSIFICATION
        predicted_class = raw_prediction[0]

        # Try to get class probabilities
        if hasattr(pipeline, "predict_proba"):
            try:
                proba_array  = pipeline.predict_proba(input_df)[0]  # shape: (n_classes,)
                class_labels = pipeline.classes_

                # Map class index to label string
                all_probabilities = {
                    str(cls): round(float(prob), 6)
                    for cls, prob in zip(class_labels, proba_array)
                }
                probability = float(max(proba_array))

            except Exception as exc:
                logger.warning(f"[PREDICT] predict_proba failed: {exc}")
                all_probabilities = None
                probability       = None

        prediction_value = (
            int(predicted_class)
            if isinstance(predicted_class, (np.integer,))
            else str(predicted_class)
        )
        prediction_label = str(predicted_class)

    # ── Step 8: Measure latency ───────────────────────────────────────────────
    latency_ms = int((time.monotonic() - request_start) * 1000)

    # ── Step 9: Log prediction (fire-and-forget) ──────────────────────────────
    prediction_output = {
        "prediction":       prediction_value,
        "probability":      probability,
        "all_probabilities": all_probabilities,
    }
    storage.log_prediction(
        project_id=project_id,
        user_id=project.user_id,
        input_data=payload.features,
        prediction_output=prediction_output,
        model_version=project.winning_model or "unknown",
        latency_ms=latency_ms,
    )

    logger.info(
        f"[PREDICT] OK | project={project_id} | "
        f"pred={prediction_value} | latency={latency_ms}ms"
    )

    return PredictResponse(
        success=True,
        project_id=project_id,
        task_type=task_type,
        prediction=prediction_value,
        prediction_label=prediction_label,
        probability=probability,
        all_probabilities=all_probabilities,
        model_name=project.winning_model or "unknown",
        latency_ms=latency_ms,
    )


@router.get(
    "/{project_id}/schema",
    summary="Get expected input schema for a model",
    description=(
        "Returns the list of feature columns and their types that the model expects. "
        "Use this to dynamically build the prediction form in the frontend."
    ),
)
async def get_prediction_schema(
    project_id: str = Path(..., description="Project UUID"),
    supabase: Client = Depends(get_supabase_client_dependency),
) -> dict:
    """
    Returns metadata about what the model expects as input.
    Enables the frontend to auto-generate a prediction form with the right fields.
    """
    storage = StorageService(supabase)

    try:
        project = storage.get_project(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    if project.status != ProjectStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model not available yet (status={project.status.value}).",
        )

    return {
        "project_id":      project_id,
        "target_column":   project.target_column,
        "task_type":       project.task_type.value if project.task_type else None,
        "winning_model":   project.winning_model,
        "accuracy_score":  project.accuracy_score,
        "metric_name":     project.metric_name,
        "feature_columns": project.feature_columns,
        "n_features":      len(project.feature_columns) if project.feature_columns else 0,
    }

"""
app/services/storage.py

All Supabase interactions for the ML pipeline:
  - Downloading CSV datasets from Supabase Storage
  - Uploading serialized .joblib model files to Supabase Storage
  - Reading project rows from the PostgreSQL database
  - Updating project rows (status, scores, model_path, etc.)

Design:
  - Every public method is independently testable (takes/returns plain Python types).
  - All Supabase calls are wrapped in try/except with structured logging.
  - Uses the service-role client (bypasses RLS) — this is correct for a server worker.
  - Temporary local files are written to settings.TEMP_DIR and always cleaned up.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from supabase import Client

from app.core.config import settings
from app.models.schemas import ProjectRow, ProjectStatus, TaskType, TrainingResult

logger = logging.getLogger(__name__)


class StorageService:
    """
    Encapsulates all Supabase Storage and DB calls.
    Instantiated once per pipeline run; receives the shared client via DI.
    """

    def __init__(self, supabase: Client):
        self._sb = supabase

    # ─────────────────────────────────────────────────────────────────────────
    # CSV Download
    # ─────────────────────────────────────────────────────────────────────────

    def download_csv(self, file_path: str) -> pd.DataFrame:
        """
        Downloads a CSV from the 'datasets' Supabase Storage bucket and
        returns it as a pandas DataFrame.

        Args:
            file_path: Path inside the bucket, e.g. "{user_id}/{project_id}/data.csv"

        Returns:
            pandas DataFrame of the CSV contents.

        Raises:
            RuntimeError: If the download or CSV parse fails.
        """
        logger.info(f"Downloading dataset: {settings.DATASETS_BUCKET}/{file_path}")
        t0 = time.monotonic()

        try:
            raw_bytes: bytes = (
                self._sb.storage
                .from_(settings.DATASETS_BUCKET)
                .download(file_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download dataset from Storage "
                f"(bucket={settings.DATASETS_BUCKET}, path={file_path}): {exc}"
            ) from exc

        elapsed = time.monotonic() - t0
        logger.info(
            f"Downloaded {len(raw_bytes) / 1024:.1f} KB in {elapsed:.2f}s"
        )

        # Parse CSV from in-memory bytes — no temp file needed here
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse CSV (path={file_path}): {exc}"
            ) from exc

        logger.info(f"Parsed DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Model Upload
    # ─────────────────────────────────────────────────────────────────────────

    def upload_model(
        self,
        pipeline: Pipeline,
        user_id: str,
        project_id: str,
    ) -> str:
        """
        Serializes the sklearn Pipeline to a .joblib file and uploads it
        to the 'models' Supabase Storage bucket.

        Args:
            pipeline:   The fitted sklearn Pipeline to serialize.
            user_id:    Supabase user UUID (used to namespace the storage path).
            project_id: Project UUID (used to namespace the storage path).

        Returns:
            The storage path (str) used to store the model, e.g.
            "{user_id}/{project_id}/model.joblib"

        Raises:
            RuntimeError: If serialization or upload fails.
        """
        storage_path = f"{user_id}/{project_id}/model.joblib"
        logger.info(f"Uploading model to: {settings.MODELS_BUCKET}/{storage_path}")

        # Ensure temp directory exists
        temp_dir = Path(settings.TEMP_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Write to a named temp file, then read back as bytes for upload
        # (supabase-py storage upload expects bytes)
        tmp_path = temp_dir / f"model_{project_id}.joblib"
        try:
            joblib.dump(pipeline, tmp_path, compress=3)
            model_bytes = tmp_path.read_bytes()
            file_size_kb = len(model_bytes) / 1024
            logger.info(f"Serialized model: {file_size_kb:.1f} KB")
        except Exception as exc:
            raise RuntimeError(f"Failed to serialize model: {exc}") from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink()  # Always clean up

        t0 = time.monotonic()
        try:
            self._sb.storage.from_(settings.MODELS_BUCKET).upload(
                path=storage_path,
                file=model_bytes,
                file_options={"content-type": "application/octet-stream", "upsert": "true"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upload model to Storage "
                f"(bucket={settings.MODELS_BUCKET}, path={storage_path}): {exc}"
            ) from exc

        elapsed = time.monotonic() - t0
        logger.info(f"Model uploaded in {elapsed:.2f}s → {storage_path}")
        return storage_path

    def download_model(self, model_path: str) -> Pipeline:
        """
        Downloads and deserializes a .joblib model from Supabase Storage.
        Called at prediction time (with LRU caching in the predict service).

        Args:
            model_path: Path inside the 'models' bucket.

        Returns:
            Deserialized sklearn Pipeline ready for inference.

        Raises:
            RuntimeError: If download or deserialization fails.
        """
        logger.info(f"Loading model from: {settings.MODELS_BUCKET}/{model_path}")
        t0 = time.monotonic()

        try:
            raw_bytes: bytes = (
                self._sb.storage
                .from_(settings.MODELS_BUCKET)
                .download(model_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download model from Storage "
                f"(bucket={settings.MODELS_BUCKET}, path={model_path}): {exc}"
            ) from exc

        try:
            pipeline: Pipeline = joblib.load(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to deserialize model (path={model_path}): {exc}"
            ) from exc

        elapsed = time.monotonic() - t0
        logger.info(f"Model loaded in {elapsed:.2f}s ({len(raw_bytes)/1024:.1f} KB)")
        return pipeline

    # ─────────────────────────────────────────────────────────────────────────
    # Database — Project Row Operations
    # ─────────────────────────────────────────────────────────────────────────

    def get_project(self, project_id: str) -> ProjectRow:
        """
        Fetches a single project row by ID.

        Raises:
            ValueError:   If the project is not found.
            RuntimeError: If the DB query fails.
        """
        try:
            response = (
                self._sb.table("projects")
                .select("*")
                .eq("id", project_id)
                .single()
                .execute()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch project {project_id}: {exc}"
            ) from exc

        if not response.data:
            raise ValueError(f"Project not found: {project_id}")

        return ProjectRow(**response.data)

    def update_status(
        self,
        project_id: str,
        status: ProjectStatus,
        error_message: str | None = None,
    ) -> None:
        """
        Updates only the `status` (and optionally `error_message`) of a project row.
        Called at the START of training (→ "training") and on failure (→ "failed").
        """
        payload: dict = {"status": status.value}
        if error_message is not None:
            payload["error_message"] = error_message[:2000]  # truncate for DB

        try:
            self._sb.table("projects").update(payload).eq("id", project_id).execute()
            logger.info(f"Project {project_id} status → {status.value}")
        except Exception as exc:
            logger.error(f"Failed to update project status: {exc}")
            # Don't re-raise: status update failure should not crash the pipeline

    def update_completed(
        self,
        project_id: str,
        result: TrainingResult,
    ) -> None:
        """
        Writes the full training result to the project row on successful completion.
        Sets status → "completed" and populates all ML result fields.
        """
        payload = {
            "status":          ProjectStatus.COMPLETED.value,
            "task_type":       result.task_type.value,
            "winning_model":   result.winning_model,
            "accuracy_score":  result.primary_score,
            "metric_name":     result.metric_name,
            "all_scores":      result.all_scores,           # JSONB column
            "model_path":      result.model_path,
            "feature_columns": result.feature_columns,     # TEXT[] column
            "target_column":   result.dataset_profile.target_column,
            "trained_at":      datetime.now(timezone.utc).isoformat(),
            "error_message":   None,                        # Clear any prior error
        }

        try:
            self._sb.table("projects").update(payload).eq("id", project_id).execute()
            logger.info(
                f"Project {project_id} completed: "
                f"{result.winning_model} "
                f"({result.metric_name}={result.primary_score:.4f})"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write training results for project {project_id}: {exc}"
            ) from exc

    def log_prediction(
        self,
        project_id: str,
        user_id: str,
        input_data: dict,
        prediction_output: dict,
        model_version: str,
        latency_ms: int,
    ) -> None:
        """
        Appends a row to prediction_logs for auditing and usage monitoring.
        Failures are logged but never re-raised (non-critical path).
        """
        try:
            self._sb.table("prediction_logs").insert({
                "project_id":   project_id,
                "user_id":      user_id,
                "input_data":   input_data,
                "prediction":   prediction_output,
                "model_version": model_version,
                "latency_ms":   latency_ms,
            }).execute()
        except Exception as exc:
            logger.warning(f"Failed to write prediction log: {exc}")

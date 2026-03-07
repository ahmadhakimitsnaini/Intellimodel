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

# application/prediction/service.py

import logging
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.core.supabase_client import get_supabase_client_dependency
from app.domain.models import (
    PredictRequest,
    PredictResponse,
    ProjectStatus,
    TaskType,
)
from app.infrastructure.supabase_storage import StorageService
from app.infrastructure.model_cache import model_cache

logger = logging.getLogger(__name__)
router = APIRouter()


class PredictionService:
    def __init__(self, storage_service: StorageService):
        """
        Dependency Injection: Menerima instance storage_service 
        agar service ini tidak terikat erat (tightly coupled) dengan inisialisasi database.
        """
        self.storage = storage_service

    def execute_prediction(self, project_id: str, payload: PredictRequest) -> PredictResponse:
        """
        Mengeksekusi alur prediksi secara lengkap:
        Validasi -> Load Model -> Inferensi -> Log -> Return
        """
        request_start = time.monotonic()

        # 1. Baca metadata project dan validasi status
        project = self._get_and_validate_project(project_id)

        # 2. Ambil model dari cache atau download via storage
        pipeline = self._get_model_pipeline(project_id, project.model_path)

        # 3. Validasi skema input
        self._validate_features(payload.features, project.feature_columns)

        # 4. Bangun DataFrame input
        try:
            input_df = pd.DataFrame([payload.features])[project.feature_columns]
        except Exception as exc:
            raise ValueError(f"Failed to build input DataFrame: {exc}")

        # 5. Jalankan inferensi (Sklearn Pipeline)
        try:
            raw_prediction = pipeline.predict(input_df)
        except Exception as exc:
            logger.error(f"[PREDICT] Inference error | project={project_id}: {exc}")
            raise RuntimeError(f"Model inference failed: {exc}")

        # 6. Ekstrak hasil prediksi
        prediction_data = self._extract_prediction_results(
            pipeline=pipeline,
            input_df=input_df,
            raw_prediction=raw_prediction,
            task_type=project.task_type
        )

        # Hitung latensi murni proses AI
        latency_ms = int((time.monotonic() - request_start) * 1000)

        # 7. Log prediksi ke database
        self._log_prediction_result(
            project_id=project_id,
            user_id=project.user_id,
            features=payload.features,
            prediction_data=prediction_data,
            winning_model=project.winning_model,
            latency_ms=latency_ms
        )

        logger.info(
            f"[PREDICT] OK | project={project_id} | "
            f"pred={prediction_data['prediction_value']} | latency={latency_ms}ms"
        )

        # 8. Kembalikan Response
        return PredictResponse(
            success=True,
            project_id=project_id,
            task_type=project.task_type,
            prediction=prediction_data["prediction_value"],
            prediction_label=prediction_data.get("prediction_label"),
            probability=prediction_data.get("probability"),
            all_probabilities=prediction_data.get("all_probabilities"),
            model_name=project.winning_model or "unknown",
            latency_ms=latency_ms,
        )

    # --- Helper Methods (Private) ---

    def _get_and_validate_project(self, project_id: str):
        project = self.storage.get_project(project_id)
        
        if project.status == ProjectStatus.TRAINING:
            raise ValueError("Model is still training. Please try again in a few moments.")
        if project.status == ProjectStatus.FAILED:
            raise ValueError(f"Training failed: {project.error_message}.")
        if project.status != ProjectStatus.COMPLETED:
            raise ValueError(f"Model not available (status: {project.status.value}).")
            
        if not project.model_path or not project.feature_columns or not project.task_type:
            raise RuntimeError("Project metadata is incomplete.")
            
        return project

    def _get_model_pipeline(self, project_id: str, model_path: str):
        pipeline = model_cache.get(project_id)
        if pipeline is None:
            logger.info(f"[PREDICT] Cache miss — loading from Storage | project={project_id}")
            pipeline = self.storage.download_model(model_path)
            model_cache.put(project_id, pipeline)
        return pipeline

    def _validate_features(self, provided_features: dict, expected_features: list):
        provided_set = set(provided_features.keys())
        missing_features = set(expected_features) - provided_set
        
        if missing_features:
            raise ValueError(
                f"Missing required feature columns: {sorted(missing_features)}. "
                f"Expected: {expected_features}"
            )

    def _extract_prediction_results(self, pipeline, input_df: pd.DataFrame, raw_prediction: np.ndarray, task_type: TaskType) -> Dict[str, Any]:
        result = {
            "prediction_value": None,
            "prediction_label": None,
            "probability": None,
            "all_probabilities": None
        }

        if task_type == TaskType.REGRESSION:
            result["prediction_value"] = float(raw_prediction[0])
        else:  # CLASSIFICATION
            predicted_class = raw_prediction[0]

            if hasattr(pipeline, "predict_proba"):
                try:
                    proba_array = pipeline.predict_proba(input_df)[0]
                    class_labels = pipeline.classes_
                    result["all_probabilities"] = {
                        str(cls): round(float(prob), 6)
                        for cls, prob in zip(class_labels, proba_array)
                    }
                    result["probability"] = float(max(proba_array))
                except Exception as exc:
                    logger.warning(f"[PREDICT] predict_proba failed: {exc}")

            result["prediction_value"] = (
                int(predicted_class) if isinstance(predicted_class, np.integer) else str(predicted_class)
            )
            result["prediction_label"] = str(predicted_class)
            
        return result

    def _log_prediction_result(self, project_id: str, user_id: str, features: dict, prediction_data: dict, winning_model: str, latency_ms: int):
        prediction_output = {
            "prediction": prediction_data["prediction_value"],
            "probability": prediction_data["probability"],
            "all_probabilities": prediction_data["all_probabilities"],
        }
        self.storage.log_prediction(
            project_id=project_id,
            user_id=user_id,
            input_data=features,
            prediction_output=prediction_output,
            model_version=winning_model or "unknown",
            latency_ms=latency_ms,
        )


@router.post(
    "/{project_id}",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Run prediction for a trained project model",
)
def predict(
    project_id: str,
    payload: PredictRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
) -> PredictResponse:
    """
    FastAPI route that delegates to PredictionService.
    """
    storage = StorageService(supabase)
    service = PredictionService(storage_service=storage)

    try:
        return service.execute_prediction(project_id, payload)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

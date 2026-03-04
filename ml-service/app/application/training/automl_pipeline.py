"""
app/services/automl.py

AutoML Pipeline Orchestrator.

This is the single entry point for the entire training workflow:

  download CSV  →  preprocess  →  train 3 models  →  select best  →
  upload model  →  update DB   →  return TrainingResult

It composes DataPreprocessor + ModelTrainer + StorageService and handles
the full lifecycle of a project status update, including error recovery.

The public method run_pipeline() is called from the /train route as a
FastAPI BackgroundTask — it runs in a thread pool, not in the event loop.
"""

from __future__ import annotations

import logging
import time

from supabase import Client

from app.core.config import settings
from app.models.schemas import ProjectRow, ProjectStatus, TrainingResult
from app.services.preprocessor import DataPreprocessor
from app.services.storage import StorageService
from app.services.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """
    Orchestrates the full AutoML training workflow for a single project.

    Design:
      - Constructed once per background job (not a singleton).
      - Each public method maps to one stage of the pipeline.
      - Status updates happen at the START (→ training) and END (→ completed/failed)
        of the run so the frontend always sees a current state.
      - Any exception at any stage marks the project as "failed" with the
        error message stored in the DB for user visibility.
    """

    def __init__(self, supabase: Client):
        self._storage    = StorageService(supabase)
        self._preprocessor = DataPreprocessor(
            test_size=settings.TEST_SPLIT_RATIO,
            random_state=settings.RANDOM_STATE,
        )
        self._trainer = ModelTrainer(random_state=settings.RANDOM_STATE)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point — called as FastAPI BackgroundTask
    # ─────────────────────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        project_id: str,
        file_path: str,
        target_column: str,
    ) -> None:
        """
        Executes the full AutoML pipeline end-to-end.

        This method is intentionally synchronous and designed to run in a
        FastAPI BackgroundTasks thread. It does NOT raise exceptions — all
        errors are caught, logged, and written to the DB as status="failed".

        Args:
            project_id:    UUID of the projects row to update.
            file_path:     Path inside the 'datasets' bucket.
            target_column: Target column name in the CSV.
        """
        pipeline_start = time.monotonic()
        logger.info(
            f"[Pipeline] START | project={project_id} "
            f"| file={file_path} | target={target_column}"
        )

        # ── Stage 0: Fetch project and get user_id for storage namespacing ──
        try:
            project: ProjectRow = self._storage.get_project(project_id)
            user_id = project.user_id
        except Exception as exc:
            logger.error(f"[Pipeline] Cannot load project {project_id}: {exc}")
            return  # Can't update status without knowing user_id — silent fail

        # ── Stage 1: Mark project as "training" ─────────────────────────────
        self._storage.update_status(project_id, ProjectStatus.TRAINING)

        try:
            result = self._execute_stages(
                project_id=project_id,
                user_id=user_id,
                file_path=file_path,
                target_column=target_column,
            )

            # ── Stage Final: Write completed results to DB ───────────────────
            self._storage.update_completed(project_id, result)

            total_time = time.monotonic() - pipeline_start
            logger.info(
                f"[Pipeline] COMPLETE | project={project_id} "
                f"| winner={result.winning_model} "
                f"| {result.metric_name}={result.primary_score:.4f} "
                f"| total_time={total_time:.1f}s"
            )

        except Exception as exc:
            # ── Error Recovery: mark project as "failed" ────────────────────
            error_msg = str(exc)
            logger.error(
                f"[Pipeline] FAILED | project={project_id} | error={error_msg}",
                exc_info=True,
            )
            self._storage.update_status(
                project_id,
                ProjectStatus.FAILED,
                error_message=error_msg,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Private: main execution stages (raises on any failure)
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_stages(
        self,
        project_id: str,
        user_id: str,
        file_path: str,
        target_column: str,
    ) -> TrainingResult:
        """
        Runs stages 2–6. Any exception propagates up to run_pipeline()
        which handles error recovery.
        """

        # ── Stage 2: Download CSV from Supabase Storage ──────────────────────
        logger.info(f"[Stage 2/6] Downloading dataset...")
        df = self._storage.download_csv(file_path)

        # ── Stage 3: Preprocess ──────────────────────────────────────────────
        logger.info(f"[Stage 3/6] Preprocessing ({df.shape})...")
        preprocessed = self._preprocessor.build(df, target_column)
        profile = preprocessed.profile
        logger.info(
            f"           task={profile.task_type.value} "
            f"| features={profile.n_features} "
            f"| train_rows={preprocessed.X_train.shape[0]}"
        )

        # ── Stage 4: Train all candidates and select winner ──────────────────
        logger.info(f"[Stage 4/6] Training candidate models...")
        model_results, best_pipeline = self._trainer.train(preprocessed)

        best_result   = model_results[0]   # Already sorted best → worst
        all_scores    = {r.model_name: r.primary_score for r in model_results}
        all_metrics   = {r.model_name: r.all_metrics   for r in model_results}

        # ── Stage 5: Upload .joblib to Supabase Storage ──────────────────────
        logger.info(f"[Stage 5/6] Uploading model...")
        model_path = self._storage.upload_model(
            pipeline=best_pipeline,
            user_id=user_id,
            project_id=project_id,
        )

        # ── Stage 6: Assemble TrainingResult ─────────────────────────────────
        logger.info(f"[Stage 6/6] Assembling result...")
        total_training_time = sum(r.training_time_sec for r in model_results)

        return TrainingResult(
            project_id=project_id,
            task_type=profile.task_type,
            winning_model=best_result.model_name,
            primary_score=best_result.primary_score,
            metric_name=best_result.metric_name,
            all_scores=all_scores,
            all_metrics=all_metrics,
            feature_columns=profile.feature_columns,
            model_path=model_path,
            dataset_profile=profile,
            total_training_time_sec=total_training_time,
        )

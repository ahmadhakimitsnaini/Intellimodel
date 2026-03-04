"""
app/services/trainer.py

Multi-model AutoML trainer.

Responsibilities:
  1. Define the candidate model roster (3 per task type)
  2. Train each candidate against the preprocessed data
  3. Evaluate each candidate with task-appropriate metrics
  4. Select the single best model by primary metric
  5. Build a final sklearn Pipeline (feature_pipeline + best_model) ready for
     serialization and inference

Candidate models:
  Classification:  RandomForestClassifier, LogisticRegression, GradientBoostingClassifier
  Regression:      RandomForestRegressor,  LinearRegression,   GradientBoostingRegressor

Primary selection metric:
  Classification → weighted F1-score  (robust to class imbalance)
  Regression     → R² score           (scale-independent)

All secondary metrics are also computed and stored in the result for the
dashboard to display (accuracy, precision, recall, MAE, RMSE, etc.).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from app.models.schemas import DatasetProfile, ModelResult, TaskType
from app.services.preprocessor import PreprocessedData

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains the candidate model roster and returns a ranked list of ModelResults
    plus the full sklearn Pipeline for the best model.

    Usage:
        trainer = ModelTrainer(random_state=42)
        results, best_pipeline = trainer.train(preprocessed_data)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        data: PreprocessedData,
    ) -> tuple[list[ModelResult], Pipeline]:
        """
        Train all candidates and return results sorted by primary score (desc).

        Args:
            data: PreprocessedData from the preprocessor step.

        Returns:
            Tuple of:
              - results:      All ModelResults, sorted best → worst
              - best_pipeline: Fitted sklearn Pipeline (preprocessor + best model)
        """
        task_type = data.profile.task_type
        candidates = self._get_candidates(task_type)

        logger.info(
            f"Starting training: {len(candidates)} candidates | "
            f"task={task_type.value} | "
            f"train_rows={data.X_train.shape[0]}"
        )

        results: list[ModelResult] = []
        trained_estimators: dict[str, Any] = {}

        for model_name, estimator in candidates:
            result, fitted_estimator = self._train_single(
                model_name=model_name,
                estimator=estimator,
                data=data,
            )
            results.append(result)
            trained_estimators[model_name] = fitted_estimator
            logger.info(
                f"  [{model_name}] "
                f"{result.metric_name}={result.primary_score:.4f} "
                f"({result.training_time_sec:.1f}s)"
            )

        # Sort by primary metric descending
        results.sort(key=lambda r: r.primary_score, reverse=True)
        best_result = results[0]

        logger.info(
            f"Winner: {best_result.model_name} "
            f"({best_result.metric_name}={best_result.primary_score:.4f})"
        )

        # Build the final serializable pipeline
        best_pipeline = self._build_final_pipeline(
            feature_pipeline=data.feature_pipeline,
            estimator=trained_estimators[best_result.model_name],
        )

        return results, best_pipeline

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate definitions
    # ─────────────────────────────────────────────────────────────────────────

    def _get_candidates(
        self, task_type: TaskType
    ) -> list[tuple[str, Any]]:
        """
        Returns the list of (name, unfitted_estimator) pairs for the given task.

        Hyperparameter choices:
          - Conservative defaults that work well out-of-the-box on most tabular data.
          - n_jobs=-1 to use all CPU cores for tree ensembles.
          - max_iter=1000 for LogisticRegression to avoid ConvergenceWarnings.
        """
        if task_type == TaskType.CLASSIFICATION:
            return [
                (
                    "RandomForestClassifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_split=5,
                        n_jobs=-1,
                        random_state=self.random_state,
                        class_weight="balanced",  # handles imbalanced classes
                    ),
                ),
                (
                    "LogisticRegression",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        C=1.0,
                        class_weight="balanced",
                        random_state=self.random_state,
                        # n_jobs removed: no effect in sklearn ≥ 1.8
                    ),
                ),
                (
                    "GradientBoostingClassifier",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=4,
                        subsample=0.8,
                        random_state=self.random_state,
                    ),
                ),
            ]

        # TaskType.REGRESSION
        return [
            (
                "RandomForestRegressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=self.random_state,
                ),
            ),
            (
                "LinearRegression",
                LinearRegression(n_jobs=-1),
            ),
            (
                "GradientBoostingRegressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.8,
                    random_state=self.random_state,
                ),
            ),
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Single-model training and evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _train_single(
        self,
        model_name: str,
        estimator: Any,
        data: PreprocessedData,
    ) -> tuple[ModelResult, Any]:
        """
        Fits one estimator on X_train/y_train, evaluates on X_test/y_test.
        Returns (ModelResult, fitted_estimator).
        """
        t0 = time.monotonic()

        try:
            estimator.fit(data.X_train, data.y_train)
        except Exception as exc:
            logger.warning(f"[{model_name}] training failed: {exc}")
            # Return a zero-score result so this model is never selected
            return (
                ModelResult(
                    model_name=model_name,
                    primary_score=0.0,
                    metric_name=self._primary_metric_name(data.profile.task_type),
                    all_metrics={},
                    training_time_sec=time.monotonic() - t0,
                ),
                estimator,
            )

        elapsed = time.monotonic() - t0
        y_pred = estimator.predict(data.X_test)

        metrics = self._compute_metrics(
            task_type=data.profile.task_type,
            y_true=data.y_test,
            y_pred=y_pred,
            estimator=estimator,
            X_test=data.X_test,
        )

        primary_metric_name = self._primary_metric_name(data.profile.task_type)
        primary_score       = metrics[primary_metric_name]

        return (
            ModelResult(
                model_name=model_name,
                primary_score=primary_score,
                metric_name=primary_metric_name,
                all_metrics=metrics,
                training_time_sec=elapsed,
            ),
            estimator,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def _primary_metric_name(self, task_type: TaskType) -> str:
        return "f1_weighted" if task_type == TaskType.CLASSIFICATION else "r2"

    def _compute_metrics(
        self,
        task_type: TaskType,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        estimator: Any,
        X_test: np.ndarray,
    ) -> dict[str, float]:
        """
        Computes a full suite of metrics for the given task type.
        All values are rounded to 6 decimal places for storage.
        """
        metrics: dict[str, float] = {}

        if task_type == TaskType.CLASSIFICATION:
            # zero_division=0 suppresses warnings for classes not in test set
            metrics["accuracy"]       = round(float(accuracy_score(y_true, y_pred)), 6)
            metrics["f1_weighted"]    = round(float(f1_score(y_true, y_pred, average="weighted",   zero_division=0)), 6)
            metrics["f1_macro"]       = round(float(f1_score(y_true, y_pred, average="macro",      zero_division=0)), 6)
            metrics["precision_weighted"] = round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
            metrics["recall_weighted"]    = round(float(recall_score(y_true, y_pred,    average="weighted", zero_division=0)), 6)

        else:  # REGRESSION
            mse = mean_squared_error(y_true, y_pred)
            metrics["r2"]   = round(float(r2_score(y_true, y_pred)), 6)
            metrics["mae"]  = round(float(mean_absolute_error(y_true, y_pred)), 6)
            metrics["mse"]  = round(float(mse), 6)
            metrics["rmse"] = round(float(np.sqrt(mse)), 6)

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Final pipeline assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _build_final_pipeline(
        self,
        feature_pipeline: Pipeline,
        estimator: Any,
    ) -> Pipeline:
        """
        Wraps the fitted feature pipeline and best estimator into a single
        sklearn Pipeline. This pipeline:
          - Accepts a raw pandas DataFrame (same format as the training data)
          - Applies all preprocessing transformations
          - Returns predictions directly

        This single object is what gets serialized to .joblib and later
        loaded for inference — zero risk of train/serve skew.

        IMPORTANT: The feature_pipeline is already fitted (from the preprocessor).
        We add the estimator as the second step. The combined pipeline's
        transform step is handled by the preprocessor step (already fitted),
        while predict calls the estimator step.
        """
        # Clone the fitted preprocessor step and the estimator into a new Pipeline
        # Note: feature_pipeline.named_steps["preprocessor"] is the ColumnTransformer
        return Pipeline([
            ("preprocessor", feature_pipeline.named_steps["preprocessor"]),
            ("model",        estimator),
        ])

"""
app/models/schemas.py

All Pydantic models used across the AutoML pipeline.

Design principles:
  - Every data boundary (API request/response, internal service contracts,
    Supabase DB rows) has an explicit schema here.
  - No raw dicts cross module boundaries — always use a typed model.
  - Schemas are read-only (frozen where appropriate) to prevent accidental mutation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMS
# =============================================================================

class TaskType(str, Enum):
    """ML problem type — drives which models and metrics are used."""
    CLASSIFICATION = "classification"
    REGRESSION     = "regression"


class ProjectStatus(str, Enum):
    """Lifecycle states of a training job (mirrors the DB CHECK constraint)."""
    PENDING   = "pending"
    TRAINING  = "training"
    COMPLETED = "completed"
    FAILED    = "failed"


class ModelName(str, Enum):
    """Canonical names of the candidate models the pipeline will try."""
    # Classification
    RANDOM_FOREST_CLASSIFIER  = "RandomForestClassifier"
    LOGISTIC_REGRESSION       = "LogisticRegression"
    GRADIENT_BOOSTING_CLF     = "GradientBoostingClassifier"
    # Regression
    RANDOM_FOREST_REGRESSOR   = "RandomForestRegressor"
    LINEAR_REGRESSION         = "LinearRegression"
    GRADIENT_BOOSTING_REG     = "GradientBoostingRegressor"


# =============================================================================
# API REQUEST / RESPONSE SCHEMAS
# =============================================================================

class TrainRequest(BaseModel):
    """
    Payload the frontend POSTs to /train/ to kick off a training job.
    All three fields are required — the pipeline cannot start without them.
    """
    project_id: str = Field(
        ...,
        description="UUID of the projects row in Supabase.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    file_path: str = Field(
        ...,
        description=(
            "Path inside the 'datasets' Supabase Storage bucket. "
            "Format: {user_id}/{project_id}/filename.csv"
        ),
        examples=["abc123/550e8400/customer_data.csv"],
    )
    target_column: str = Field(
        ...,
        description="Name of the CSV column to predict (the label / y variable).",
        examples=["churn", "price", "diagnosis"],
        min_length=1,
    )

    @field_validator("file_path")
    @classmethod
    def file_path_must_be_csv(cls, v: str) -> str:
        if not v.lower().endswith(".csv"):
            raise ValueError("file_path must point to a .csv file")
        return v

    @field_validator("target_column")
    @classmethod
    def target_column_no_whitespace(cls, v: str) -> str:
        return v.strip()


class TrainResponse(BaseModel):
    """Immediate acknowledgement returned when a training job is accepted."""
    success:    bool
    project_id: str
    status:     ProjectStatus
    message:    str


class PredictRequest(BaseModel):
    """
    Input to POST /predict/{project_id}.
    `features` is a flat key→value map of the input row to classify/regress.
    """
    features: dict[str, Any] = Field(
        ...,
        description=(
            "Feature column names mapped to their values. "
            "Must include all columns the model was trained on "
            "(excluding the target column)."
        ),
        examples=[{"age": 35, "tenure_months": 24, "monthly_charges": 65.5}],
        min_length=1,
    )


class PredictResponse(BaseModel):
    """Output returned by POST /predict/{project_id}."""
    success:          bool
    project_id:       str
    task_type:        TaskType
    prediction:       Any           # float for regression, str/int for classification
    prediction_label: str | None    # Human-readable class label (classification only)
    probability:      float | None  # Confidence of winning class (classification only)
    all_probabilities: dict[str, float] | None  # Full class→prob map (classification only)
    model_name:       str
    latency_ms:       int


# =============================================================================
# INTERNAL PIPELINE SCHEMAS
# These cross the boundary between the route handler and the service layer.
# =============================================================================

class DatasetProfile(BaseModel):
    """
    Summary statistics about a dataset, produced by the preprocessor.
    Stored in memory during the pipeline run; not persisted to DB.
    """
    n_rows:              int
    n_cols:              int
    n_features:          int    # After target column removed
    feature_columns:     list[str]
    numeric_features:    list[str]
    categorical_features: list[str]
    target_column:       str
    task_type:           TaskType
    class_counts:        dict[str, int] | None   # Classification only
    target_min:          float | None            # Regression only
    target_max:          float | None            # Regression only
    target_mean:         float | None            # Regression only
    missing_values_total: int
    missing_by_column:   dict[str, int]


class ModelResult(BaseModel):
    """
    Results from training and evaluating a single candidate model.
    Produced by the trainer; multiple results are compared to pick the winner.
    """
    model_name:    str
    primary_score: float          # The metric used for model selection
    metric_name:   str            # e.g. "f1_weighted" or "r2"
    all_metrics:   dict[str, float]  # Full dict of computed metrics
    training_time_sec: float


class TrainingResult(BaseModel):
    """
    Final output of the full AutoML pipeline run.
    Returned by AutoMLService.run() and used to update the DB row.
    """
    project_id:      str
    task_type:       TaskType
    winning_model:   str
    primary_score:   float
    metric_name:     str
    all_scores:      dict[str, float]    # model_name → primary_score for all candidates
    all_metrics:     dict[str, dict[str, float]]  # model_name → full metrics
    feature_columns: list[str]
    model_path:      str                 # Path inside Supabase models bucket
    dataset_profile: DatasetProfile
    total_training_time_sec: float


# =============================================================================
# SUPABASE ROW SCHEMAS
# Typed representations of what we read from / write to the projects table.
# =============================================================================

class ProjectRow(BaseModel):
    """
    Represents a row from the public.projects table.
    Used when loading a project's metadata for prediction.
    """
    id:              str
    user_id:         str
    name:            str
    status:          ProjectStatus
    task_type:       TaskType | None      = None
    winning_model:   str | None           = None
    accuracy_score:  float | None         = None
    metric_name:     str | None           = None
    all_scores:      dict | None          = None
    model_path:      str | None           = None
    target_column:   str | None           = None
    feature_columns: list[str] | None     = None
    dataset_path:    str | None           = None
    error_message:   str | None           = None
    created_at:      datetime | None      = None
    trained_at:      datetime | None      = None

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def validate_completed_project(self) -> "ProjectRow":
        """If status is completed, all training fields must be present."""
        if self.status == ProjectStatus.COMPLETED:
            missing = [
                f for f in ("task_type", "winning_model", "model_path", "feature_columns")
                if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(
                    f"Project {self.id} has status=completed but "
                    f"missing required fields: {missing}"
                )
        return self

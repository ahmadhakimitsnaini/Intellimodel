"""
app/models/schemas.py

All Pydantic models used across the AutoML pipeline.

Design principles:
  - Every data boundary (API request/response, internal service contracts,
    Supabase DB rows) has an explicit schema here.
  - No raw dicts cross module boundaries — always use a typed model.
  - Schemas are read-only (frozen where appropriate) to prevent accidental mutation.
"""

"""
app/domain/models.py

Domain models and Data Transfer Objects (DTOs) for the AutoML pipeline.
Grouped by business context (Core/Project, Training, Prediction) rather than technical layers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# SHARED / CORE DOMAIN
# =============================================================================

class TaskType(str, Enum):
    """ML problem type — drives which models and metrics are used."""
    CLASSIFICATION = "classification"
    REGRESSION     = "regression"


class ProjectStatus(str, Enum):
    """Lifecycle states of a training job."""
    PENDING   = "pending"
    TRAINING  = "training"
    COMPLETED = "completed"
    FAILED    = "failed"


class ModelName(str, Enum):
    """Canonical names of the candidate models the pipeline will try."""
    RANDOM_FOREST_CLASSIFIER  = "RandomForestClassifier"
    LOGISTIC_REGRESSION       = "LogisticRegression"
    GRADIENT_BOOSTING_CLF     = "GradientBoostingClassifier"
    RANDOM_FOREST_REGRESSOR   = "RandomForestRegressor"
    LINEAR_REGRESSION         = "LinearRegression"
    GRADIENT_BOOSTING_REG     = "GradientBoostingRegressor"


class ProjectRow(BaseModel):
    """
    Domain entity representing a Project in the database.
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


# =============================================================================
# TRAINING DOMAIN (Internal Models & API DTOs)
# =============================================================================

class DatasetProfile(BaseModel):
    """Internal Domain Model: Summary statistics about a dataset."""
    n_rows:              int
    n_cols:              int
    n_features:          int
    feature_columns:     list[str]
    numeric_features:    list[str]
    categorical_features: list[str]
    target_column:       str
    task_type:           TaskType
    class_counts:        dict[str, int] | None
    target_min:          float | None
    target_max:          float | None
    target_mean:         float | None
    missing_values_total: int
    missing_by_column:   dict[str, int]


class ModelResult(BaseModel):
    """Internal Domain Model: Results from training a single candidate model."""
    model_name:    str
    primary_score: float
    metric_name:   str
    all_metrics:   dict[str, float]
    training_time_sec: float


class TrainingResult(BaseModel):
    """Internal Domain Model: Final output of the full AutoML pipeline run."""
    project_id:      str
    task_type:       TaskType
    winning_model:   str
    primary_score:   float
    metric_name:     str
    all_scores:      dict[str, float]
    all_metrics:     dict[str, dict[str, float]]
    feature_columns: list[str]
    model_path:      str
    dataset_profile: DatasetProfile
    total_training_time_sec: float


class TrainRequest(BaseModel):
    """API DTO: Payload to kick off a training job."""
    project_id: str = Field(..., description="UUID of the projects row in Supabase.")
    file_path: str = Field(
        ...,
        description="Path inside the 'datasets' Supabase Storage bucket.",
        examples=["abc123/550e8400/customer_data.csv"],
    )
    target_column: str = Field(..., description="Name of the CSV column to predict.", min_length=1)

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
    """API DTO: Acknowledgement returned when a training job is accepted."""
    success:    bool
    project_id: str
    status:     ProjectStatus
    message:    str


# =============================================================================
# PREDICTION DOMAIN (API DTOs)
# =============================================================================

class PredictRequest(BaseModel):
    """API DTO: Input features to classify/regress."""
    features: dict[str, Any] = Field(
        ...,
        description="Feature column names mapped to their values.",
        examples=[{"age": 35, "tenure_months": 24, "monthly_charges": 65.5}],
        min_length=1,
    )


class PredictResponse(BaseModel):
    """API DTO: Output returned after inference."""
    success:          bool
    project_id:       str
    task_type:        TaskType
    prediction:       Any
    prediction_label: str | None
    probability:      float | None
    all_probabilities: dict[str, float] | None
    model_name:       str
    latency_ms:       int
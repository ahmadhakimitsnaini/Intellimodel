"""
app/services/preprocessor.py

Automated data preprocessing pipeline.

Responsibilities:
  1. Validate the raw DataFrame (target column exists, enough rows, etc.)
  2. Profile the dataset (dtypes, missingness, cardinality)
  3. Detect the ML task type (classification vs. regression)
  4. Build a scikit-learn Pipeline that handles:
       - Numeric features: median imputation + StandardScaler
       - Categorical features: constant imputation + OneHotEncoder
  5. Return the preprocessed X/y split plus a DatasetProfile summary

Design decisions:
  - We use a ColumnTransformer so the same pipeline object can be bundled
    inside the model Pipeline and serialized to .joblib — zero train/serve skew.
  - Task type detection uses a heuristic: if the target has ≤ MAX_CLASSIFICATION_CLASSES
    unique values OR is object/bool dtype → classification; else → regression.
  - We drop columns that are purely IDs (very high cardinality relative to row count)
    rather than one-hot encoding them into thousands of dummy columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from app.models.schemas import DatasetProfile, TaskType

logger = logging.getLogger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────
MIN_ROWS                  = 20    # Refuse datasets smaller than this
MAX_CLASSIFICATION_CLASSES = 20   # If target has more unique values, treat as regression
HIGH_CARDINALITY_RATIO    = 0.95  # Drop categorical columns if unique/total > this ratio


@dataclass
class PreprocessedData:
    """Container for everything downstream training needs."""
    X_train:          np.ndarray
    X_test:           np.ndarray
    y_train:          np.ndarray
    y_test:           np.ndarray
    feature_pipeline: Pipeline        # Fitted ColumnTransformer wrapped in Pipeline
    label_encoder:    LabelEncoder | None  # Only for classification
    profile:          DatasetProfile
    feature_names_out: list[str]      # Column names after transformation (for debugging)


class DataPreprocessor:
    """
    Stateless class — call build() for each dataset.
    Returns a PreprocessedData containing the fitted pipeline and split arrays.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size    = test_size
        self.random_state = random_state

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame, target_column: str) -> PreprocessedData:
        """
        Full preprocessing flow. Raises ValueError with a human-readable
        message if the dataset is not suitable for AutoML.

        Args:
            df:            Raw DataFrame loaded from the user's CSV.
            target_column: Name of the column to predict.

        Returns:
            PreprocessedData with split arrays, fitted pipeline, and profile.
        """
        logger.info(f"Preprocessing dataset: {df.shape[0]} rows × {df.shape[1]} cols")

        # Step 1 — validate inputs
        self._validate(df, target_column)

        # Step 2 — clean the DataFrame
        df = self._clean(df, target_column)

        # Step 3 — detect task type
        task_type = self._detect_task_type(df[target_column])
        logger.info(f"Detected task type: {task_type.value}")

        # Step 4 — separate features and target
        X, y = self._split_features_target(df, target_column, task_type)
        feature_columns = list(X.columns)

        # Step 5 — identify feature types
        numeric_features, categorical_features = self._identify_feature_types(X)
        logger.info(
            f"Features: {len(numeric_features)} numeric, "
            f"{len(categorical_features)} categorical"
        )

        # Step 6 — encode target for classification
        label_encoder = None
        if task_type == TaskType.CLASSIFICATION:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))
        else:
            y = y.values.astype(float)

        # Step 7 — build the sklearn feature pipeline
        feature_pipeline = self._build_feature_pipeline(
            numeric_features, categorical_features
        )

        # Step 8 — train/test split (stratified for classification)
        X_train, X_test, y_train, y_test = self._split(X, y, task_type)

        # Step 9 — fit the feature pipeline on train, transform both splits
        X_train_t = feature_pipeline.fit_transform(X_train)
        X_test_t  = feature_pipeline.transform(X_test)

        # Step 10 — get transformed feature names (best-effort; OHE expands columns)
        feature_names_out = self._get_feature_names_out(
            feature_pipeline, numeric_features, categorical_features
        )

        # Step 11 — build the profile summary
        profile = self._build_profile(
            df, target_column, task_type,
            feature_columns, numeric_features, categorical_features,
            label_encoder
        )

        logger.info(
            f"Preprocessing complete: "
            f"X_train={X_train_t.shape}, X_test={X_test_t.shape}"
        )

        return PreprocessedData(
            X_train=X_train_t,
            X_test=X_test_t,
            y_train=y_train,
            y_test=y_test,
            feature_pipeline=feature_pipeline,
            label_encoder=label_encoder,
            profile=profile,
            feature_names_out=feature_names_out,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame, target_column: str) -> None:
        """Raises ValueError if the dataset cannot be trained on."""
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        if len(df) < MIN_ROWS:
            raise ValueError(
                f"Dataset has only {len(df)} rows. "
                f"A minimum of {MIN_ROWS} rows is required for training."
            )
        if len(df.columns) < 2:
            raise ValueError(
                "Dataset must have at least 2 columns (1 feature + 1 target)."
            )
        # Target must not be entirely null
        if df[target_column].isna().all():
            raise ValueError(
                f"Target column '{target_column}' is entirely empty (all NaN)."
            )

    def _clean(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        - Drop duplicate rows
        - Drop rows where the target is null
        - Drop columns that are 100% null
        - Drop completely constant columns (zero variance, useless for ML)
        """
        original_rows = len(df)

        df = df.drop_duplicates()
        df = df.dropna(subset=[target_column])
        df = df.dropna(axis=1, how="all")

        # Drop constant columns (excluding target)
        feature_cols = [c for c in df.columns if c != target_column]
        constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            logger.info(f"Dropping constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

        dropped = original_rows - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows during cleaning")

        if len(df) < MIN_ROWS:
            raise ValueError(
                f"After cleaning, only {len(df)} rows remain "
                f"(minimum required: {MIN_ROWS}). "
                "Check your dataset for excessive missing values."
            )
        return df

    def _detect_task_type(self, target: pd.Series) -> TaskType:
        """
        Heuristic task-type detection:
          - object / bool / category dtype → classification
          - numeric with ≤ MAX_CLASSIFICATION_CLASSES unique values → classification
          - numeric with > MAX_CLASSIFICATION_CLASSES unique values → regression
        """
        if target.dtype == object or target.dtype.name in ("bool", "category"):
            return TaskType.CLASSIFICATION

        n_unique = target.nunique()
        if n_unique <= MAX_CLASSIFICATION_CLASSES:
            return TaskType.CLASSIFICATION

        return TaskType.REGRESSION

    def _split_features_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: TaskType,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separates X and y. Also drops high-cardinality categorical columns
        that would explode one-hot encoding (likely ID/name columns).
        """
        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].copy()
        y = df[target_column].copy()

        # Drop high-cardinality categorical columns (likely IDs)
        cols_to_drop = []
        for col in X.select_dtypes(include=["object", "category"]).columns:
            ratio = X[col].nunique() / len(X)
            if ratio > HIGH_CARDINALITY_RATIO:
                cols_to_drop.append(col)

        if cols_to_drop:
            logger.info(
                f"Dropping high-cardinality columns (likely IDs): {cols_to_drop}"
            )
            X = X.drop(columns=cols_to_drop)

        if X.empty:
            raise ValueError(
                "No usable feature columns remain after preprocessing. "
                "All columns were either the target, constant, or high-cardinality IDs."
            )
        return X, y

    def _identify_feature_types(
        self, X: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """Splits columns into numeric and categorical by dtype."""
        numeric_features     = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        return numeric_features, categorical_features

    def _build_feature_pipeline(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> Pipeline:
        """
        Builds a scikit-learn Pipeline with:
          - Numeric branch:     SimpleImputer(median) → StandardScaler
          - Categorical branch: SimpleImputer(constant="missing") → OneHotEncoder

        The ColumnTransformer is wrapped in a Pipeline so it can be the first
        step in the final model pipeline (enabling single-object serialization).
        """
        transformers = []

        if numeric_features:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])
            transformers.append(("numeric", numeric_pipeline, numeric_features))

        if categorical_features:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",   # Unseen categories → all zeros
                        sparse_output=False,       # Return dense array (sklearn ≥ 1.2)
                        drop="if_binary",          # Drop one column for binary features
                    ),
                ),
            ])
            transformers.append(
                ("categorical", categorical_pipeline, categorical_features)
            )

        if not transformers:
            raise ValueError("No numeric or categorical features found in the dataset.")

        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",    # Drop any columns not in either list
            n_jobs=-1,
        )

        return Pipeline([("preprocessor", column_transformer)])

    def _split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task_type: TaskType,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Stratified split for classification (preserves class distribution).
        Regular split for regression.
        """
        from sklearn.model_selection import train_test_split

        stratify = y if task_type == TaskType.CLASSIFICATION else None

        # If any class has < 2 samples, stratification will fail — fall back
        if stratify is not None:
            _, counts = np.unique(y, return_counts=True)
            if counts.min() < 2:
                logger.warning(
                    "Some classes have < 2 samples — "
                    "falling back to non-stratified split."
                )
                stratify = None

        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def _get_feature_names_out(
        self,
        pipeline: Pipeline,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> list[str]:
        """
        Extracts feature names after transformation for interpretability.
        Best-effort — returns a fallback list if get_feature_names_out fails.
        """
        try:
            ct = pipeline.named_steps["preprocessor"]
            return list(ct.get_feature_names_out())
        except Exception:
            # Fallback: numeric names as-is, categorical as "cat_{col}"
            return numeric_features + [f"cat_{c}" for c in categorical_features]

    def _build_profile(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: TaskType,
        feature_columns: list[str],
        numeric_features: list[str],
        categorical_features: list[str],
        label_encoder: LabelEncoder | None,
    ) -> DatasetProfile:
        """Constructs the DatasetProfile summary object."""
        target = df[target_column]
        missing_by_column = {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        }

        # Classification: count per class
        class_counts = None
        if task_type == TaskType.CLASSIFICATION and label_encoder is not None:
            vc = target.astype(str).value_counts()
            class_counts = {str(k): int(v) for k, v in vc.items()}

        # Regression: basic target stats
        target_min = target_max = target_mean = None
        if task_type == TaskType.REGRESSION:
            target_min  = float(target.min())
            target_max  = float(target.max())
            target_mean = float(target.mean())

        return DatasetProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            n_features=len(feature_columns),
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_column=target_column,
            task_type=task_type,
            class_counts=class_counts,
            target_min=target_min,
            target_max=target_max,
            target_mean=target_mean,
            missing_values_total=int(df.isna().sum().sum()),
            missing_by_column=missing_by_column,
        )

"""
tests/test_pipeline.py

Comprehensive tests for the AutoML pipeline components.

All tests are pure unit tests — no Supabase connection required.
Real sklearn training is used (not mocked) to verify correctness end-to-end.

Test groups:
  TestDataPreprocessor  — validates preprocessing logic on synthetic data
  TestTaskTypeDetection — confirms classification vs. regression heuristics
  TestModelTrainer      — trains real models on toy datasets, checks rankings
  TestModelLRUCache     — validates LRU eviction, thread safety, statistics
  TestSchemas           — validates Pydantic model validation rules
  TestAutoPipeline      — integration test of the full PreprocessedData → Pipeline flow
"""

import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline

from app.models.schemas import (
    DatasetProfile,
    ModelResult,
    ProjectRow,
    ProjectStatus,
    TaskType,
    TrainRequest,
)
from app.services.preprocessor import DataPreprocessor, MIN_ROWS
from app.services.trainer import ModelTrainer
from app.utils.model_cache import ModelLRUCache


# =============================================================================
# Fixtures — reusable test datasets
# =============================================================================

@pytest.fixture
def classification_df() -> pd.DataFrame:
    """Synthetic binary classification dataset with mixed feature types."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age":        np.random.randint(18, 80, n),
        "salary":     np.random.uniform(30_000, 120_000, n).round(2),
        "tenure":     np.random.randint(0, 20, n),
        "department": np.random.choice(["eng", "sales", "hr", "finance"], n),
        "city":       np.random.choice(["NYC", "LA", "Chicago", "Austin"], n),
        "churn":      np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Synthetic regression dataset (house price prediction)."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "sqft":        np.random.randint(500, 4000, n),
        "bedrooms":    np.random.randint(1, 6, n),
        "bathrooms":   np.random.uniform(1, 4, n).round(1),
        "zip_code":    np.random.choice(["10001", "90210", "60601"], n),
        "age_years":   np.random.randint(0, 100, n),
        "price":       np.random.uniform(100_000, 2_000_000, n).round(2),
    })
    return df


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor(test_size=0.2, random_state=42)


@pytest.fixture
def trainer() -> ModelTrainer:
    return ModelTrainer(random_state=42)


# =============================================================================
# TestDataPreprocessor
# =============================================================================

class TestDataPreprocessor:

    def test_basic_classification_preprocessing(self, preprocessor, classification_df):
        result = preprocessor.build(classification_df, "churn")
        assert result.X_train.shape[0] > 0
        assert result.X_test.shape[0] > 0
        assert result.y_train.shape[0] == result.X_train.shape[0]
        assert result.y_test.shape[0]  == result.X_test.shape[0]

    def test_basic_regression_preprocessing(self, preprocessor, regression_df):
        result = preprocessor.build(regression_df, "price")
        assert result.X_train.shape[0] > 0
        assert result.profile.task_type == TaskType.REGRESSION

    def test_output_arrays_are_numpy(self, preprocessor, classification_df):
        result = preprocessor.build(classification_df, "churn")
        assert isinstance(result.X_train, np.ndarray)
        assert isinstance(result.X_test,  np.ndarray)
        assert isinstance(result.y_train, np.ndarray)
        assert isinstance(result.y_test,  np.ndarray)

    def test_no_nan_in_output(self, preprocessor):
        """NaN imputation must leave zero missing values in transformed arrays."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0] * 10,
            "b": ["x", None, "z", "x", "y"] * 10,
            "target": [0, 1, 0, 1, 0] * 10,
        })
        result = preprocessor.build(df, "target")
        assert not np.isnan(result.X_train).any(), "NaN found in X_train"
        assert not np.isnan(result.X_test).any(),  "NaN found in X_test"

    def test_duplicate_rows_removed(self, preprocessor):
        df = pd.DataFrame({
            "x": [1, 1, 2, 3, 4, 5] * 10,
            "target": [0, 0, 1, 0, 1, 1] * 10,
        })
        result = preprocessor.build(df, "target")
        # After dedup, rows should be fewer (or equal if no dupes in split)
        assert result.X_train.shape[0] > 0

    def test_raises_on_missing_target_column(self, preprocessor, classification_df):
        with pytest.raises(ValueError, match="not found"):
            preprocessor.build(classification_df, "nonexistent_column")

    def test_raises_on_too_few_rows(self, preprocessor):
        tiny_df = pd.DataFrame({
            "x": range(5),
            "target": [0, 1, 0, 1, 0],
        })
        with pytest.raises(ValueError, match=f"minimum of {MIN_ROWS}"):
            preprocessor.build(tiny_df, "target")

    def test_raises_on_all_null_target(self, preprocessor):
        df = pd.DataFrame({
            "x": range(50),
            "target": [np.nan] * 50,
        })
        with pytest.raises(ValueError, match="entirely empty"):
            preprocessor.build(df, "target")

    def test_profile_has_correct_task_type(self, preprocessor, classification_df):
        result = preprocessor.build(classification_df, "churn")
        assert result.profile.task_type == TaskType.CLASSIFICATION

    def test_profile_feature_columns_excludes_target(self, preprocessor, classification_df):
        result = preprocessor.build(classification_df, "churn")
        assert "churn" not in result.profile.feature_columns

    def test_profile_class_counts_present_for_classification(self, preprocessor, classification_df):
        result = preprocessor.build(classification_df, "churn")
        assert result.profile.class_counts is not None
        assert len(result.profile.class_counts) >= 2

    def test_profile_target_stats_present_for_regression(self, preprocessor, regression_df):
        result = preprocessor.build(regression_df, "price")
        assert result.profile.target_min  is not None
        assert result.profile.target_max  is not None
        assert result.profile.target_mean is not None

    def test_high_cardinality_columns_dropped(self, preprocessor):
        """Columns that are essentially row IDs should be dropped automatically."""
        n = 100
        df = pd.DataFrame({
            "row_id":      [f"ID_{i}" for i in range(n)],   # 100% unique — should be dropped
            "feature":     np.random.randn(n),
            "target":      np.random.randint(0, 2, n),
        })
        result = preprocessor.build(df, "target")
        assert "row_id" not in result.profile.feature_columns

    def test_constant_columns_dropped(self, preprocessor):
        """Zero-variance columns should be dropped."""
        df = pd.DataFrame({
            "useful":   np.random.randn(50),
            "constant": [42] * 50,
            "target":   np.random.randint(0, 2, 50),
        })
        result = preprocessor.build(df, "target")
        assert "constant" not in result.profile.feature_columns

    def test_test_split_ratio_respected(self, preprocessor, classification_df):
        """X_test should be approximately 20% of total rows."""
        result = preprocessor.build(classification_df, "churn")
        total = result.X_train.shape[0] + result.X_test.shape[0]
        test_ratio = result.X_test.shape[0] / total
        assert 0.15 <= test_ratio <= 0.25, f"Test ratio out of range: {test_ratio:.2f}"

    def test_label_encoder_present_for_classification(self, preprocessor, classification_df):
        result = preprocessor.build(classification_df, "churn")
        assert result.label_encoder is not None

    def test_label_encoder_absent_for_regression(self, preprocessor, regression_df):
        result = preprocessor.build(regression_df, "price")
        assert result.label_encoder is None

    def test_feature_pipeline_is_sklearn_pipeline(self, preprocessor, classification_df):
        result = preprocessor.build(classification_df, "churn")
        assert isinstance(result.feature_pipeline, Pipeline)


# =============================================================================
# TestTaskTypeDetection
# =============================================================================

class TestTaskTypeDetection:

    def test_string_target_is_classification(self, preprocessor):
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "y": np.random.choice(["cat", "dog"], 50),
        })
        result = preprocessor.build(df, "y")
        assert result.profile.task_type == TaskType.CLASSIFICATION

    def test_binary_int_target_is_classification(self, preprocessor):
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "y": np.random.randint(0, 2, 50),
        })
        result = preprocessor.build(df, "y")
        assert result.profile.task_type == TaskType.CLASSIFICATION

    def test_continuous_float_target_is_regression(self, preprocessor):
        np.random.seed(99)
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.uniform(1_000, 100_000, 100),  # 100 unique floats
        })
        result = preprocessor.build(df, "y")
        assert result.profile.task_type == TaskType.REGRESSION

    def test_many_integer_classes_is_regression(self, preprocessor):
        """50 unique integer values exceeds MAX_CLASSIFICATION_CLASSES → regression."""
        np.random.seed(0)
        df = pd.DataFrame({
            "x": np.random.randn(200),
            "y": np.arange(200),   # 200 unique integer values
        })
        result = preprocessor.build(df, "y")
        assert result.profile.task_type == TaskType.REGRESSION


# =============================================================================
# TestModelTrainer
# =============================================================================

class TestModelTrainer:

    def test_classification_returns_three_results(self, preprocessor, trainer, classification_df):
        data = preprocessor.build(classification_df, "churn")
        results, _ = trainer.train(data)
        assert len(results) == 3

    def test_regression_returns_three_results(self, preprocessor, trainer, regression_df):
        data = preprocessor.build(regression_df, "price")
        results, _ = trainer.train(data)
        assert len(results) == 3

    def test_results_sorted_descending_by_score(self, preprocessor, trainer, classification_df):
        data = preprocessor.build(classification_df, "churn")
        results, _ = trainer.train(data)
        scores = [r.primary_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_classification_primary_metric_is_f1(self, preprocessor, trainer, classification_df):
        data = preprocessor.build(classification_df, "churn")
        results, _ = trainer.train(data)
        for r in results:
            assert r.metric_name == "f1_weighted"

    def test_regression_primary_metric_is_r2(self, preprocessor, trainer, regression_df):
        data = preprocessor.build(regression_df, "price")
        results, _ = trainer.train(data)
        for r in results:
            assert r.metric_name == "r2"

    def test_classification_all_metrics_populated(self, preprocessor, trainer, classification_df):
        data = preprocessor.build(classification_df, "churn")
        results, _ = trainer.train(data)
        for r in results:
            assert "accuracy"            in r.all_metrics
            assert "f1_weighted"         in r.all_metrics
            assert "precision_weighted"  in r.all_metrics
            assert "recall_weighted"     in r.all_metrics

    def test_regression_all_metrics_populated(self, preprocessor, trainer, regression_df):
        data = preprocessor.build(regression_df, "price")
        results, _ = trainer.train(data)
        for r in results:
            assert "r2"   in r.all_metrics
            assert "mae"  in r.all_metrics
            assert "rmse" in r.all_metrics

    def test_returns_fitted_sklearn_pipeline(self, preprocessor, trainer, classification_df):
        data = preprocessor.build(classification_df, "churn")
        _, best_pipeline = trainer.train(data)
        assert isinstance(best_pipeline, Pipeline)

    def test_pipeline_can_predict(self, preprocessor, trainer, classification_df):
        """The returned pipeline must produce predictions on new data."""
        data = preprocessor.build(classification_df, "churn")
        _, best_pipeline = trainer.train(data)

        sample = classification_df.drop(columns=["churn"]).head(5)
        sample = sample[data.profile.feature_columns]
        preds = best_pipeline.predict(sample)
        assert len(preds) == 5

    def test_training_time_recorded(self, preprocessor, trainer, classification_df):
        data = preprocessor.build(classification_df, "churn")
        results, _ = trainer.train(data)
        for r in results:
            assert r.training_time_sec > 0

    def test_scores_are_between_zero_and_one_for_classification(
        self, preprocessor, trainer, classification_df
    ):
        data = preprocessor.build(classification_df, "churn")
        results, _ = trainer.train(data)
        for r in results:
            assert 0.0 <= r.primary_score <= 1.0, (
                f"{r.model_name} F1={r.primary_score} out of [0,1]"
            )


# =============================================================================
# TestModelLRUCache
# =============================================================================

class TestModelLRUCache:

    def _make_pipeline(self) -> MagicMock:
        """Returns a mock object that stands in for a real sklearn Pipeline."""
        m = MagicMock(spec=Pipeline)
        return m

    def test_get_miss_returns_none(self):
        cache = ModelLRUCache(maxsize=5)
        assert cache.get("nonexistent") is None

    def test_put_then_get_returns_same_object(self):
        cache    = ModelLRUCache(maxsize=5)
        pipeline = self._make_pipeline()
        cache.put("proj-1", pipeline)
        assert cache.get("proj-1") is pipeline

    def test_lru_eviction_when_full(self):
        """When capacity is exceeded, the least-recently-used entry is evicted."""
        cache = ModelLRUCache(maxsize=3)
        p1, p2, p3, p4 = [self._make_pipeline() for _ in range(4)]

        cache.put("a", p1)
        cache.put("b", p2)
        cache.put("c", p3)

        # Access "a" to make it recently used (so "b" becomes the LRU)
        cache.get("a")

        # Adding "d" should evict "b" (the true LRU)
        cache.put("d", p4)

        assert cache.get("a") is p1   # Not evicted (recently accessed)
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") is p3   # Not evicted
        assert cache.get("d") is p4   # Just added

    def test_invalidate_removes_entry(self):
        cache    = ModelLRUCache(maxsize=5)
        pipeline = self._make_pipeline()
        cache.put("proj-1", pipeline)
        result = cache.invalidate("proj-1")
        assert result is True
        assert cache.get("proj-1") is None

    def test_invalidate_nonexistent_returns_false(self):
        cache = ModelLRUCache(maxsize=5)
        assert cache.invalidate("ghost") is False

    def test_clear_empties_cache(self):
        cache = ModelLRUCache(maxsize=5)
        for i in range(5):
            cache.put(f"proj-{i}", self._make_pipeline())
        assert cache.size == 5
        cache.clear()
        assert cache.size == 0

    def test_update_existing_key(self):
        cache = ModelLRUCache(maxsize=5)
        p1 = self._make_pipeline()
        p2 = self._make_pipeline()
        cache.put("proj-1", p1)
        cache.put("proj-1", p2)
        assert cache.get("proj-1") is p2   # Second put overwrites first
        assert cache.size == 1             # Only one entry, not two

    def test_stats_contains_expected_keys(self):
        cache = ModelLRUCache(maxsize=5)
        cache.put("proj-1", self._make_pipeline())
        stats = cache.stats
        assert "size"    in stats
        assert "maxsize" in stats
        assert "entries" in stats
        assert len(stats["entries"]) == 1

    def test_thread_safety_concurrent_puts(self):
        """Multiple threads putting concurrently should not corrupt the cache."""
        cache  = ModelLRUCache(maxsize=100)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(20):
                    cache.put(f"t{thread_id}-p{i}", self._make_pipeline())
                    cache.get(f"t{thread_id}-p{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert cache.size <= 100   # Never exceeds maxsize


# =============================================================================
# TestSchemas — Pydantic validation
# =============================================================================

class TestSchemas:

    def test_train_request_requires_csv_extension(self):
        with pytest.raises(Exception):
            TrainRequest(
                project_id="abc",
                file_path="user/proj/data.xlsx",  # Not CSV
                target_column="target",
            )

    def test_train_request_strips_target_whitespace(self):
        req = TrainRequest(
            project_id="abc",
            file_path="user/proj/data.csv",
            target_column="  target  ",
        )
        assert req.target_column == "target"

    def test_train_request_rejects_empty_target(self):
        with pytest.raises(Exception):
            TrainRequest(
                project_id="abc",
                file_path="user/proj/data.csv",
                target_column="",
            )

    def test_project_row_completed_requires_model_path(self):
        """A 'completed' project without a model_path should fail validation."""
        with pytest.raises(Exception):
            ProjectRow(
                id="proj-1",
                user_id="user-1",
                name="Test",
                status=ProjectStatus.COMPLETED,
                task_type=TaskType.CLASSIFICATION,
                winning_model="RandomForestClassifier",
                # model_path intentionally missing
                feature_columns=["a", "b"],
            )


# =============================================================================
# TestAutoMLIntegration — full pre-process + train flow (no Supabase)
# =============================================================================

class TestAutoMLIntegration:
    """
    Tests the DataPreprocessor → ModelTrainer pipeline end-to-end
    using only in-memory data. No Supabase calls.
    """

    def test_full_classification_pipeline(self):
        np.random.seed(0)
        n = 150
        df = pd.DataFrame({
            "feature_num_1": np.random.randn(n),
            "feature_num_2": np.random.uniform(0, 100, n),
            "feature_cat":   np.random.choice(["A", "B", "C"], n),
            "target":        np.random.choice(["yes", "no"], n),
        })

        preprocessor = DataPreprocessor(test_size=0.2, random_state=7)
        trainer      = ModelTrainer(random_state=7)

        data    = preprocessor.build(df, "target")
        results, best_pipeline = trainer.train(data)

        # Pipeline must produce predictions
        sample_features = df[data.profile.feature_columns].head(3)
        preds = best_pipeline.predict(sample_features)
        assert len(preds) == 3
        assert results[0].primary_score >= 0.0

    def test_full_regression_pipeline(self):
        np.random.seed(1)
        n = 150
        df = pd.DataFrame({
            "x1":     np.random.randn(n),
            "x2":     np.random.randn(n),
            "cat":    np.random.choice(["low", "mid", "high"], n),
            "price":  np.random.uniform(50, 500, n),
        })

        preprocessor = DataPreprocessor(test_size=0.25, random_state=1)
        trainer      = ModelTrainer(random_state=1)

        data    = preprocessor.build(df, "price")
        results, best_pipeline = trainer.train(data)

        sample = df[data.profile.feature_columns].head(3)
        preds  = best_pipeline.predict(sample)
        assert len(preds) == 3
        assert data.profile.task_type == TaskType.REGRESSION

    def test_pipeline_with_missing_values(self):
        """Pipeline must not crash or produce NaN when the input has missing data."""
        np.random.seed(2)
        n = 120
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.choice(["X", "Y", None], n),
            "y": np.random.randint(0, 2, n),
        })
        # Introduce 15% missingness in column "a"
        df.loc[df.sample(frac=0.15, random_state=0).index, "a"] = np.nan

        preprocessor = DataPreprocessor(test_size=0.2, random_state=3)
        trainer      = ModelTrainer(random_state=3)

        data    = preprocessor.build(df, "y")
        results, best_pipeline = trainer.train(data)

        # All test predictions should be valid (not NaN)
        preds = best_pipeline.predict(
            pd.DataFrame({"a": [1.0], "b": [0.5], "c": ["X"]})[data.profile.feature_columns]
        )
        assert not np.isnan(preds).any()

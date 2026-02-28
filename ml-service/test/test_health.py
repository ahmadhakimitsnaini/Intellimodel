"""
tests/test_health.py

Tests for the /health endpoints.
These are pure unit tests — they mock Supabase so no real connection is needed.

Run: pytest tests/test_health.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture: TestClient with mocked Supabase
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """
    Creates a FastAPI TestClient with all Supabase calls mocked out.
    This allows health endpoint tests to run without a real Supabase instance.
    """
    # Mock the Supabase client BEFORE importing the app,
    # so the lifespan startup doesn't fail.
    mock_supabase = MagicMock()

    # Mock DB query chain: .table().select().limit().execute()
    mock_execute = MagicMock()
    mock_execute.count = 0
    mock_supabase.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_execute

    # Mock Storage query chain: .storage.from_().list()
    mock_supabase.storage.from_.return_value.list.return_value = []

    with patch("app.core.supabase_client.get_supabase_client", return_value=mock_supabase):
        with patch("app.core.supabase_client.create_client", return_value=mock_supabase):
            # Import app after patching
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as test_client:
                yield test_client


# ---------------------------------------------------------------------------
# Tests: GET /health/
# ---------------------------------------------------------------------------
class TestLiveness:
    def test_liveness_returns_200(self, client):
        response = client.get("/health/")
        assert response.status_code == 200

    def test_liveness_returns_alive_status(self, client):
        data = client.get("/health/").json()
        assert data["status"] == "alive"

    def test_liveness_includes_timestamp(self, client):
        data = client.get("/health/").json()
        assert "timestamp" in data
        assert data["timestamp"] is not None

    def test_liveness_is_fast(self, client):
        """Liveness should respond in under 100ms — it does no I/O."""
        import time
        start = time.monotonic()
        client.get("/health/")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 100, f"Liveness took {elapsed_ms:.0f}ms — too slow"


# ---------------------------------------------------------------------------
# Tests: GET /health/detail
# ---------------------------------------------------------------------------
class TestDetailHealth:
    def test_detail_returns_200(self, client):
        response = client.get("/health/detail")
        assert response.status_code == 200

    def test_detail_includes_version(self, client):
        data = client.get("/health/detail").json()
        assert "version" in data
        assert data["version"] is not None

    def test_detail_includes_uptime(self, client):
        data = client.get("/health/detail").json()
        assert "uptime" in data
        assert "seconds" in data["uptime"]
        assert "human" in data["uptime"]
        assert data["uptime"]["seconds"] >= 0

    def test_detail_includes_dependencies(self, client):
        data = client.get("/health/detail").json()
        assert "dependencies" in data
        assert "supabase_db" in data["dependencies"]
        assert "supabase_storage" in data["dependencies"]

    def test_detail_includes_configuration(self, client):
        data = client.get("/health/detail").json()
        assert "configuration" in data
        config = data["configuration"]
        assert "datasets_bucket" in config
        assert "models_bucket" in config
        assert "test_split_ratio" in config

    def test_detail_does_not_expose_full_supabase_url(self, client):
        """Supabase URL in detail response should be truncated for security."""
        data = client.get("/health/detail").json()
        config = data["configuration"]
        # The full URL should be truncated with "..."
        assert "..." in config["supabase_url"]


# ---------------------------------------------------------------------------
# Tests: Root endpoint
# ---------------------------------------------------------------------------
class TestRoot:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_service_info(self, client):
        data = client.get("/").json()
        assert data["service"] == "AutoML ML Microservice"
        assert data["status"] == "running"
        assert "/docs" in data["docs"]


# ---------------------------------------------------------------------------
# Tests: Train stub endpoint
# ---------------------------------------------------------------------------
class TestTrainStub:
    def test_train_returns_200_with_valid_payload(self, client):
        payload = {
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "file_path": "abc123/550e8400/data.csv",
            "target_column": "churn",
        }
        response = client.post("/train/", json=payload)
        assert response.status_code == 200

    def test_train_returns_queued_status(self, client):
        payload = {
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "file_path": "abc123/550e8400/data.csv",
            "target_column": "churn",
        }
        data = client.post("/train/", json=payload).json()
        assert data["success"] is True
        assert data["status"] == "queued"

    def test_train_rejects_missing_fields(self, client):
        # Missing target_column
        payload = {
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "file_path": "abc123/550e8400/data.csv",
        }
        response = client.post("/train/", json=payload)
        assert response.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Tests: Predict stub endpoint
# ---------------------------------------------------------------------------
class TestPredictStub:
    def test_predict_returns_200_with_valid_payload(self, client):
        payload = {"features": {"age": 35, "tenure": 12, "charges": 65.5}}
        response = client.post(
            "/predict/550e8400-e29b-41d4-a716-446655440000",
            json=payload,
        )
        assert response.status_code == 200

    def test_predict_returns_project_id(self, client):
        project_id = "550e8400-e29b-41d4-a716-446655440000"
        payload = {"features": {"age": 35}}
        data = client.post(f"/predict/{project_id}", json=payload).json()
        assert data["project_id"] == project_id

    def test_predict_rejects_empty_features(self, client):
        # features key is required
        response = client.post(
            "/predict/550e8400-e29b-41d4-a716-446655440000",
            json={},
        )
        assert response.status_code == 422

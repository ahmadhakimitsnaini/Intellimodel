"""
app/core/config.py

Centralized configuration management using pydantic-settings.
All values are loaded from environment variables (or a .env file).

Usage:
    from app.core.config import settings
    print(settings.SUPABASE_URL)
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required .env keys (must be set before running the service):
        SUPABASE_URL            — Your project's Supabase REST URL
        SUPABASE_SERVICE_KEY    — Service role key (bypasses RLS — keep secret!)

    Optional (have sensible defaults):
        ENVIRONMENT             — "development" | "staging" | "production"
        ALLOWED_ORIGINS         — Comma-separated list of allowed CORS origins
        MODEL_CACHE_SIZE        — Number of models to keep hot in memory
        TRAINING_TIMEOUT_SEC    — Max seconds a training job can run
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env in the working directory
        env_file_encoding="utf-8",
        case_sensitive=True,       # Env var names are UPPER_CASE
        extra="ignore",            # Silently ignore unknown env vars
    )

    # ── Supabase ─────────────────────────────────────────────────────────────
    SUPABASE_URL: str = Field(
        ...,  # Required — no default
        description="Supabase project URL (e.g. https://xxxx.supabase.co)",
    )
    SUPABASE_SERVICE_KEY: str = Field(
        ...,  # Required — no default
        description=(
            "Supabase SERVICE ROLE key. "
            "Bypasses RLS. NEVER expose to the frontend."
        ),
    )

    # ── Supabase Storage Bucket Names ────────────────────────────────────────
    DATASETS_BUCKET: str = Field(
        default="datasets",
        description="Supabase Storage bucket for raw CSV uploads",
    )
    MODELS_BUCKET: str = Field(
        default="models",
        description="Supabase Storage bucket for serialized .joblib model files",
    )

    # ── Application ───────────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(
        default="development",
        description="Runtime environment: development | staging | production",
    )
    APP_VERSION: str = Field(default="1.0.0")

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description=(
            "List of frontend origins allowed by CORS. "
            "In production, set this to your actual domain(s)."
        ),
    )

    # ── ML Training Configuration ─────────────────────────────────────────────
    TRAINING_TIMEOUT_SEC: int = Field(
        default=300,
        description="Maximum seconds a single training job is allowed to run",
        ge=30,
        le=3600,
    )
    TEST_SPLIT_RATIO: float = Field(
        default=0.2,
        description="Fraction of data held out for model evaluation",
        ge=0.1,
        le=0.4,
    )
    RANDOM_STATE: int = Field(
        default=42,
        description="Random seed for reproducible model training",
    )

    # ── Model Caching ─────────────────────────────────────────────────────────
    MODEL_CACHE_SIZE: int = Field(
        default=10,
        description=(
            "Number of loaded models to keep in memory (LRU cache). "
            "Larger value = faster predictions, higher RAM usage."
        ),
        ge=1,
        le=100,
    )

    # ── Temporary File Handling ───────────────────────────────────────────────
    TEMP_DIR: str = Field(
        default="/tmp/automl",
        description="Local temp directory for downloading datasets before training",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}, got '{v}'")
        return v

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        """Allow ALLOWED_ORIGINS to be set as a comma-separated string in .env"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use this factory everywhere instead of instantiating Settings() directly.
    The @lru_cache ensures the .env file is only read once per process.
    """
    return Settings()


# Module-level singleton for convenient import:
#   from app.core.config import settings
settings: Settings = get_settings()

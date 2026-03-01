"""
app/core/supabase_client.py

Manages a single shared Supabase client instance for the entire FastAPI process.

Why a singleton?
  - The supabase-py client is not inherently async, but creating one per request
    is wasteful. A module-level singleton initialized at startup is the correct
    pattern for this architecture.
  - All storage and database calls in routes/services import `get_supabase_client()`
    which returns the same cached instance.

Authentication:
  - We ALWAYS use the SERVICE_ROLE key here (not the anon key).
  - The service role bypasses Row Level Security, which is intentional —
    this server-side worker needs to read any user's dataset and write
    model files back to their namespace.
  - NEVER send the service role key to the frontend or log it.
"""

import logging
from typing import Optional

from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — one client for the entire FastAPI process lifetime
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Returns the shared Supabase client instance, initializing it on first call.

    Called once at application startup (in main.py lifespan) and then
    accessed via app.state.supabase in route handlers, or imported directly
    in service modules.

    Returns:
        supabase.Client: Configured with SERVICE_ROLE key.

    Raises:
        RuntimeError: If SUPABASE_URL or SUPABASE_SERVICE_KEY are not set.
        Exception:    If the Supabase client fails to initialize.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not settings.SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured. "
            "Set it in your .env file or environment variables."
        )
    if not settings.SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_KEY is not configured. "
            "Set it in your .env file or environment variables."
        )

    try:
        logger.info(f"Initializing Supabase client → {settings.SUPABASE_URL[:40]}...")

        options = ClientOptions(
            # Persist session is false because this is a server-side service worker,
            # not a user-facing browser session.
            postgrest_client_timeout=30,   # 30s timeout on DB calls
            storage_client_timeout=120,    # 2min timeout on large file transfers
        )

        _supabase_client = create_client(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=settings.SUPABASE_SERVICE_KEY,
            options=options,
        )

        logger.info("✓ Supabase client created successfully")
        return _supabase_client

    except Exception as e:
        logger.error(f"✗ Failed to initialize Supabase client: {e}")
        raise


def close_supabase_client() -> None:
    """
    Cleans up the Supabase client on application shutdown.
    Called from the lifespan context manager in main.py.
    """
    global _supabase_client

    if _supabase_client is not None:
        # supabase-py doesn't have an explicit close() but we clear the reference
        # to allow garbage collection and signal clean shutdown in logs.
        _supabase_client = None
        logger.info("✓ Supabase client reference released")


def get_supabase_client_dependency() -> Client:
    """
    FastAPI dependency injection version.
    Use this in route function signatures with Depends():

        from fastapi import Depends
        from app.core.supabase_client import get_supabase_client_dependency

        @router.post("/train")
        async def train_model(
            supabase: Client = Depends(get_supabase_client_dependency)
        ):
            ...

    This is equivalent to get_supabase_client() but wrapped for FastAPI's DI system,
    making routes easier to unit test by swapping out the dependency.
    """
    return get_supabase_client()

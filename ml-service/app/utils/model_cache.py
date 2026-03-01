"""
app/utils/model_cache.py

In-memory LRU cache for loaded sklearn Pipelines.

Problem it solves:
  Every call to POST /predict/{project_id} would otherwise re-download and
  re-deserialize the .joblib file from Supabase Storage. For an actively-used
  model this adds 200–500ms latency on every request and wastes bandwidth.

Solution:
  A module-level LRU cache keyed by `project_id`. Loaded models stay hot in
  memory. When the cache is full (MODEL_CACHE_SIZE), the least-recently-used
  model is evicted.

Thread safety:
  Python's GIL protects the OrderedDict operations. For multi-process deployments
  (gunicorn with multiple workers) each worker has its own cache — acceptable
  because the underlying models are identical across workers.

Usage:
    from app.utils.model_cache import model_cache

    # Store after loading from Storage
    model_cache.put(project_id, pipeline)

    # Retrieve (returns None if not cached)
    pipeline = model_cache.get(project_id)

    # Invalidate (e.g., after re-training)
    model_cache.invalidate(project_id)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from sklearn.pipeline import Pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata stored alongside each cached model."""
    pipeline:       Pipeline
    project_id:     str
    loaded_at:      float = field(default_factory=time.time)
    hit_count:      int   = 0
    last_accessed:  float = field(default_factory=time.time)


class ModelLRUCache:
    """
    Thread-safe LRU cache for sklearn Pipeline objects.

    Capacity is set from settings.MODEL_CACHE_SIZE at instantiation.
    The internal store is an OrderedDict: most-recently-used items are moved
    to the end; when capacity is exceeded, the first (oldest) item is evicted.
    """

    def __init__(self, maxsize: int | None = None):
        self._maxsize = maxsize or settings.MODEL_CACHE_SIZE
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock  = threading.Lock()

        logger.info(f"ModelLRUCache initialized (maxsize={self._maxsize})")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get(self, project_id: str) -> Pipeline | None:
        """
        Retrieves a cached pipeline. Returns None on cache miss.
        On hit: moves the entry to the end (marks as most-recently-used)
        and increments the hit counter.
        """
        with self._lock:
            entry = self._store.get(project_id)
            if entry is None:
                logger.debug(f"Cache MISS: project={project_id}")
                return None

            # Move to end (most recently used)
            self._store.move_to_end(project_id)
            entry.hit_count    += 1
            entry.last_accessed = time.time()

            logger.debug(
                f"Cache HIT: project={project_id} | hits={entry.hit_count}"
            )
            return entry.pipeline

    def put(self, project_id: str, pipeline: Pipeline) -> None:
        """
        Stores a pipeline. If the key already exists, the entry is updated
        and moved to the end. If at capacity, the LRU (first) entry is evicted.
        """
        with self._lock:
            if project_id in self._store:
                # Update existing entry
                self._store.move_to_end(project_id)
                self._store[project_id].pipeline      = pipeline
                self._store[project_id].last_accessed = time.time()
                logger.debug(f"Cache UPDATE: project={project_id}")
                return

            # Evict LRU if at capacity
            if len(self._store) >= self._maxsize:
                evicted_id, _ = self._store.popitem(last=False)
                logger.debug(f"Cache EVICT: project={evicted_id} (LRU)")

            self._store[project_id] = CacheEntry(
                pipeline=pipeline,
                project_id=project_id,
            )
            logger.debug(
                f"Cache STORE: project={project_id} "
                f"| cache_size={len(self._store)}/{self._maxsize}"
            )

    def invalidate(self, project_id: str) -> bool:
        """
        Removes a specific entry from the cache.
        Returns True if the entry existed, False if it was not cached.
        """
        with self._lock:
            if project_id in self._store:
                del self._store[project_id]
                logger.info(f"Cache INVALIDATE: project={project_id}")
                return True
            return False

    def clear(self) -> None:
        """Clears all cached models. Used during graceful shutdown."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.info(f"Cache CLEAR: removed {count} entries")

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Current number of cached models."""
        return len(self._store)

    @property
    def stats(self) -> dict:
        """Returns a diagnostic snapshot of cache state."""
        with self._lock:
            return {
                "size":    len(self._store),
                "maxsize": self._maxsize,
                "entries": [
                    {
                        "project_id":    eid,
                        "hit_count":     e.hit_count,
                        "loaded_at":     e.loaded_at,
                        "last_accessed": e.last_accessed,
                    }
                    for eid, e in self._store.items()
                ],
            }


# ── Module-level singleton ────────────────────────────────────────────────────
# Imported everywhere via: from app.utils.model_cache import model_cache
model_cache: ModelLRUCache = ModelLRUCache()

"""
Persistent local cache — zero external accounts, zero API keys.

Streamlit's own @st.cache_data is already excellent for in-process
caching, but it doesn't survive a full app restart/redeploy. This module
adds a second layer using `diskcache` — a pure-Python, zero-config,
zero-account package that persists to a local SQLite-backed file — as a
free stand-in for what Upstash Redis would give you (shared, persistent
caching), minus the "shared across multiple server instances" part,
which genuinely does need a real external service.

When you're ready to add Upstash Redis (or any Redis), only this file
needs to change: swap the diskcache.Cache object for a redis-py client
with the same get/set interface, and every call site (data_fetcher.py
etc.) keeps working unmodified.
"""

import os
import time
import hashlib
import pickle
from functools import wraps
from typing import Any, Callable, Optional

try:
    import diskcache
    _DISKCACHE_AVAILABLE = True
except ImportError:
    _DISKCACHE_AVAILABLE = False

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"
)

_cache_instance = None


def get_cache():
    """Lazily-initialised singleton diskcache instance."""
    global _cache_instance
    if _cache_instance is None and _DISKCACHE_AVAILABLE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _cache_instance = diskcache.Cache(CACHE_DIR, size_limit=200 * 1024 * 1024)  # 200MB cap
    return _cache_instance


def _make_key(prefix: str, *args, **kwargs) -> str:
    raw = f"{prefix}:{args}:{sorted(kwargs.items())}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def cached_call(prefix: str, ttl_seconds: int, fn: Callable, *args, **kwargs) -> Any:
    """
    Check disk cache first; on miss, call fn(*args, **kwargs), store the
    result, return it. Falls back to calling fn() directly with no
    caching if diskcache isn't installed — never breaks functionality,
    only the speed benefit is lost.
    """
    cache = get_cache()
    if cache is None:
        return fn(*args, **kwargs)

    key = _make_key(prefix, *args, **kwargs)
    try:
        cached = cache.get(key, default=None)
        if cached is not None:
            value, stored_at = cached
            if time.time() - stored_at < ttl_seconds:
                return value
    except Exception:
        pass  # corrupted cache entry — fall through to a fresh fetch

    result = fn(*args, **kwargs)
    try:
        cache.set(key, (result, time.time()))
    except Exception:
        pass  # cache write failure should never break the app
    return result


def disk_cache(prefix: str, ttl_seconds: int = 300):
    """Decorator version of cached_call, for wrapping a function definition directly."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return cached_call(prefix, ttl_seconds, fn, *args, **kwargs)
        return wrapper
    return decorator


def cache_stats() -> dict:
    """Diagnostics for a settings/admin page — size, item count, hit info."""
    cache = get_cache()
    if cache is None:
        return {"available": False, "reason": "diskcache not installed"}
    try:
        return {
            "available": True,
            "item_count": len(cache),
            "volume_bytes": cache.volume(),
            "volume_mb": round(cache.volume() / (1024 * 1024), 2),
            "directory": CACHE_DIR,
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


def clear_cache():
    cache = get_cache()
    if cache is not None:
        try:
            cache.clear()
        except Exception:
            pass

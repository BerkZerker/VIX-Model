"""Health check endpoint for the VIX Alert Bot.

Provides /health and /status endpoints via FastAPI.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="VIX Alert Bot Health", version="0.1.0")

# Module-level state populated by main.py at startup
_state: dict = {
    "start_time": time.time(),
    "model_version": "unknown",
    "last_data_timestamp": None,
    "staleness_tracker": None,
    "db": None,
}


def set_state(
    model_version: str = "unknown",
    staleness_tracker=None,
    db=None,
) -> None:
    """Called by main.py to inject runtime state into the health module."""
    _state["model_version"] = model_version
    _state["staleness_tracker"] = staleness_tracker
    _state["db"] = db


def update_last_data_timestamp(ts: str) -> None:
    """Update the last successful data timestamp."""
    _state["last_data_timestamp"] = ts


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        # maxrss on macOS is in bytes, on Linux in KB
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return 0.0


@app.get("/health")
async def health():
    """Basic health check — is the bot alive?"""
    uptime = time.time() - _state["start_time"]
    memory_mb = _get_memory_mb()

    staleness_ok = True
    if _state["staleness_tracker"]:
        status = _state["staleness_tracker"].check()
        staleness_ok = not status.any_critical_stale

    return JSONResponse(
        content={
            "status": "healthy" if staleness_ok else "degraded",
            "uptime_seconds": round(uptime, 1),
            "model_version": _state["model_version"],
            "last_data_timestamp": _state["last_data_timestamp"],
            "memory_mb": round(memory_mb, 1),
            "data_fresh": staleness_ok,
        },
        status_code=200 if staleness_ok else 503,
    )


@app.get("/status")
async def status():
    """Detailed status with staleness breakdown."""
    uptime = time.time() - _state["start_time"]
    memory_mb = _get_memory_mb()

    staleness_detail = {}
    staleness_ok = True
    if _state["staleness_tracker"]:
        st = _state["staleness_tracker"].check()
        staleness_ok = not st.any_critical_stale
        now = datetime.now(timezone.utc)
        for sym, entry in st.entries.items():
            age = (now - entry.last_update).total_seconds() if entry.last_update.year > 1 else -1
            staleness_detail[sym] = {
                "stale": entry.is_stale,
                "age_seconds": round(age, 1),
                "last_update": entry.last_update.isoformat() if entry.last_update.year > 1 else None,
            }

    return JSONResponse(
        content={
            "status": "healthy" if staleness_ok else "degraded",
            "uptime_seconds": round(uptime, 1),
            "uptime_human": _format_uptime(uptime),
            "model_version": _state["model_version"],
            "last_data_timestamp": _state["last_data_timestamp"],
            "memory_mb": round(memory_mb, 1),
            "data_fresh": staleness_ok,
            "staleness": staleness_detail,
            "pid": os.getpid(),
        },
    )


def _format_uptime(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

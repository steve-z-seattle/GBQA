"""Lightweight debug logging utilities for GBQA."""

from __future__ import annotations

import os


def debug_log(msg: str) -> None:
    """Write a debug message to the file pointed by GBQA_DEBUG_LOG."""
    path = os.environ.get("GBQA_DEBUG_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    except Exception:
        pass

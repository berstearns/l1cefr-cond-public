"""l1cefr_cond — the train/eval experiment binaries for l1cefr-cond.

The algebraic domain model lives in ``domain/``; this package holds the console
entry points (``l1cefr-train``, ``l1cefr-eval``) wired in ``pyproject.toml``.
Both are currently STUBS — training/eval logic is deliberately absent until a
technique/eval binary is implemented (SCAFFOLD §3).
"""
from __future__ import annotations

__version__ = "0.0.0"

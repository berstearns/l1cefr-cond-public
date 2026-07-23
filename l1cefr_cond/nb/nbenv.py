"""Layered environment reader for notebooks — the templating substrate.

Public notebooks must contain **no** environment-specific value hardcoded (see
``marimos/auto-rules/notebooks-development/public-notebooks-are-templated-only-private-ones-may-hardcode.md``).
Instead every such value is read here with a fixed precedence:

    process env var  >  .env file (repo root)  >  Colab userdata  >  default

A *required* key that resolves to nothing raises with the key name — so a missing
value fails LOUD (naming what to set) instead of silently using the author's path.
The same source works on a laptop, a Jupyter kernel, and Colab; nothing here is
Colab-specific except the optional ``google.colab.userdata`` lookup (guarded).

This module has no third-party dependencies and never imports the batch pipeline.

GRAFTED VERBATIM from gen-gec-errant v2 (src/gen_gec_errant/nbenv.py) per
specs/96-lm-head-notebook-pairs.md — reused unchanged so the LM-head notebooks
inherit the identical env-detect substrate.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

# sentinel written by the converter/.env.template for values a public user must fill
PLACEHOLDER_PREFIX = "PLACEHOLDER_"


@lru_cache(maxsize=None)
def _dotenv(dotenv_path: Optional[str] = None) -> Dict[str, str]:
    """Parse a simple KEY=VALUE ``.env`` (no interpolation, ``#`` comments, quotes
    stripped). Searches CWD then parents for ``.env`` when no path is given.
    Returns {} when none is found — a missing .env is normal, not an error."""
    p = Path(dotenv_path) if dotenv_path else None
    if p is None:
        for base in [Path.cwd(), *Path.cwd().parents]:
            cand = base / ".env"
            if cand.is_file():
                p = cand
                break
    if p is None or not p.is_file():
        return {}
    out: Dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        out[key.strip()] = val
    return out


def _userdata(key: str) -> Optional[str]:
    """Colab Secrets (``userdata``) — returns None off Colab or when unset/denied."""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        v = userdata.get(key)
        return v if v else None
    except Exception:
        return None


def layered_get(
    key: str,
    default: Optional[str] = None,
    *,
    required: bool = False,
    dotenv_path: Optional[str] = None,
) -> Optional[str]:
    """Resolve ``key`` by precedence env var > .env > Colab userdata > default.

    A blank value at any layer is treated as unset and falls through. When
    ``required`` and the result is empty or a ``PLACEHOLDER_*``, raise a
    RuntimeError naming the key and the channels to set it.
    """
    for source in (os.environ.get(key), _dotenv(dotenv_path).get(key), _userdata(key)):
        if source is not None and str(source).strip() != "":
            return source
    val = default
    if required and (val is None or str(val).strip() == "" or str(val).startswith(PLACEHOLDER_PREFIX)):
        raise RuntimeError(
            f"required config '{key}' is unset — set it via an env var, a repo-root .env "
            f"(see .env.template), or Colab userdata."
        )
    return val


def is_placeholder(value: Optional[str]) -> bool:
    """True when a resolved value is still an unfilled ``PLACEHOLDER_*`` sentinel."""
    return value is None or str(value).startswith(PLACEHOLDER_PREFIX)

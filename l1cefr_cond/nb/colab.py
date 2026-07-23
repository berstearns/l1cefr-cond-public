"""Colab detection — the ``is_colab`` half of gen-gec-errant v2's ``colab.py``.

Only ``is_colab()`` is grafted (verbatim): the v2 ``resolve_model_path`` helper
pulled in the whole generation-config/torch stack, which the LM-head notebooks
do not need (base decoders load straight from the HF Hub via ``from_pretrained``).
Kept torch-free so importing the broker layer stays cheap.
"""
from __future__ import annotations


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False

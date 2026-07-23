"""domain — the algebraic domain model for l1cefr-cond (``mypy --strict`` gate).

Illegal states are unrepresentable: newtypes + validating smart constructors
for scalars (``L1``, ``CEFR``, ``AdapterId``, ``RunId``, ``Seed``), a frozen
value object for the grid unit (``Cell``), and a sum type for the conditioning
``Method``. Parse untrusted strings at the boundary; trust the types inside.
"""
from __future__ import annotations

from domain.adapter import AdapterId, RunId, Seed, adapter_id, run_id, seed
from domain.cefr import CEFR, cefr
from domain.cell import Cell, cell_id
from domain.l1 import L1, LANG2VEC_DIM, l1, lang2vec_features
from domain.method import (
    DPO,
    GRPO,
    ControlToken,
    DecodingTime,
    Hypernet,
    LoRA,
    Method,
    Steering,
    TaskVector,
    describe,
)

__all__ = [
    # scalars
    "L1", "l1", "LANG2VEC_DIM", "lang2vec_features",
    "CEFR", "cefr",
    "AdapterId", "adapter_id", "RunId", "run_id", "Seed", "seed",
    # products / sums
    "Cell", "cell_id",
    "Method", "describe",
    "ControlToken", "LoRA", "Hypernet", "DPO", "GRPO",
    "Steering", "DecodingTime", "TaskVector",
]

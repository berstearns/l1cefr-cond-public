"""The frozen 6 L1 × 5 CEFR conditioning grid — built on the domain boundary.

``N_L1=6`` (de,fr,es,zh,ja,ar — the ``domain.l1`` powered subset), ``N_CEFR=5``
(A1..C1 — the working grid drops C2, specs/90 Q-A). The head module's L1/CEFR
embeddings index into THIS order, so the order is load-bearing and frozen. Codes
are parsed at the domain boundary (``l1()``/``cefr()``) — an out-of-grid code
raises rather than silently mis-indexing.
"""
from __future__ import annotations

from typing import Iterator, Tuple

from domain.cefr import cefr
from domain.l1 import l1

# Frozen order — the embedding index of every cell. DO NOT reorder (it would
# silently relabel every trained embedding row).
L1_ORDER: Tuple[str, ...] = ("de", "fr", "es", "zh", "ja", "ar")
CEFR_ORDER: Tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1")

N_L1: int = len(L1_ORDER)      # 6
N_CEFR: int = len(CEFR_ORDER)  # 5

# validate the frozen order against the domain boundary at import time
_L1_SET = {l1(c) for c in L1_ORDER}          # each must be a powered L1
_CEFR_SET = {cefr(b) for b in CEFR_ORDER}    # each must be a valid CEFR band
assert len(_L1_SET) == N_L1, "L1_ORDER has a duplicate / non-powered code"
assert len(_CEFR_SET) == N_CEFR, "CEFR_ORDER has a duplicate / invalid band"

_L1_TO_IDX = {code: i for i, code in enumerate(L1_ORDER)}
_CEFR_TO_IDX = {band: i for i, band in enumerate(CEFR_ORDER)}


def l1_index(code: str) -> int:
    """Grid index for an L1 code (parsed at the domain boundary first)."""
    c = str(l1(code))
    if c not in _L1_TO_IDX:
        raise ValueError(f"L1 {c!r} is valid but not in the frozen grid {L1_ORDER}")
    return _L1_TO_IDX[c]


def cefr_index(band: str) -> int:
    """Grid index for a CEFR band (parsed at the domain boundary first)."""
    b = cefr(band).name
    if b not in _CEFR_TO_IDX:
        raise ValueError(f"CEFR {b!r} is valid but not in the working grid {CEFR_ORDER} (C2 dropped)")
    return _CEFR_TO_IDX[b]


def cells() -> Iterator[Tuple[str, str, int, int]]:
    """Iterate the 30-cell grid as ``(l1_code, cefr_band, l1_idx, cefr_idx)``."""
    for li, lc in enumerate(L1_ORDER):
        for ci, cb in enumerate(CEFR_ORDER):
            yield lc, cb, li, ci


if __name__ == "__main__":  # boundary check (good passes / bad raises)
    assert N_L1 == 6 and N_CEFR == 5
    assert l1_index("DE") == 0 and cefr_index("c1") == 4
    assert len(list(cells())) == 30
    try:
        cefr_index("C2")  # dropped from the working grid
    except ValueError as e:
        print(f"OK: C2 correctly rejected -> {e}")
    else:
        raise AssertionError("expected C2 to be rejected")
    print(f"OK: grid {N_L1}x{N_CEFR}=30 cells; L1_ORDER={L1_ORDER} CEFR_ORDER={CEFR_ORDER}")

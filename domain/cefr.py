"""domain/cefr.py — the CEFR proficiency axis as an ORDINAL sum type [S02].

``IntEnum`` makes the order *real*: ``CEFR.B2 > CEFR.B1`` is a true statement,
so monotonicity checks (predicted-level ordering, curriculum staging [T12]) are
plain comparisons. ``cefr()`` is the boundary that parses an untrusted band.
"""
from __future__ import annotations

from enum import IntEnum


class CEFR(IntEnum):
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


def cefr(s: str) -> CEFR:
    """Parse an untrusted CEFR band into the ordinal ``CEFR`` (the one boundary)."""
    try:
        return CEFR[s.strip().upper()]
    except KeyError:
        raise ValueError(f"not a CEFR band: {s!r} (allowed: {[c.name for c in CEFR]})") from None


if __name__ == "__main__":  # boundary check + the monotonicity invariant
    assert cefr("b2") is CEFR.B2
    assert cefr("A1") < cefr("B1") < cefr("C2"), "CEFR must be ordinal/monotone"
    try:
        cefr("D9")
    except ValueError as e:
        print(f"OK: bad CEFR rejected -> {e}")
    else:
        raise AssertionError("expected cefr('D9') to raise")
    print(f"OK: cefr('b2') = {cefr('b2')!r}; A1 < B1 < C2 holds")

"""domain/adapter.py — identity newtypes for the run/adapter estate.

``AdapterId``, ``RunId``, ``Seed`` are distinct KINDs even though two wrap
``str`` and one wraps ``int`` — no primitive obsession, so ``mypy --strict``
rejects passing a ``Seed`` where a ``RunId`` is expected. ``RunId`` is a UUID
(experiment-binaries: runs are UUID-keyed, append-only); ``Seed`` is bounded to
the numpy/torch RNG range.
"""
from __future__ import annotations

import uuid
from typing import NewType

AdapterId = NewType("AdapterId", str)  # the on-disk adapter slug (cell_id-shaped)
RunId = NewType("RunId", str)          # uuid4, the DB primary key
Seed = NewType("Seed", int)

_MAX_SEED = 2**32  # numpy/torch RNG seed range: [0, 2**32)


def adapter_id(s: str) -> AdapterId:
    """Parse a non-empty, whitespace-free adapter slug."""
    t = s.strip()
    if not t or any(ch.isspace() for ch in s):
        raise ValueError(f"adapter id must be non-empty and whitespace-free: {s!r}")
    return AdapterId(t)


def run_id(s: str) -> RunId:
    """Parse a UUID string into a ``RunId`` (experiment-binaries: UUID-keyed runs)."""
    try:
        return RunId(str(uuid.UUID(s.strip())))
    except ValueError:
        raise ValueError(f"run id must be a UUID, got {s!r}") from None


def seed(n: int) -> Seed:
    """Parse an int into a ``Seed`` bounded to [0, 2**32)."""
    if not 0 <= n < _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}), got {n}")
    return Seed(n)


if __name__ == "__main__":  # boundary check for each newtype (good builds / bad raises)
    rid = run_id(str(uuid.uuid4()))
    aid = adapter_id("pythia-410m__lora__de__B2__seed0")
    sd = seed(0)
    assert int(sd) == 0 and len(rid) == 36 and aid

    try:
        run_id("not-a-uuid")
    except ValueError as e:
        print(f"OK: bad run_id rejected -> {e}")
    else:
        raise AssertionError("expected run_id('not-a-uuid') to raise")

    try:
        seed(-1)
    except ValueError as e:
        print(f"OK: bad seed rejected -> {e}")
    else:
        raise AssertionError("expected seed(-1) to raise")

    try:
        adapter_id("  ")
    except ValueError as e:
        print(f"OK: bad adapter_id rejected -> {e}")
    else:
        raise AssertionError("expected adapter_id('  ') to raise")

    print(f"OK: run_id={rid} adapter_id={aid!r} seed={int(sd)}")

"""domain/l1.py — the L1 (native-language) axis as a proof [S01].

A scalar is a proof: ``L1`` is not a bare ``str``. ``mypy --strict`` enforces the
KIND (the mypy-only distinction); ``l1()`` — the one boundary — enforces the
VALUE (the runtime constraint). Untrusted EFCAMDAT nationality fields, CLI args
and config values are parsed here into ``L1`` and trusted everywhere inside.

Refs: S01 (the L1 knob — typological vectors; lang2vec/URIEL ``[R34]``).
"""
from __future__ import annotations

from typing import NewType

L1 = NewType("L1", str)  # the KIND — mypy-only distinction from a raw str

# The ISO-639-1 subset with enough EFCAMDAT signal to power a cell [EX01/S01].
# nationality != L1 — the corpus mapping happens upstream; this is the target set.
_ISO: frozenset[str] = frozenset({"de", "fr", "zh", "es", "ja", "ar"})


def l1(code: str) -> L1:
    """Parse an untrusted language code into an ``L1`` (the one boundary) [S01].

    Good input passes; anything outside the powered subset raises — never a
    silent ``None``/fallback (domain-errors-are-boundary-exceptions).
    """
    c = code.strip().lower()
    if c not in _ISO:
        raise ValueError(f"unknown/underpowered L1: {code!r} (allowed: {sorted(_ISO)})")
    return L1(c)  # the VALUE — validated at runtime


# --- lang2vec typological lookup (stub) [S01] --------------------------------
# S01: the L1 knob is a *typological vector* ``t`` (URIEL/lang2vec [R34], data
# frozen at 2019). That ``t`` is exactly what the hypernetwork [T07] conditions
# on to generate an adapter for an unseen L1.
LANG2VEC_DIM: int = 289  # URIEL syntax_knn + phonology_knn + inventory_knn feature count

# lang2vec keys on ISO-639-3; the 2->3 lift for our subset (used by a real impl).
_ISO3: dict[str, str] = {
    "de": "deu", "fr": "fra", "zh": "cmn", "es": "spa", "ja": "jpn", "ar": "arb",
}


def lang2vec_features(code: L1) -> tuple[float, ...]:
    """Typological feature vector for an ``L1`` [S01] (STUB).

    TODO[S01]: replace with a real ``lang2vec.get_features(_ISO3[code],
    "syntax_knn phonology_knn inventory_knn")`` lookup. Returns a fixed-dim
    zero vector for now so the *type* is honest ahead of the data.
    """
    _ = _ISO3[code]  # a real impl keys URIEL by the ISO-639-3 code
    return tuple(0.0 for _ in range(LANG2VEC_DIM))


if __name__ == "__main__":  # the one runnable boundary check (good passes / bad raises)
    good = l1("DE")
    assert good == "de", good
    assert len(lang2vec_features(good)) == LANG2VEC_DIM
    try:
        l1("xx")  # not in the powered subset
    except ValueError as e:
        print(f"OK: bad L1 rejected -> {e}")
    else:
        raise AssertionError("expected l1('xx') to raise")
    print(f"OK: l1('DE') = {good!r}; lang2vec dim = {LANG2VEC_DIM}")

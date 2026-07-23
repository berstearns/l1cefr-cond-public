"""Per-generation stop-reason coverage for lm-head generations.

Shape reused from gen-gec-errant (notebooks/gpt2-strategies: every generation
records a stop_reason + a truncated flag, aggregated per source): here every
generation row carries ``stop_reason`` ∈ {eos, max_new_tokens, other} and each
(l1×cefr) cell publishes the coverage, so a run whose generations are clipped at
a fixed cap is visible from the artifact instead of only from a post-hoc pass.

A generation clipped at the cap is a PREFIX, not a completed continuation; a
run of them makes any length-conditioned analysis a comparison of clipping
behaviour, so the truncation rate is a first-class published number.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

STOP_REASONS = ("eos", "max_new_tokens", "other")

# Degeneracy definitions match the m1 analytics that screened the five published
# heads (2026-07-23): a generation is degenerate if it is empty, or its
# type-token ratio is below DEGEN_TTR, or the same token repeats in a run of
# DEGEN_RUN or more. Pinned as constants so the pipeline and a later analytics
# pass agree by construction.
DEGEN_TTR = 0.5
DEGEN_RUN = 5


def degeneracy_flags(text: str) -> Dict[str, object]:
    """Per-generation degeneracy: empty, TTR, longest same-token run, verdict.
    ``degenerate`` = empty OR (has tokens AND TTR < DEGEN_TTR) OR run >= DEGEN_RUN."""
    toks = (text or "").split()
    total = len(toks)
    empty = len((text or "").strip()) == 0
    ttr = (len(set(toks)) / total) if total else 0.0
    max_run = cur = 0
    prev = None
    for t in toks:
        cur = cur + 1 if t == prev else 1
        max_run = max(max_run, cur)
        prev = t
    degenerate = empty or (total > 0 and ttr < DEGEN_TTR) or (max_run >= DEGEN_RUN)
    return {"empty": empty, "n_tokens": total, "ttr": round(ttr, 4),
            "max_rep_run": max_run, "degenerate": degenerate}


def cell_health(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate the generation rows of ONE (l1, cefr) cell: stop-reason coverage
    + truncation rate, and — when the rows carry degeneracy flags — empty count,
    the worst (min) TTR, the worst repeated-token run, and the degenerate rate."""
    n = len(rows)
    stop = Counter(str(r.get("stop_reason")) for r in rows)
    trunc = sum(1 for r in rows if r.get("stop_reason") == "max_new_tokens")
    degen = sum(1 for r in rows if r.get("degenerate"))
    out: Dict[str, object] = {
        "n": n,
        "stop_reasons": {k: stop.get(k, 0) for k in STOP_REASONS},
        "truncation_rate": (trunc / n) if n else 0.0,
        "empty": sum(1 for r in rows if r.get("empty")),
        "degenerate": degen,
        "degenerate_rate": (degen / n) if n else 0.0,
        "min_ttr": min((float(r.get("ttr", 0.0)) for r in rows), default=0.0),
        "max_rep_run": max((int(r.get("max_rep_run", 0)) for r in rows), default=0),
    }
    return out

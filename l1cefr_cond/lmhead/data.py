"""Writer-level EFCAMDAT corpus loader for LM-head conditioning.

Reads a broker-acquired corpus (``.jsonl`` or ``.csv``) whose rows carry at least
``l1``, ``cefr``, ``text`` (and, for the real split, ``writer_id`` + ``split`` so
train/test/few-shot stay author-disjoint — the notebook only ever trains the head
on the TRAIN split). Rows are mapped to the frozen 6×5 grid indices at load time;
a row whose (l1, cefr) is outside the working grid is skipped (logged by count),
never silently mis-indexed.

``nationality ≠ L1``: this loader trusts the corpus's ``l1`` column (already
mapped upstream by the Coordinator's data-prep) — it does not re-derive L1 from a
nationality field.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from l1cefr_cond.lmhead.grid import cefr_index, l1_index


@dataclass(frozen=True)
class Example:
    text: str
    l1_idx: int
    cefr_idx: int
    l1_code: str
    cefr_band: str
    writer_id: str = ""


def _rows(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"unsupported corpus format: {path.suffix!r} (want .jsonl or .csv)")


def load_corpus(
    path: Path,
    split: Optional[str] = None,
    *,
    min_chars: int = 1,
) -> List[Example]:
    """Load examples, optionally filtered to a ``split`` (train/test/fewshot).

    Out-of-grid (l1, cefr) rows are skipped; the count is printed so a silent
    drop can't masquerade as clean data.
    """
    out: List[Example] = []
    skipped_grid = 0
    skipped_empty = 0
    for r in _rows(Path(path)):
        if split is not None and str(r.get("split", "")).strip() and str(r["split"]) != split:
            continue
        text = str(r.get("text", "")).strip()
        if len(text) < min_chars:
            skipped_empty += 1
            continue
        try:
            li = l1_index(str(r["l1"]))
            ci = cefr_index(str(r["cefr"]))
        except (KeyError, ValueError):
            skipped_grid += 1
            continue
        out.append(
            Example(
                text=text, l1_idx=li, cefr_idx=ci,
                l1_code=str(r["l1"]), cefr_band=str(r["cefr"]),
                writer_id=str(r.get("writer_id", r.get("learner_id", ""))),
            )
        )
    print(f"  [data] loaded {len(out)} examples from {path}"
          + (f" (split={split})" if split else "")
          + (f"; skipped {skipped_grid} out-of-grid, {skipped_empty} empty" if (skipped_grid or skipped_empty) else ""))
    return out

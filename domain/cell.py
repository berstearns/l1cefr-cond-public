"""domain/cell.py — a ``Cell`` is the unit of the experiment grid (a value object).

``Cell(l1, cefr)`` is a frozen, slotted value object: equal by value, "changed"
via ``dataclasses.replace``. ``cell_id()`` renders the [I02] run-id fragment
``base x method x l1 x cefr x seed`` — the same token used for the run directory
(SCAFFOLD §3) and the adapter registry [I02].
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.adapter import Seed
from domain.cefr import CEFR
from domain.l1 import L1


@dataclass(frozen=True, slots=True)
class Cell:
    l1: L1
    cefr: CEFR


def cell_id(base: str, method: str, cell: Cell, seed: Seed) -> str:
    """The [I02] id fragment ``base x method x l1 x cefr x seed``.

    Mirrors ``experiments/runs/<base>__<method>__<l1>__<cefr>__seed<k>/``
    (SCAFFOLD §3); the adapter registry is generated from these tokens [I02].
    """
    return f"{base}__{method}__{cell.l1}__{cell.cefr.name}__seed{int(seed)}"


if __name__ == "__main__":  # boundary check (good builds; equal-by-value; bad L1 raises)
    from domain.adapter import seed as mk_seed
    from domain.cefr import cefr as mk_cefr
    from domain.l1 import l1 as mk_l1

    c = Cell(l1=mk_l1("de"), cefr=mk_cefr("B2"))
    assert c == Cell(l1=mk_l1("DE"), cefr=mk_cefr("b2")), "value objects compare by value"
    frag = cell_id("pythia-410m", "lora", c, mk_seed(0))
    assert frag == "pythia-410m__lora__de__B2__seed0", frag
    try:
        Cell(l1=mk_l1("english"), cefr=mk_cefr("B2"))  # bad L1 rejected at the boundary
    except ValueError as e:
        print(f"OK: bad Cell rejected upstream -> {e}")
    else:
        raise AssertionError("expected an out-of-subset L1 to raise")
    print(f"OK: cell_id -> {frag}")

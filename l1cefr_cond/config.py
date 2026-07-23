"""l1cefr_cond/config.py — parse a train/eval config.yaml into domain types.

The one boundary between untrusted YAML and the algebraic domain model
(``domain.cell.Cell``, ``domain.method.Method``). Only the ``control_token``
method is recognized so far [T01] — extend ``parse_method`` as later
techniques land (each addition is one ``match`` arm, per ``domain/method.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from domain.cefr import cefr
from domain.cell import Cell
from domain.l1 import l1
from domain.method import ControlToken, Method


def load_yaml(path: Path) -> dict[str, Any]:
    """Load + validate a config is a YAML mapping (the one file-boundary check)."""
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def parse_method(raw: dict[str, Any]) -> Method:
    kind = raw["type"]
    if kind == "control_token":
        return ControlToken(scheme=raw["scheme"])
    raise ValueError(f"unrecognized/not-yet-wired method type: {kind!r}")


def parse_cell(raw: dict[str, Any]) -> Cell:
    return Cell(l1=l1(raw["l1"]), cefr=cefr(raw["cefr"]))


def parse_train_config(path: Path) -> tuple[Cell, Method, dict[str, Any]]:
    raw = load_yaml(path)
    cell = parse_cell(raw["cell"])
    method = parse_method(raw["method"])
    return cell, method, raw

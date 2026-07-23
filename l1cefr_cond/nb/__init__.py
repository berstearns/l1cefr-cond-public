"""l1cefr_cond.nb — notebook-facing plumbing GRAFTED from gen-gec-errant v2.

Verbatim graft of the two-twin Colab infra (see specs/96-lm-head-notebook-pairs.md):
``nbenv`` (layered env reader), ``brokers`` (acquire/publish through a swappable
backend), ``colab.is_colab`` — reused unchanged so the L1×CEFR LM-head notebooks
inherit the *same* broker/preflight/env-detect substrate that already runs green
on Colab T4. Source of the graft:
``/home/b/p/research-sketches/automatic-generation-correction-errortagging-v2``.

The ONE project-specific piece is :mod:`l1cefr_cond.nb.registry` — a minimal,
torch-free registry supplying exactly the surface ``brokers`` needs (base decoders
load from the HF Hub → not brokered; only the writer-level EFCAMDAT corpus is a
broker resource). This package imports **no** torch and stays import-cheap.
"""
from __future__ import annotations

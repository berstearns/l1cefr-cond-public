"""Minimal, torch-free registry — the ONE project-specific piece of the graft.

``brokers.py`` (grafted verbatim from gen-gec-errant v2) imports four names from a
``registry`` module: ``DatasetConfig``, ``PathConfig``, ``MODEL_REGISTRY``,
``DATASET_REGISTRY``. v2's registry drags in the whole generation/pipeline/torch
stack; the LM-head notebooks need none of it — the base decoders load straight
from the HF Hub (``gdrive_subpath=None`` ⇒ "control" ⇒ not a broker resource), so
the ONLY brokered resource is the writer-level EFCAMDAT corpus. This module
re-implements just that surface, torch-free, so importing the broker layer is cheap.

Frozen grid context (specs/96, CLAUDE.md): the 11-base suite with the smallest of
each family ENABLED to start; conditioning is over 6 L1 × 5 CEFR. Models here are
the *native* (frozen) backbones — the LM-head module is what gets trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ── Dataset config ─────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """A single corpus resource (a brokered ``corpora/<filename>`` resource)."""
    name: str
    filename: str
    description: str
    text_column: str = "text"


# ── Model config (torch-free; the fields brokers.py + the notebook read) ─
@dataclass
class ModelConfig:
    """A decoder backbone. ``gdrive_subpath=None`` ⇒ loads from the HF Hub by id
    (a "control" in broker terms — nothing to fetch). ``enabled`` gates the
    smallest-per-family start set (specs/96 CONFIG cell)."""
    name: str
    hf_model_id: str
    model_family: str
    params: str
    # F22 repro-provenance pin: the explicit HF-Hub commit SHA the backbone loads
    # at (``from_pretrained(..., revision=…)``). ``None`` ⇒ unpinned ``main`` (only
    # the not-yet-enabled suite members). Resolved by scripts/pin_backbone_revisions.py.
    revision: Optional[str] = None
    gdrive_subpath: Optional[str] = None
    checkpoint_subdir: str = ""
    enabled: bool = False
    description: str = ""


# ── Path config ────────────────────────────────────────────────────────
@dataclass
class PathConfig:
    """Root paths for data, models, and output. ``for_colab``/``for_local`` mirror
    v2 so the grafted brokers resolve identically on either runtime."""
    data_root: Path
    models_root: Path
    output_root: Path

    @classmethod
    def for_colab(cls) -> "PathConfig":
        gdrive = Path("/content/drive/MyDrive")
        return cls(
            data_root=gdrive / "l1cefr-cond/data",
            models_root=gdrive / "_p/artificial-learners/models",
            output_root=gdrive / "l1cefr-cond-lm-head-outputs",
        )

    @classmethod
    def for_local(cls) -> "PathConfig":
        return cls(
            data_root=Path("data"),
            models_root=Path("models"),
            output_root=Path("outputs/lm-head"),
        )

    def dataset_path(self, dataset: DatasetConfig) -> Path:
        return self.data_root / dataset.filename

    def model_gdrive_path(self, model: ModelConfig) -> Optional[Path]:
        if model.gdrive_subpath is None:
            return None
        p = self.models_root / model.gdrive_subpath
        return p / model.checkpoint_subdir if model.checkpoint_subdir else p


# ── Model registry — the 11-base suite (native frozen backbones) ────────
# hf ids verified present in the local HF cache for the 3 ENABLED bases
# (gpt2, EleutherAI/pythia-70m [untied head], HuggingFaceTB/SmolLM2-135M).
def _m(name: str, hf: str, fam: str, params: str, enabled: bool = False,
       revision: Optional[str] = None) -> ModelConfig:
    return ModelConfig(name=name, hf_model_id=hf, model_family=fam, params=params,
                       revision=revision, enabled=enabled,
                       description=f"{fam} {params} native (frozen backbone)")


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # F22 pin: 3 ENABLED backbones fixed to explicit HF-Hub commit SHAs (resolved
    # 2026-07-18 by scripts/pin_backbone_revisions.py). Re-run it --check to detect drift.
    "gpt2-small":   _m("gpt2-small",   "gpt2",                     "gpt2",    "124M", enabled=True,
                       revision="607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "pythia-70m":   _m("pythia-70m",   "EleutherAI/pythia-70m",    "pythia",  "70M",  enabled=True,
                       revision="a39f36b100fe8a5377810d56c3f4789b9c53ac42"),
    "smollm2-135m": _m("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", "smollm2", "135M", enabled=True,
                       revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    # ── rest of the suite (commented-equivalent: enabled=False) ──
    "gpt2-medium":  _m("gpt2-medium",  "gpt2-medium",              "gpt2",    "355M"),
    "gpt2-large":   _m("gpt2-large",   "gpt2-large",               "gpt2",    "774M"),
    "pythia-160m":  _m("pythia-160m",  "EleutherAI/pythia-160m",   "pythia",  "160M"),
    "pythia-410m":  _m("pythia-410m",  "EleutherAI/pythia-410m",   "pythia",  "410M"),  # headline (untied)
    "pythia-1b":    _m("pythia-1b",    "EleutherAI/pythia-1b",     "pythia",  "1B"),
    "pythia-1.4b":  _m("pythia-1.4b",  "EleutherAI/pythia-1.4b",   "pythia",  "1.4B"),
    "smollm2-360m": _m("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", "smollm2", "360M"),
    "smollm2-1.7b": _m("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B", "smollm2", "1.7B"),
    # ultra-tiny tied head — smoke stand-in only (2 hidden dims), never a real cell
    "tiny-gpt2":    _m("tiny-gpt2",    "sshleifer/tiny-gpt2",      "gpt2",    "toy"),
}


def enabled_models() -> List[str]:
    """The smallest-per-family start set (specs/96)."""
    return [k for k, m in MODEL_REGISTRY.items() if m.enabled]


# ── Dataset registry — the brokered corpora ─────────────────────────────
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    # the real WRITER-LEVEL author-disjoint split (nationality ≠ L1); produced by
    # the Coordinator's data-prep, may not exist locally yet — the notebook reads
    # whatever the broker hands it in this schema.
    "efcamdat-writer-split": DatasetConfig(
        name="efcamdat-writer-split",
        filename="splits/efcamdat-writer-split.jsonl",
        description="EFCAMDAT writer-level author-disjoint split (learner_id, l1, cefr, text)",
    ),
    # ⚠️ REAL DATA, RANDOM SPLIT AT WRITING LEVEL — **NOT** writer-disjoint.
    # Built by scripts/build_efcamdat_corpus.py from norm-EFCAMDAT-ALL-CONCAT.csv,
    # which carries NO learner/writer id column (only writing_id, l1, cefr_level,
    # text), so an author-disjoint split is IMPOSSIBLE from this source. The same
    # learner's texts may therefore land in both train and validation.
    #
    # This VIOLATES CLAUDE.md non-negotiable #1 (split by WRITER, not sentence).
    # The user accepted it EXPLICITLY and KNOWINGLY as an interim corpus. It is a
    # training/plumbing corpus, NOT a specs/96 science result: L1-recoverability
    # measured on it may reflect writer memorisation rather than L1 features.
    #
    # The key is deliberately named for what it IS. 'efcamdat-writer-split' stays
    # RESERVED for a real author-disjoint split and must never be pointed here —
    # a run that says "writer-split" while reading these bytes would launder the
    # leakage into the record, which is the one failure nobody would catch later.
    "efcamdat-random-writinglevel": DatasetConfig(
        name="efcamdat-random-writinglevel",
        filename="efcamdat-random-writinglevel/efcamdat_random_writinglevel.jsonl",
        description=(
            "EFCAMDAT 6 L1 × 5 CEFR (A1..C1), RANDOM writing-level split — NOT writer-disjoint "
            "(source has no learner id); train+validation in one file via the 'split' column"
        ),
    ),
    # the committed tiny fixture (scripts/build_smoke_fixture.py) — writer-disjoint
    # de/B1 + fr/A2, for the S0 headless smoke. Fixture data, not a corpus.
    "efcamdat-smoke": DatasetConfig(
        name="efcamdat-smoke",
        filename="smoke/efcamdat_smoke.jsonl",
        description="Tiny writer-disjoint EFCAMDAT-shaped smoke fixture (S0)",
    ),
}

# Which columns a corpus row must carry for the LM-head data loader.
CORPUS_SCHEMA: List[str] = ["l1", "cefr", "text"]

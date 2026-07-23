"""l1cefr-eval — the E01-E06 evaluation binary (STUB).

Contract (experiment-binaries + SCAFFOLD §3), NOT yet implemented:
  - load the parent train run + checkpoint (the eval row carries
    ``parent_run_id``);
  - generate ``preds.jsonl``; submit to the Coordinator-hosted CEFR / NLI /
    ERRANT scorers (held-out instances — the eval-reward firewall);
  - write ``eval.json`` + one ``metric`` row per E02/E03/E04/E05/E06 score
    (tidy/long; ``split`` = ``<l1>`` or ``<cefr>``).

The CLI + config-parse below IS wired (S0-SETUP-ONLY item 5): it proves an
eval ``config.yaml`` parses. It stops there — no scorer is called, no
prediction is generated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from l1cefr_cond.config import load_yaml

_REQUIRED_KEYS = ("run_dir", "dataset", "scorers")


def main() -> int:
    """Entry point for ``l1cefr-eval`` (parses config; eval itself is a stub)."""
    parser = argparse.ArgumentParser(
        prog="l1cefr-eval",
        description="Evaluate a trained cell against the held-out scorers (STUB — evaluation not implemented).",
    )
    parser.add_argument("config", type=Path, help="path to an eval config.yaml")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"{args.config}: missing required key(s): {missing}")
    print(f"CONFIG OK: {args.config} -> run_dir={raw['run_dir']} scorers={sorted(raw['scorers'])}")
    print("l1cefr-eval: STUB — evaluation not implemented.")
    print("TODO[E02/E03/E04]: preds.jsonl -> held-out scorers -> eval.json + metric rows.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

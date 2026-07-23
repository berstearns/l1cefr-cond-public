"""l1cefr-train — the adapter/PEFT training binary.

Contract (experiment-binaries + SCAFFOLD §3):
  - one ``config.yaml`` drives the run; flags only override;
  - refuse to start on a dirty git tree; set all seeds;
  - write ``config.yaml`` + ``adapter/`` + ``run.log`` (completion sentinel +
    ``$?``) + ``run-info.json`` into the run directory;
  - idempotent — refuses to overwrite a green (``DONE rc=0``) run.

Only the ``control_token`` method [T01] is implemented: a real forward +
backward + optimizer step on the full backbone (T01 is continual pretraining,
not PEFT — there is no adapter to freeze-around; ``n_trainable_params ==
n_params`` by design, logged explicitly in ``run-info.json`` so that isn't
mistaken for a forgotten freeze). Every other ``Method`` variant is still a
stub: ``domain/method.py`` already makes illegal method configs
unconstructible, so ``config.py`` only ever hands this file a
``ControlToken`` today — the ``match`` below is future-proofing, not dead code.

PEFT weight-load parity (``huggingface-weight-loading``) does not apply here:
this loads a stock HF checkpoint via ``AutoModelForCausalLM.from_pretrained``
directly, it does not port weights into a hand-rolled module.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from domain.cell import Cell
from domain.method import ControlToken, describe
from l1cefr_cond.config import parse_train_config

REPO_ROOT = Path(__file__).resolve().parents[1]

Logger = Callable[[str], None]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_is_dirty() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return bool(out.strip())


def _load_smoke_rows(path: Path, l1_code: str, cefr_band: str, split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["l1"] == l1_code and row["cefr"] == cefr_band and row["split"] == split:
                rows.append(row)
    return rows


def run_control_token_step(cell: Cell, raw: dict[str, Any], log: Logger) -> dict[str, Any]:
    """One real forward + backward + optimizer step [T01], CPU-only."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_val = int(raw.get("seed", 0))
    torch.manual_seed(seed_val)

    base_model = raw["base_model"]
    revision = raw.get("model_revision")
    if not revision:
        raise ValueError("config must pin model_revision (never 'main'/'latest') — repro-provenance canon")

    log(f"loading {base_model}@{revision} (CPU)")
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision)

    control_tokens = list(raw.get("tokens", []))
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": control_tokens})
    if n_added:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"registered {n_added} control token(s): {control_tokens}; vocab now {len(tokenizer)}")

    fixture_path = REPO_ROOT / raw["dataset"]["path"]
    split = raw["dataset"]["split"]
    rows = _load_smoke_rows(fixture_path, str(cell.l1), cell.cefr.name, split)
    if not rows:
        raise ValueError(f"no fixture rows for cell {cell.l1}/{cell.cefr.name} split={split!r} in {fixture_path}")

    l1_tok, cefr_tok = f"<l1={cell.l1}>", f"<cefr={cell.cefr.name.lower()}>"
    texts = [f"{l1_tok}{cefr_tok} {row['text']}" for row in rows]
    log(f"{len(texts)} training example(s) for cell {cell.l1}/{cell.cefr.name}")

    max_len = int(raw.get("sequence_len", 64))
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100

    model.train()
    lr = float(raw.get("learning_rate", 5e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    with torch.no_grad():
        norm_before = sum(p.detach().float().norm().item() for p in model.parameters())

    optimizer.zero_grad()
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
    loss = out.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss on the smoke step: {loss.item()!r}")
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        norm_after = sum(p.detach().float().norm().item() for p in model.parameters())
    if norm_after == norm_before:
        raise RuntimeError("param norm unchanged after backward()+step() — the optimizer step was a no-op")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"loss={loss.item():.6f} param_norm {norm_before:.6f} -> {norm_after:.6f} "
        f"(delta={norm_after - norm_before:.6e}) trainable={n_trainable}/{n_params}")

    return {
        "loss": loss.item(),
        "n_examples": len(rows),
        "n_params": n_params,
        "n_trainable_params": n_trainable,
        "param_norm_before": norm_before,
        "param_norm_after": norm_after,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_revision": revision,
        "seed": seed_val,
        "model": model,
        "tokenizer": tokenizer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="l1cefr-train",
        description="Train a conditioning adapter for one (L1, CEFR) cell.",
    )
    parser.add_argument("config", type=Path, help="path to a train config.yaml")
    args = parser.parse_args()

    cell, method, raw = parse_train_config(args.config)

    match method:
        case ControlToken():
            pass
        case _:
            print(f"l1cefr-train: STUB — {describe(method)} not implemented yet.")
            return 1

    if _git_is_dirty():
        print("l1cefr-train: REFUSING to start — git tree is dirty (experiment-binaries canon).",
              file=sys.stderr)
        return 1

    output_dir = REPO_ROOT / raw["output_dir"]
    existing_log = output_dir / "run.log"
    if existing_log.exists() and "DONE rc=0" in existing_log.read_text():
        print(f"l1cefr-train: REFUSING to overwrite — {output_dir} already holds a green run.",
              file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(args.config.read_text())

    rc = 0
    with existing_log.open("w") as logf:
        def log(msg: str) -> None:
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

        try:
            log(f"l1cefr-train: cell={cell.l1}/{cell.cefr.name} method={describe(method)}")
            result = run_control_token_step(cell, raw, log)

            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            result["model"].save_pretrained(adapter_dir)
            result["tokenizer"].save_pretrained(adapter_dir)
            (adapter_dir / "POINTER.md").write_text(
                "T01 control-token method: a full-backbone continual-pretrain "
                "checkpoint, not a LoRA/PEFT adapter — there is no low-rank "
                "delta to point at, so the full (tiny) model is saved here.\n"
            )
            log(f"wrote adapter -> {adapter_dir}")

            run_info = {
                "git_sha": _git_sha(),
                "git_dirty": False,
                "cell": {"l1": str(cell.l1), "cefr": cell.cefr.name},
                "method": describe(method),
                "model": raw["base_model"],
                "model_revision": result["model_revision"],
                "seed": result["seed"],
                "torch_version": result["torch_version"],
                "transformers_version": result["transformers_version"],
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "device": "cpu",
                "n_params": result["n_params"],
                "n_trainable_params": result["n_trainable_params"],
                "peft": False,
                "peft_note": "T01 trains the full backbone; n_trainable_params == n_params by design.",
                "n_examples": result["n_examples"],
                "loss": result["loss"],
                "param_norm_before": result["param_norm_before"],
                "param_norm_after": result["param_norm_after"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (output_dir / "run-info.json").write_text(json.dumps(run_info, indent=2) + "\n")
            log(f"wrote run-info -> {output_dir / 'run-info.json'}")
            log("DONE rc=0")
            rc = 0
        except Exception as e:  # keep the sentinel honest on any failure — never a mid-run snapshot
            log(f"ERROR: {e!r}")
            log("DONE rc=1")
            rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

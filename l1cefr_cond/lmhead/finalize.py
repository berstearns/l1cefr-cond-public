"""Terminal completion sentinel for lm-head runs.

A published run dir holding preds/ + run-info.json but no exit status is
formally indistinguishable from a mid-run snapshot: run-info.json is written
per model as that model finishes, so a Colab disconnect after model 1 of 3
leaves a dir that looks exactly like a deliberate 1-model run. The sentinel
pair written here (run.log ending in ``DONE rc=<n>`` + DONE.json) is written
only after the publish cell's all-fail assert has passed, and is published
LAST so its mtime postdates every data file on the remote: a dir without it
is a snapshot, a dir with it carries its own exit status.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _run_infos(run_out: Path) -> list[dict]:
    """All run-info.json files under run_out (top level, then one per model dir)."""
    infos = []
    for p in [run_out / "run-info.json", *sorted(run_out.glob("*/run-info.json"))]:
        if p.is_file():
            try:
                infos.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
    return infos


def finalize_run(run_out, dest, out_broker, summaries, failures, manifest) -> int:
    """Write run.log (last line ``DONE rc=<rc>``) + DONE.json into run_out and
    publish them so the sentinel is the newest object in the remote dir.

    rc=0: every model succeeded; rc=2: partial (>=1 ok AND >=1 failed);
    rc=1: a "successful" model dir lacks checkpoints/head.safetensors or
    checkpoints/trainer_state.json — green-without-weights is refused.
    The all-fail case never reaches here — the publish cell asserts
    len(summaries) >= 1 first, so a zero-success run raises instead of
    minting a sentinel.
    """
    run_out = Path(run_out)
    rc = 0 if not failures else 2
    models_ok = [s.get("model_key") for s in summaries]
    models_failed = [f.get("model_key") for f in failures]

    # ── weights-on-disk gate: a "successful" model whose dir lacks the trained
    # head + loss curve is exactly the discarded-head failure this sentinel must
    # make impossible — refuse to mint rc=0 (or even the partial rc=2) for it.
    missing = [
        s.get("model_key") for s in summaries
        if not (run_out / str(s.get("model_key")) / "checkpoints" / "head.safetensors").is_file()
        or not (run_out / str(s.get("model_key")) / "checkpoints" / "trainer_state.json").is_file()
    ]
    if missing:
        rc = 1

    # ── remote readback (the one-layer-deeper false green): the local files
    # existing says nothing about the PUBLISHED copy — a partial rclone upload
    # would leave Drive with some files but not others while this sentinel goes
    # green. Verify EVERYTHING the run produced, not just the two checkpoint
    # files: publish() (notebook cell 20) pushed the whole tree — the 30
    # preds/<l1>__<cefr>.jsonl, run-info.json, figures/progress.png, and every
    # checkpoints/<tag>/ — and the old hardcoded 2-element list left ~32
    # artifacts UNDECIDED, so the sentinel could not actually attest "all
    # training artifacts on gdrive". A mismatch/absence RAISES (no sentinel is
    # minted at all — unfinalized, correctly).
    #
    # run.log + DONE.json are written LATER in this function (below) and the
    # final out_broker.publish re-pushes them so they postdate the data on the
    # remote; they do not exist on disk yet at this line, so rglob cannot pick
    # them up — verified against the write order (readback here, sentinel writes
    # after). That is intended: the sentinel must not verify itself.
    if not missing:
        relpaths = sorted(str(p.relative_to(run_out))
                          for p in run_out.rglob("*") if p.is_file())
        out_broker.verify_published(run_out, dest, relpaths)

    infos = _run_infos(run_out)
    git_sha = next((i.get("git_sha") for i in infos if i.get("git_sha")), "unknown")
    rows_consumed = sum(int(i.get("rows_consumed") or 0) for i in infos)
    n_cells = next((int(i["n_cells"]) for i in infos if i.get("n_cells")), 0)
    n_preds = sum(int(s.get("n_preds") or 0) for s in summaries)
    checkpoints = {
        str(i.get("model_key")): {
            "checkpoint_sha256": (i.get("checkpoint") or {}).get("sha256"),
            "final_loss": (i.get("loss_curve") or {}).get("final_loss"),
        }
        for i in infos if i.get("model_key")
    }
    ts = datetime.now(timezone.utc).isoformat()

    (run_out / "run.log").write_text(
        f"PROVENANCE ts={ts} git_sha={git_sha} models_ok={models_ok} "
        f"models_failed={models_failed} rows_consumed={rows_consumed} "
        f"n_cells={n_cells} n_preds={n_preds} "
        f"checkpoints={checkpoints} missing_checkpoints={missing}\n"
        f"DONE rc={rc}\n"
    )
    (run_out / "DONE.json").write_text(json.dumps({
        "status": ("ok" if rc == 0 else "partial" if rc == 2 else "missing-checkpoint"),
        "missing_checkpoints": missing,
        "checkpoints": checkpoints,
        "rc": rc,
        "generated_at": ts,
        "models_ok": models_ok,
        "models_failed": models_failed,
        "n_cells": n_cells,
        "n_preds": n_preds,
        "git_sha": git_sha,
    }, indent=2) + "\n")

    # Re-publish AFTER the data publish: an idempotent copy whose only new bytes
    # are run.log + DONE.json, so the sentinel's remote mtime postdates the data.
    # If this publish dies, the remote dir has data but no sentinel — which reads,
    # correctly, as "not finalized".
    out_broker.publish(run_out, dest)
    return rc

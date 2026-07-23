"""Periodic checkpoint upload/pull for lm-head runs (env-driven, rclone).

Why this exists: the end-of-run publish only helps a run that ENDS. A Colab
preemption at step 4,000 of 4,898 used to lose everything; with each periodic
checkpoint uploaded (and readback-verified) as it is written, a disconnect
costs at most one interval, and a fresh VM pulls the newest state back before
training (see runner.run_grid).

Configuration comes from the same baked env the notebooks already carry —
GGE_OUTPUT_BROKER / GGE_OUTPUT_RCLONE_DEST / GGE_RCLONE_CONF_PATH — so a local
smoke run (no rclone env) silently gets no uploader and stays offline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

KNOWN_STRATEGIES = {"additive-bias-mlp", "bilinear-interaction", "film",
                    "gated-suppression", "lora-delta"}


def strategy_dest(base: str, strategy: str) -> str:
    """Key the DEST on the strategy axis — same normalization as the notebook
    publish cell (hole g: the resume-pull must read the folder the run actually
    publishes to): strip a baked strategy tail, right or wrong, then append
    this run's strategy."""
    b = base.rstrip("/")
    if b.rsplit("/", 1)[-1] in KNOWN_STRATEGIES:
        b = b.rsplit("/", 1)[0]
    return f"{b}/{strategy}"


def resolve_ckpt_dest(strategy: str) -> Optional[str]:
    """The strategy-keyed remote DEST, or None when this run has no rclone output
    env (local smoke / gdrive-mount runs upload nothing mid-run)."""
    if os.environ.get("GGE_OUTPUT_BROKER") != "rclone":
        return None
    base = os.environ.get("GGE_OUTPUT_RCLONE_DEST")
    return strategy_dest(base, strategy) if base else None


def _conf() -> Optional[str]:
    p = os.environ.get("GGE_RCLONE_CONF_PATH")
    return p if p and Path(p).is_file() else None


def make_ckpt_uploader() -> Optional[Callable[[Path, str], bool]]:
    """An ``upload(local, remote_dir) -> bool`` that rclone-copies a checkpoint
    file/dir and READBACK-verifies it landed with the local byte counts (hole c:
    rclone's exit code is not proof). Returns None when the env carries no
    rclone output config — callers treat that as "no periodic upload"."""
    conf = _conf()
    if os.environ.get("GGE_OUTPUT_BROKER") != "rclone" or not conf or not shutil.which("rclone"):
        return None

    def upload(local: Path, remote_dir: str) -> bool:
        local = Path(local)
        files = [local] if local.is_file() else sorted(p for p in local.rglob("*") if p.is_file())
        if not files:
            print(f"  [ckpt-upload] nothing to upload at {local}")
            return False
        cmd = ["rclone", "copy", str(local), remote_dir, "--config", conf, "--stats=0"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  ⚠️  [ckpt-upload] rclone copy failed rc={r.returncode}: "
                  f"{(r.stderr or '').strip()[-300:]}")
            return False
        ls = subprocess.run(["rclone", "lsl", remote_dir, "--config", conf],
                            capture_output=True, text=True, timeout=120)
        remote_bytes = {}
        for ln in ls.stdout.splitlines():
            parts = ln.strip().split(None, 3)
            if len(parts) == 4:
                remote_bytes[parts[3]] = int(parts[0])
        for f in files:
            rel = f.name if local.is_file() else str(f.relative_to(local))
            want = f.stat().st_size
            if want <= 0 or remote_bytes.get(rel, -1) != want:
                print(f"  ⚠️  [ckpt-upload] readback MISMATCH {rel}: "
                      f"local={want}B remote={remote_bytes.get(rel, 'absent')}")
                return False
        print(f"  [ckpt-upload] ✅ {len(files)} file(s) verified on {remote_dir}")
        return True

    return upload


def pull_run_checkpoints(dest: str, model_key: str, ck_root: Path) -> bool:
    """Fetch the run's remote checkpoints (strategy-keyed DEST) into the local
    checkpoints dir so a fresh VM resumes instead of restarting (holes f+g).
    Best-effort: an empty/absent remote dir is a first run, not an error."""
    conf = _conf()
    if not conf or not shutil.which("rclone"):
        return False
    remote = f"{dest}/{model_key}/checkpoints"
    ck_root.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rclone", "copy", remote, str(ck_root), "--config", conf, "--stats=0"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  [ckpt-pull] no remote checkpoints at {remote} (first run, or: "
              f"{(r.stderr or '').strip()[-200:]})")
        return False
    n = sum(1 for _ in ck_root.rglob("head-state.pt"))
    print(f"  [ckpt-pull] pulled {remote} -> {ck_root} ({n} resume state(s))")
    return n > 0

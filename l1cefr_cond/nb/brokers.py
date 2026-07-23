"""Acquisition brokers — one interface, N interchangeable backends.

GRAFTED VERBATIM from gen-gec-errant v2 (src/gen_gec_errant/brokers.py) per
specs/96-lm-head-notebook-pairs.md. The ONLY change from the source is the
registry import path (``gen_gec_errant.registry`` → ``l1cefr_cond.nb.registry``);
all broker logic, verification, idempotency and the backend set are unchanged so
the LM-head notebooks inherit the identical, Colab-green acquisition substrate.

Why this module exists
----------------------
The first Colab run died on arrival because acquisition was hardcoded as
``git clone <private repo>`` — a source the runtime could not reach (2026-07-15).
Any single hardcoded source (a mount path, a private repo, one remote) eventually
meets a runtime that lacks it. A *broker* resolves a **logical resource name**
(``"checkpoints/ft-gpt2-small"``, ``"corpora/norm-CELVA-SP.csv"``) to a local
path via a swappable backend, chosen in the notebook CONFIG cell — and it
**verifies** what it fetched before returning it (fetch-without-verify is how a
silent fallback ships). See
``marimos/auto-rules/notebooks-development/acquire-data-and-models-through-a-broker-never-hardcode-one-source.md``.

Design notes
------------
* **No caller ever holds a URL.** Callers pass logical names; the manifest +
  the chosen broker own the addressing. Adding a backend touches no notebook cell.
* **No duplicated path literals.** Addresses are derived from the same
  ``PathConfig`` + registry the batch pipeline uses (``model_gdrive_path`` /
  ``dataset_path``), so the manifest can never drift from the run config.
* **The batch pipeline does NOT import this module** — it is notebook-facing
  plumbing only. It imports the registry (which transitively pulls the config
  stack, incl. torch); the *optional* backends — ``huggingface_hub`` (HfBroker),
  the rclone binary, and ``curl``/``tar`` (fserve) — are imported/invoked lazily
  inside their backend so the manifest + verification logic needs none of them.
* **Credentials are themselves acquired resources** (rclone.conf, fserve
  passcode) — delivered via a secure channel (fserve URL / Colab ``userdata``),
  never committed or inlined. This module only *consumes* a conf path / passcode.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

from l1cefr_cond.nb.registry import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    DatasetConfig,
    PathConfig,
)

def _sha256(path: Path) -> str:
    """Streaming sha256 of a file (chunked so a 39 MB resume dict does not load
    whole). Used by verify_published to compare CONTENT, not just byte counts —
    size(a) == size(b) does not imply content(a) == content(b)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
GDRIVE_PREFIX = "/content/drive/MyDrive/"

# Canonical Colab/Drive layout — the single source of the drive-relative subpaths
# used by the remote backends (rclone/fserve). Local backends use their own roots.
_DRIVE_PATHS = PathConfig.for_colab()


# ── Resource identity + verification ────────────────────────────────────

@dataclass(frozen=True)
class ResourceSpec:
    """A logical resource and the registry entity that defines its location.

    ``kind`` drives verification; ``registry_key`` lets every backend resolve the
    concrete path from a ``PathConfig`` (no path literal is stored here, so the
    manifest cannot drift from the run's ``PathConfig``/registry).
    """
    name: str            # logical name, e.g. "checkpoints/ft-gpt2-small"
    kind: str            # "checkpoint" | "corpus"
    registry_key: str    # MODEL_REGISTRY key (checkpoint) or DATASET_REGISTRY key (corpus)
    hf_model_id: str = ""  # optional HF repo id serving the same resource (HfBroker / controls)


def verify(spec: ResourceSpec, path: Path) -> bool:
    """Structural verification keyed on the resource kind.

    checkpoint ⇒ a directory with ``config.json`` + a weights file;
    corpus     ⇒ a non-empty file whose basename matches the expected filename.
    """
    if spec.kind == "checkpoint":
        return (
            path.is_dir()
            and (path / "config.json").is_file()
            and any((path / w).is_file() for w in WEIGHT_FILES)
        )
    if spec.kind == "corpus":
        return path.is_file() and path.stat().st_size > 0
    raise ValueError(f"unknown resource kind: {spec.kind!r}")


# ── Manifest ────────────────────────────────────────────────────────────

def _corpus_name(dataset_key: str) -> str:
    return f"corpora/{DATASET_REGISTRY[dataset_key].filename}"


def _checkpoint_name(model_key: str) -> str:
    return f"checkpoints/{model_key}"


def build_manifest(
    model_keys: Optional[list[str]] = None,
    dataset_keys: Optional[list[str]] = None,
) -> Dict[str, ResourceSpec]:
    """Build the logical-name → ResourceSpec manifest from the registry.

    Only artificial-learner checkpoints (``gdrive_subpath`` set) become
    ``checkpoints/*`` resources — matched controls load from the HF Hub by id and
    need no broker. Every dataset becomes a ``corpora/<filename>`` resource.
    """
    if model_keys is None:
        model_keys = [k for k, m in MODEL_REGISTRY.items() if m.gdrive_subpath]
    if dataset_keys is None:
        dataset_keys = list(DATASET_REGISTRY)

    manifest: Dict[str, ResourceSpec] = {}
    for k in model_keys:
        m = MODEL_REGISTRY[k]
        if m.gdrive_subpath is None:
            continue  # control — not a broker resource
        manifest[_checkpoint_name(k)] = ResourceSpec(
            name=_checkpoint_name(k), kind="checkpoint", registry_key=k,
            hf_model_id=m.hf_model_id or "",
        )
    for dk in dataset_keys:
        manifest[_corpus_name(dk)] = ResourceSpec(
            name=_corpus_name(dk), kind="corpus", registry_key=dk,
        )
    return manifest


# ── Path resolution helpers (shared by backends) ────────────────────────

def _abs_path(spec: ResourceSpec, paths: PathConfig) -> Path:
    """Resolve a spec to an absolute path under the given PathConfig's roots."""
    if spec.kind == "checkpoint":
        p = paths.model_gdrive_path(MODEL_REGISTRY[spec.registry_key])
        if p is None:
            raise ValueError(f"{spec.name}: model has no gdrive_subpath (control?)")
        return Path(p)
    return Path(paths.dataset_path(DATASET_REGISTRY[spec.registry_key]))


def _drive_relpath(spec: ResourceSpec) -> str:
    """Path of the resource RELATIVE to ``/content/drive/MyDrive/`` (rclone/fserve)."""
    s = str(_abs_path(spec, _DRIVE_PATHS))
    if not s.startswith(GDRIVE_PREFIX):
        raise ValueError(f"{spec.name}: {s} is not under {GDRIVE_PREFIX}")
    return s[len(GDRIVE_PREFIX):]


# ── Broker protocol ─────────────────────────────────────────────────────

@runtime_checkable
class Broker(Protocol):
    """One interface, N backends. ``acquire`` is idempotent + verifying."""
    name: str

    def acquire(self, resource: str, dest: Path) -> Path:  # pragma: no cover - protocol
        """Return a local path holding ``resource``, fetching it if absent.

        Idempotent: a verified-present destination is returned without refetching.
        Raises RuntimeError if the fetched artifact fails verification.
        """
        ...

    def publish(self, local_dir: Path, dest: str) -> str:  # pragma: no cover - protocol
        """Push a finished results directory to a durable sink; return where it landed.

        Symmetric to ``acquire``: the OUTPUT side of the broker. ``dest`` is a
        filesystem path (local/gdrive) or an ``rclone`` target (``remote:path``).
        """
        ...


class _BaseBroker:
    """Shared idempotency + verification wrapper. Subclasses implement ``_fetch``."""
    name = "base"

    def __init__(self, manifest: Dict[str, ResourceSpec]):
        self.manifest = manifest

    def _spec(self, resource: str) -> ResourceSpec:
        try:
            return self.manifest[resource]
        except KeyError:
            raise KeyError(
                f"{resource!r} not in manifest (have: {sorted(self.manifest)[:6]}…)"
            ) from None

    def _dest_for(self, spec: ResourceSpec, dest: Path) -> Path:
        """A checkpoint lands as a directory ``dest``; a corpus as ``dest/<filename>``
        (or ``dest`` if it already names a file)."""
        if spec.kind == "corpus" and (dest.is_dir() or dest.suffix == ""):
            return dest / Path(spec.name).name
        return dest

    def acquire(self, resource: str, dest: Path) -> Path:
        spec = self._spec(resource)
        target = self._dest_for(spec, Path(dest))
        if verify(spec, target):
            print(f"  [{self.name}] {resource}: already present ✅ {target}")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [{self.name}] {resource}: fetching → {target}")
        self._fetch(spec, target)
        if not verify(spec, target):
            raise RuntimeError(
                f"[{self.name}] {resource}: fetched to {target} but verification FAILED "
                f"(kind={spec.kind}) — refusing to return an unverified artifact"
            )
        print(f"  [{self.name}] {resource}: verified ✅ {target}")
        return target

    def _fetch(self, spec: ResourceSpec, target: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def publish(self, local_dir: Path, dest: str) -> str:
        """Default sink = a filesystem copy (covers ``local`` + mounted ``gdrive``).
        Copies ``local_dir`` → ``dest`` (no-op when they are the same path); verifies
        the destination is non-empty. rclone/fserve override this."""
        src = Path(local_dir)
        if not src.is_dir():
            raise RuntimeError(f"[{self.name}] publish: results dir not found: {src}")
        d = Path(dest)
        if d.resolve() != src.resolve():
            d.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, d, dirs_exist_ok=True)
        if not any(d.iterdir()):
            raise RuntimeError(f"[{self.name}] publish: sink {d} is empty after copy")
        print(f"  [{self.name}] published results → {d}")
        return str(d)

    def verify_published(self, local_dir: Path, dest: str, relpaths: List[str]) -> None:
        """Assert each ``relpath`` under ``local_dir`` also exists at ``dest`` with
        the SAME CONTENT (sha256), not merely the same size — a partial or
        truncated copy can match on bytes-that-happen-to-equal while differing in
        content. publish() returning is not proof the bytes landed; this makes the
        sentinel layer refuse a sink that only looks populated. Filesystem sink
        (local/gdrive mount): both sides are on disk, so hash both. rclone
        overrides for the remote case. Any mismatch/absence RAISES."""
        for rel in relpaths:
            src, dst = Path(local_dir) / rel, Path(dest) / rel
            if not src.is_file() or src.stat().st_size <= 0:
                raise RuntimeError(
                    f"[{self.name}] verify_published: local {rel} missing/empty — nothing to attest")
            if not dst.is_file():
                raise RuntimeError(
                    f"[{self.name}] verify_published: {rel} ABSENT at {dest} — "
                    f"the published sink does NOT carry this artifact")
            if _sha256(src) != _sha256(dst):
                raise RuntimeError(
                    f"[{self.name}] verify_published: {rel} CONTENT mismatch at {dest} "
                    f"(sha256 differs) — refusing green")
        print(f"  [{self.name}] verify_published: {len(relpaths)} artifact(s) "
              f"content-verified (sha256) at {dest}")


# ── Backends ─────────────────────────────────────────────────────────────

class LocalBroker(_BaseBroker):
    """Resources already on the local filesystem (laptop runs). No copy — verify
    in place and return the registry-resolved path (``dest`` is ignored)."""
    name = "local"

    def __init__(self, manifest: Dict[str, ResourceSpec], paths: PathConfig):
        super().__init__(manifest)
        self.paths = paths

    def acquire(self, resource: str, dest: Path) -> Path:
        spec = self._spec(resource)
        src = _abs_path(spec, self.paths)
        if not verify(spec, src):
            raise RuntimeError(f"[local] {resource}: not present/valid at {src}")
        print(f"  [local] {resource}: verified in place ✅ {src}")
        return src


class GdriveBroker(_BaseBroker):
    """Mounted Google Drive (Colab) or a local Drive mirror. Copies the resource
    from the mount to ``dest`` on fast local disk (parity with the old
    ``resolve_model_path`` SSD copy), then verifies."""
    name = "gdrive"

    def __init__(self, manifest: Dict[str, ResourceSpec], paths: PathConfig):
        super().__init__(manifest)
        self.paths = paths

    def _fetch(self, spec: ResourceSpec, target: Path) -> None:
        src = _abs_path(spec, self.paths)
        if not src.exists():
            raise RuntimeError(f"[gdrive] {spec.name}: not on the mount at {src}")
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)


class RcloneBroker(_BaseBroker):
    """Any rclone remote that mirrors the Drive (default ``i:``). Ensures the
    rclone binary is present (subprocess install on Colab), uses a delivered
    ``rclone.conf`` (never committed), and ``rclone copy``s the resource.

    ``conf_path`` is REQUIRED (upfront-creds pattern, like ``FserveBroker``): without an
    explicit conf, rclone falls back to the ambient SYSTEM ``rclone.conf``, and if that
    happens to hold the remote (a dev laptop where ``i:`` is configured, a stale Colab VM)
    the copy proceeds SILENTLY from an unvetted config — exactly the raw-rclone-copy-with-no-
    conf failure the project forbids
    (``default-colab-to-the-gdrive-broker-never-emit-a-raw-rclone-copy-with-no-conf``). We fail
    loud at construction instead."""
    name = "rclone"

    def __init__(
        self,
        manifest: Dict[str, ResourceSpec],
        remote: str = "i:",
        conf_path: Optional[str] = None,
    ):
        super().__init__(manifest)
        if not conf_path:
            raise ValueError(
                "RcloneBroker requires an explicit rclone.conf — set GGE_RCLONE_CONF_PATH to a "
                "delivered rclone.conf so it never silently falls back to the SYSTEM config "
                "(a stale/unvetted remote would copy from the wrong place). On Colab use "
                "BROKER=gdrive (Drive is mounted; no rclone/conf needed)."
            )
        self.remote = remote if remote.endswith(":") else remote + ":"
        self.conf_path = conf_path

    def _ensure_rclone(self) -> None:
        if shutil.which("rclone"):
            return
        # rclone is a Go binary, not a pip package — use the official installer.
        print("  [rclone] installing rclone (official installer)…")
        subprocess.check_call(
            "curl -fsSL https://rclone.org/install.sh | sudo bash", shell=True
        )

    def _ensure_remote(self, remote: Optional[str] = None) -> None:
        """Fail LOUD if the remote isn't configured on this runtime — so a missing
        rclone.conf (e.g. a fresh Colab VM) can never become a raw ``rclone copy``
        exit-1. On Colab the fix is BROKER=gdrive (Drive is mounted, no rclone)."""
        remote = remote or self.remote
        cmd = ["rclone", "listremotes", "--config", self.conf_path]   # conf_path required (see __init__)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        remotes = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
        if remote not in remotes:
            raise RuntimeError(
                f"rclone broker: remote {remote!r} not found on this runtime "
                f"(no rclone.conf with that remote). Set GGE_RCLONE_CONF_PATH to a delivered "
                f"rclone.conf, or — on Colab — use BROKER=gdrive (Drive is mounted; no rclone/conf needed)."
            )

    def _remote_path(self, spec: ResourceSpec) -> str:
        return self.remote + _drive_relpath(spec)

    def _fetch(self, spec: ResourceSpec, target: Path) -> None:
        self._ensure_rclone()
        self._ensure_remote()
        src = self._remote_path(spec)
        cmd = ["rclone", "copy", "--config", self.conf_path]   # conf_path required (see __init__)
        # a corpus is a single file: copy it into the parent dir; a checkpoint is a dir.
        dst = str(target if spec.kind == "checkpoint" else target.parent)
        cmd += [src, dst, "--progress"]
        print(f"  [rclone] {' '.join(cmd)}")
        subprocess.check_call(cmd)

    def publish(self, local_dir: Path, dest: str) -> str:
        """Push a results dir to a durable rclone target ``dest`` (``remote:path`` —
        a *different* remote/bucket from the input, e.g. S3/B2). Fails loud if the
        dest remote isn't configured. Returns the ``dest`` it copied to."""
        src = Path(local_dir)
        if not src.is_dir():
            raise RuntimeError(f"[rclone] publish: results dir not found: {src}")
        self._ensure_rclone()
        dest_remote = (dest.split(":", 1)[0] + ":") if ":" in dest else self.remote
        self._ensure_remote(dest_remote)
        cmd = ["rclone", "copy", str(src), dest, "--config", self.conf_path, "--progress"]   # conf_path required
        print(f"  [rclone] publish: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print(f"  [rclone] published results → {dest}")
        return dest

    def verify_published(self, local_dir: Path, dest: str, relpaths: List[str]) -> None:
        """Remote readback that verifies CONTENT, not just size. Pulls
        ``rclone hashsum sha256 dest`` and compares each file's remote hash to the
        locally-computed sha256 — size(a) == size(b) does not imply equal content
        (and runner.py already records a checkpoint sha256 in DONE.json, so a
        size-only gate is weaker than evidence the artifact itself carries). A
        backend that cannot supply a hash for a file falls back to a SIZE check
        for that file, and the summary line names how many were size-only, so a
        size verification is never silently reported as a content one. Any
        mismatch/absence RAISES: no sentinel is minted."""
        self._ensure_rclone()
        # remote sha256 map (path-relative-to-dest -> hex). Backends without a
        # sha256 emit a blank/short hash; we accept only a real 64-hex digest.
        hs = subprocess.run(
            ["rclone", "hashsum", "sha256", dest, "--config", self.conf_path],
            capture_output=True, text=True, timeout=600)
        remote_hash: Dict[str, str] = {}
        if hs.returncode == 0:
            for ln in hs.stdout.splitlines():
                parts = ln.strip().split(None, 1)   # "<hexhash>  <path>"
                if len(parts) == 2:
                    hexh, rp = parts[0].lower(), parts[1]
                    if len(hexh) == 64 and all(c in "0123456789abcdef" for c in hexh):
                        remote_hash[rp] = hexh
        # remote size map (always available) for presence + size fallback.
        ls = subprocess.run(
            ["rclone", "lsl", dest, "--config", self.conf_path],
            capture_output=True, text=True, timeout=120)
        if ls.returncode != 0:
            raise RuntimeError(f"[rclone] verify_published: lsl {dest} failed: {ls.stderr.strip()[-400:]}")
        remote_bytes: Dict[str, int] = {}
        for ln in ls.stdout.splitlines():
            parts = ln.strip().split(None, 3)   # size date time path
            if len(parts) == 4:
                remote_bytes[parts[3]] = int(parts[0])

        size_only: List[str] = []
        for rel in relpaths:
            src = Path(local_dir) / rel
            want = src.stat().st_size if src.is_file() else -1
            if want <= 0:
                raise RuntimeError(
                    f"[rclone] verify_published: local {rel} missing/empty — nothing to attest")
            if rel in remote_hash:
                if _sha256(src) != remote_hash[rel]:
                    raise RuntimeError(
                        f"[rclone] verify_published: {rel} CONTENT mismatch on {dest} "
                        f"(local sha256 != remote sha256) — refusing green")
            elif remote_bytes.get(rel, -1) == want:
                size_only.append(rel)   # no remote hash — size matched, flagged below
            else:
                raise RuntimeError(
                    f"[rclone] verify_published: {rel} local={want}B remote="
                    f"{remote_bytes.get(rel, 'absent')} on {dest} — the remote does "
                    f"NOT carry this artifact; refusing green")
        n_content = len(relpaths) - len(size_only)
        note = "" if not size_only else (
            f"; ⚠️ {len(size_only)} SIZE-ONLY (backend gave no sha256): "
            f"{', '.join(size_only[:3])}{'…' if len(size_only) > 3 else ''}")
        print(f"  [rclone] verify_published: {n_content} content-verified (sha256), "
              f"{len(size_only)} size-only, of {len(relpaths)} on {dest}{note}")


class FserveBroker(_BaseBroker):
    """Authenticated curl against an fserve droplet (any-IP + passcode + fail2ban +
    TTL — built for ephemeral Colab IPs, unlike IP-allowlisted ``serve``). A
    ``codename_map`` (logical name → fserve codename) is provided at share time;
    corpora arrive as files, checkpoints as a tarball that is unpacked."""
    name = "fserve"

    def __init__(
        self,
        manifest: Dict[str, ResourceSpec],
        base_url: str,
        user: str,
        passcode: str,
        codename_map: Dict[str, str],
    ):
        super().__init__(manifest)
        self.base_url = base_url.rstrip("/")
        self.user = user
        self.passcode = passcode
        self.codename_map = codename_map

    def _codename(self, spec: ResourceSpec) -> str:
        try:
            return self.codename_map[spec.name]
        except KeyError:
            raise KeyError(f"[fserve] no codename mapped for {spec.name!r}") from None

    def _curl(self, codename: str, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        # -OJ: server picks the filename (Content-Disposition); -k: self-signed TLS.
        cmd = [
            "curl", "-fsSL", "-k", "-u", f"{self.user}:{self.passcode}",
            f"{self.base_url}/{codename}", "-OJ",
        ]
        print(f"  [fserve] curl … /{codename} -OJ  (cwd={out_dir})")
        subprocess.check_call(cmd, cwd=str(out_dir))
        got = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        if not got:
            raise RuntimeError(f"[fserve] curl of {codename} produced no file")
        return got[-1]

    def _fetch(self, spec: ResourceSpec, target: Path) -> None:
        codename = self._codename(spec)
        if spec.kind == "corpus":
            fetched = self._curl(codename, target.parent)
            if fetched.name != target.name:
                fetched.replace(target)
            return
        # checkpoint: fetch a tarball into a temp dir, unpack into target
        staging = target.parent / f".{target.name}.fserve-staging"
        if staging.exists():
            shutil.rmtree(staging)
        tarball = self._curl(codename, staging)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["tar", "-xf", str(tarball), "-C", str(target), "--strip-components=1"])
        shutil.rmtree(staging, ignore_errors=True)

    def publish(self, local_dir: Path, dest: str) -> str:
        # fserve is a SERVER (it hands files out), not a sink (it doesn't receive) —
        # publishing results to it makes no sense. Use rclone / gdrive / local instead.
        raise NotImplementedError(
            "fserve is a share server, not an output sink — set GGE_OUTPUT_BROKER to "
            "rclone (a durable remote:path), gdrive (mounted Drive), or local."
        )


class HfBroker(_BaseBroker):
    """Hugging Face Hub — snapshot-downloads a repo into ``dest`` and verifies.
    Registered so Solution 1 (mounted-gdrive AL + HF controls) is just another
    broker choice; controls normally load directly via ``from_pretrained(id)``."""
    name = "hf"

    def _fetch(self, spec: ResourceSpec, target: Path) -> None:
        if not spec.hf_model_id:
            raise RuntimeError(f"[hf] {spec.name}: no hf_model_id in the manifest")
        from huggingface_hub import snapshot_download  # local import: optional dep
        snapshot_download(repo_id=spec.hf_model_id, local_dir=str(target))


# ── Factory (chosen in the notebook CONFIG cell) ────────────────────────

def make_broker(kind: str, manifest: Dict[str, ResourceSpec], **kw) -> Broker:
    """Construct a broker by name. ``kw`` are the backend's settings.

    kind="local"|"gdrive"  → needs ``paths=PathConfig``
    kind="rclone"          → ``remote="i:"``, optional ``conf_path``
    kind="fserve"          → ``base_url``, ``user``, ``passcode``, ``codename_map``
    kind="hf"              → (no settings)
    """
    kind = kind.lower()
    if kind == "local":
        return LocalBroker(manifest, kw["paths"])
    if kind == "gdrive":
        return GdriveBroker(manifest, kw["paths"])
    if kind == "rclone":
        return RcloneBroker(manifest, remote=kw.get("remote", "i:"), conf_path=kw.get("conf_path"))
    if kind == "fserve":
        return FserveBroker(
            manifest, base_url=kw["base_url"], user=kw["user"],
            passcode=kw["passcode"], codename_map=kw["codename_map"],
        )
    if kind == "hf":
        return HfBroker(manifest)
    raise ValueError(f"unknown broker kind: {kind!r} (local|gdrive|rclone|fserve|hf)")

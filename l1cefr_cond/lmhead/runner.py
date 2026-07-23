"""Train-the-head-and-generate-the-grid engine for the LM-head notebooks.

The notebook stays thin: it calls :func:`run_grid` once per enabled decoder. This
engine loads a FROZEN backbone, wires the strategy's head module (strategy 1:
:class:`AdditiveVocabBias`), asserts the two invariants (identity-at-init +
trainable-params = head-only), lightly fits the head on the WRITER-level train
split, then GENERATES conditioned continuations over the 6 L1 × 5 CEFR grid and
writes ``preds/<l1>__<cefr>.jsonl`` + ``run-info.json`` per model. It does NOT
score — scoring is the Coordinator's held-out job (specs/96 firewall).

CLI (the S0 self-check / tracked local program):
    python -m l1cefr_cond.lmhead.runner --smoke --models gpt2-small,pythia-70m --out outputs/lm-head/selfcheck
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from l1cefr_cond.lmhead.additive_bias_mlp import ConditionedLMHead
from l1cefr_cond.lmhead.data import Example, load_corpus
from l1cefr_cond.lmhead.grid import CEFR_ORDER, L1_ORDER, N_CEFR, N_L1, cells
from l1cefr_cond.nb.registry import MODEL_REGISTRY

# neutral English openers (learner-writing style); N_prompts picks a prefix
DEFAULT_PROMPTS: List[str] = [
    "My name is",
    "Yesterday I",
    "In my free time I like to",
    "Last weekend my family",
    "I think that",
    "When I was a child",
]


@dataclass
class RunConfig:
    model_key: str
    corpus_path: str
    out_dir: str
    strategy: str = "additive-bias-mlp"
    seed: int = 42
    hidden: List[int] = field(default_factory=lambda: [64])
    train_split: Optional[str] = "train"
    # ── F28: epochs, not step-slots ──────────────────────────────────────────
    # The old loop walked `train_steps × train_batch` example SLOTS with a modulo
    # index, so a 400-slot budget over a 4-row corpus visited each row ~100 times
    # and called it training. Coverage is now a property of the corpus, not of a
    # step budget: one epoch = every train row exactly once, shuffled by seed.
    epochs: int = 1
    train_batch: int = 8
    # A hard cap for debugging only. None = a full epoch. It is recorded in
    # run-info.json, so a capped run can never be mistaken for a complete one.
    max_steps: Optional[int] = None
    # ── F28/A2: the row-count floor ─────────────────────────────────────────
    # A corpus below this is a hard failure, not a warning: the failure mode being
    # guarded against produced a GREEN run with well-formed artifacts, which no
    # amount of downstream checking catches.
    min_rows: int = 10_000
    lr: float = 1e-3
    max_seq_len: int = 128
    # ── F29 §3.1: fp16 autocast on the training forward only ─────────────────
    # The identity-at-init assertion stays in fp32 (it runs before this and is not
    # wrapped) — checking an exactness invariant under reduced precision would
    # make it pass for the wrong reason.
    fp16: bool = True
    # ── A6: periodic head-state checkpoints (published by the notebook) ──────
    checkpoint_every_steps: int = 0          # 0 = off
    resume_from: Optional[str] = None        # path to a head state_dict to resume
    # ── periodic held-out EVAL, so checkpoint selection has a curve that is not
    # the training curve. A minimum of the TRAIN loss is selected on the same
    # data the head fit — not model selection. eval_every is the CADENCE (a
    # config value, not a literal in the loop); eval_split names the held-out
    # split the loader already separates (train reads 'train' and never touches
    # it). 0 disables. NB: an eval loss on efcamdat-random-writinglevel is NOT a
    # generalisation claim — that corpus is row-split, not writer-disjoint
    # (specs/90 Q-LEAK), so Disjoint(train_writers, eval_writers) is UNDECIDED.
    # An eval curve makes selection DECIDABLE; it does not make it VALID.
    eval_every: int = 500                    # 0 = off
    eval_split: Optional[str] = "validation"
    n_prompts_per_cell: int = 3
    # Default chosen to EXCEED the p90 text length of every (l1×cefr) cell in the
    # documented corpus distribution (docs/analysis-efcamdat-random-writinglevel
    # §6, n=314,686: max cell p90 = zh/C1 1,205 chars ≈ 301 tokens at ~4 chars/
    # token English BPE), so EOS — not the cap — is the binding stop for ≥90% of
    # texts in every cell, INCLUDING the high-CEFR cells where the length signal
    # lives. The old 64 sat below even the overall MEDIAN (295 chars ≈ 74 tokens),
    # which is why 88–93% of the five published heads' generations were clipped.
    max_new_tokens: int = 384
    temperature: float = 0.8
    top_p: float = 0.95
    device: str = ""  # "" => auto (cuda if available else cpu)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(dev: str) -> str:
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _latest_state(ck_root: Path, device: str):
    """The freshest periodic resume dict under checkpoints/*/head-state.pt (hole
    a: resume must read the checkpoints that are actually written, not a
    never-written end-of-run file). Returns (state|None, source-path-str)."""
    best, best_step, src = None, -1, ""
    for p in sorted(ck_root.glob("*/head-state.pt")):
        try:
            st = torch.load(p, map_location=device)
        except Exception as e:  # a truncated download must not kill the run
            print(f"  ⚠️  skipping unreadable checkpoint {p}: {type(e).__name__}: {e}")
            continue
        step = int(st.get("global_step") or 0)
        if step > best_step:
            best, best_step, src = st, step, str(p)
    return best, src


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def load_backbone(hf_id: str, device: str, revision: Optional[str] = None):
    # F22 repro-provenance: load at the PINNED commit SHA, not the moving ``main``.
    # ``revision=None`` keeps HF's default (only for not-yet-enabled suite members).
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision)
    model = model.float()          # bf16-on-disk -> fp32 (exact identity check; sketch dtype fix)
    model.eval()
    model.to(device)
    return tok, model


# strategies whose head consumes the backbone's last hidden state (not just logits)
_HIDDEN_STRATEGIES = {"lora-delta"}

# per-strategy identity-at-init tolerance. Default (0.0) = bit-exact (additive /
# FiLM / LoRA-delta add exactly 0). gated-suppression's smooth suppress-only gate
# warm-starts to a tiny UNIFORM per-vocab shift (max|Δ| ≈ 2.5e-3, softmax-invariant
# ⇒ zero KL from base), so it asserts within atol rather than bit-exact.
_IDENTITY_ATOL: Dict[str, float] = {"gated-suppression": 1e-2}


def _build_head(strategy: str, v: int, d_model: int, hidden: List[int]):
    """Construct the strategy's LM-head module (warm-started to identity). The
    logits-only heads share ``forward(logits_base, l1_idx, cefr_idx)``; the
    hidden-dependent ones take ``forward(hidden, logits_base, l1_idx, cefr_idx)``
    (ConditionedLMHead routes each correctly via needs_hidden)."""
    h = hidden or None
    if strategy == "additive-bias-mlp":
        from l1cefr_cond.lmhead.additive_bias_mlp import AdditiveVocabBias
        return AdditiveVocabBias(N_L1, N_CEFR, v, hidden=h)
    if strategy == "film":
        from l1cefr_cond.lmhead.film import LMHeadFiLM
        return LMHeadFiLM(N_L1, N_CEFR, v, hidden=h)
    if strategy == "lora-delta":
        from l1cefr_cond.lmhead.lora_delta import LMHeadLoRADelta
        return LMHeadLoRADelta(N_L1, N_CEFR, d_model, v, r=8)
    if strategy == "gated-suppression":
        from l1cefr_cond.lmhead.gated_suppression import GatedSuppression
        return GatedSuppression(N_L1, N_CEFR, v, hidden=h)
    if strategy == "bilinear-interaction":
        from l1cefr_cond.lmhead.bilinear_interaction import BilinearInteraction
        return BilinearInteraction(N_L1, N_CEFR, v, k=64)  # logits-only; uses rank k, not `hidden`
    raise ValueError(
        f"unknown lm-head strategy: {strategy!r} "
        "(additive-bias-mlp|film|lora-delta|gated-suppression|bilinear-interaction)"
    )


def build_conditioned(model, tok, hidden: List[int], device: str, strategy: str) -> ConditionedLMHead:
    # probe the ACTUAL output-vocab width + hidden width (may differ from config)
    probe = tok("Hi", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**probe, output_hidden_states=True)
    v = out.logits.shape[-1]
    d_model = out.hidden_states[-1].shape[-1]
    head = _build_head(strategy, v, d_model, hidden).to(device)
    return ConditionedLMHead(model, head, needs_hidden=strategy in _HIDDEN_STRATEGIES).to(device)


class CorpusTooSmall(RuntimeError):
    """The train split is below ``cfg.min_rows`` (F28/A2).

    Its own type because this is the one failure that MUST stop a run rather than
    degrade it: training on a handful of rows succeeds, converges beautifully, and
    emits a complete artifact describing nothing.
    """


def assert_corpus_floor(n_rows: int, cfg: RunConfig, corpus_path: str) -> None:
    """Refuse to train on a corpus below the floor. Called in preflight, before a
    single gradient step, so the failure costs seconds rather than a T4 session."""
    if n_rows < cfg.min_rows:
        raise CorpusTooSmall(
            f"train split has {n_rows} rows, below LMHEAD_MIN_ROWS={cfg.min_rows} "
            f"(corpus={corpus_path!r}). This is the 4-row smoke-fixture failure mode: "
            f"a run this small trains to a great-looking loss by memorising a handful of "
            f"sentences. Point LMHEAD_DATASET at the real corpus, or lower LMHEAD_MIN_ROWS "
            f"deliberately (it is recorded in run-info.json) if a tiny run is genuinely intended."
        )
    print(f"  ✅ corpus floor: {n_rows} train rows >= LMHEAD_MIN_ROWS={cfg.min_rows}")


def health_flags(history: List[Dict[str, object]]) -> Dict[str, object]:
    """Non-finite sentinels over BOTH per-step series, not loss alone.

    The old ``has_nan`` inspected loss only, so it read False over runs whose
    GRADIENTS went non-finite while the loss stayed finite (measured on the four
    published heads: film 8/4899, gated-suppression 6/4899). Loss and grad fail
    for different reasons — a non-finite loss is bad labels or a CE overflow, a
    non-finite grad is an optimizer/scaler explosion — so they are reported
    separately, each with a COUNT (8-of-4899 and 1-of-4899 are different runs).
    ``has_nan`` is kept as a now-correct AGGREGATE (loss OR grad) so a reader of
    the old key stops getting a false negative."""
    losses = [h["loss"] for h in history if "loss" in h and h["loss"] is not None]
    grads = [h["grad_norm"] for h in history if h.get("grad_norm") is not None]
    n_loss = sum(1 for v in losses if not math.isfinite(float(v)))
    n_grad = sum(1 for v in grads if not math.isfinite(float(v)))
    return {
        "has_nan_loss": n_loss > 0,
        "has_nan_grad": n_grad > 0,
        "n_nonfinite_loss": n_loss,
        "n_nonfinite_grad": n_grad,
        "has_nan": (n_loss > 0) or (n_grad > 0),  # aggregate, back-compat, now fires on grad too
    }


def _epoch_order(seed: int, epoch: int, n: int) -> List[int]:
    """The shuffle for one epoch, derived from (seed, epoch) alone — NOT from a
    carried rng. A carried Random advances with every shuffle, so a mid-run
    resume could never re-derive the current epoch's order; deriving each epoch
    independently makes any (epoch, step_in_epoch) coordinate replayable."""
    import random as _random
    order = list(range(n))
    _random.Random(seed * 100_003 + epoch).shuffle(order)
    return order


@torch.no_grad()
def _eval_loss(cond_lm: ConditionedLMHead, tok, examples: List[Example], cfg: RunConfig,
               device: str) -> Dict[str, float]:
    """Token-weighted mean next-token CE over a held-out example list. Computed in
    fp32 with grad off and the head in eval mode, so the eval metric is
    deterministic and independent of the training autocast — its trend, not its
    absolute offset from the (fp16) train loss, is what makes selection decidable.
    Returns {'eval_loss', 'eval_n_rows', 'eval_n_tokens'}."""
    was_training = cond_lm.head.training
    cond_lm.head.eval()
    total_loss, total_tok = 0.0, 0
    try:
        for start in range(0, len(examples), cfg.train_batch):
            batch = examples[start:start + cfg.train_batch]
            enc = tok([e.text for e in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.max_seq_len).to(device)
            input_ids, attn = enc["input_ids"], enc["attention_mask"]
            l1_idx = torch.tensor([e.l1_idx for e in batch], device=device)
            cefr_idx = torch.tensor([e.cefr_idx for e in batch], device=device)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            cond = cond_lm(input_ids, attn, l1_idx, cefr_idx)
            tgt = labels[:, 1:].reshape(-1)
            ntok = int((tgt != -100).sum())
            if ntok == 0:
                continue
            loss = F.cross_entropy(cond[:, :-1].reshape(-1, cond.size(-1)), tgt,
                                   ignore_index=-100)  # mean over the ntok targets
            total_loss += float(loss) * ntok
            total_tok += ntok
    finally:
        if was_training:
            cond_lm.head.train()
    return {"eval_loss": (total_loss / total_tok) if total_tok else float("nan"),
            "eval_n_rows": len(examples), "eval_n_tokens": total_tok}


def train_head(cond_lm: ConditionedLMHead, tok, examples: List[Example], cfg: RunConfig, device: str,
               on_checkpoint: Optional[Callable[[str, Dict[str, object]], None]] = None,
               resume_state: Optional[Dict[str, object]] = None,
               eval_examples: Optional[List[Example]] = None,
               eval_every: int = 0) -> Dict[str, object]:
    """Fit the head for ``cfg.epochs`` FULL passes over ``examples``, shuffled per
    epoch with the run seed. Returns coverage facts (rows_consumed, epochs) so the
    artifact can be checked for completeness rather than trusted.

    ``on_checkpoint(tag, meta)`` fires every ``cfg.checkpoint_every_steps`` steps
    (tag ``step-<N>``) AND at every epoch end (tag ``epoch-<E>``); ``meta``
    carries the optimizer so the callback can persist a resume dict.
    ``resume_state`` (from a periodic checkpoint) restores optimizer/epoch/step —
    completed epochs are skipped and the resumed epoch fast-forwards past its
    already-consumed batches (replayable because of _epoch_order)."""
    if not examples:
        print("  [train] no examples — skipping fit (head stays at identity init)")
        return {"steps": 0, "epochs": 0, "rows_consumed": 0, "final_loss": float("nan"),
                "history": []}

    opt = torch.optim.Adam(cond_lm.head.parameters(), lr=cfg.lr)
    start_epoch, start_batch, step = 0, 0, 0
    if resume_state:
        opt.load_state_dict(resume_state["optimizer"])  # type: ignore[arg-type]
        step = int(resume_state.get("global_step") or 0)
        start_epoch = int(resume_state.get("epoch") or 0)
        start_batch = int(resume_state.get("batch_in_epoch") or 0)
        print(f"  [train] RESUMED at global_step={step} epoch={start_epoch} "
              f"batch_in_epoch={start_batch}")
    n = len(examples)
    use_amp = bool(cfg.fp16) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("  [train] fp16 autocast ON (training forward only; identity check stayed fp32)")

    history: List[Dict[str, float]] = []
    losses: List[float] = []
    rows_consumed = 0
    stop = False

    for epoch in range(start_epoch, cfg.epochs):
        # Shuffle per epoch, seeded per (seed, epoch): without it the batches
        # follow corpus order, which in EFCAMDAT is grouped by writer/level —
        # every batch would be one cell of the grid and the head would fit the
        # ordering, not the condition.
        order = _epoch_order(cfg.seed, epoch, n)
        batch_starts = list(range(0, n, cfg.train_batch))
        first_batch = start_batch if epoch == start_epoch else 0
        for b_i in range(first_batch, len(batch_starts)):
            start = batch_starts[b_i]
            idx = order[start:start + cfg.train_batch]
            batch = [examples[i] for i in idx]
            enc = tok([e.text for e in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.max_seq_len).to(device)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            l1_idx = torch.tensor([e.l1_idx for e in batch], device=device)
            cefr_idx = torch.tensor([e.cefr_idx for e in batch], device=device)
            # next-token CE; ignore pad positions in the target
            labels = input_ids.clone()
            labels[attn == 0] = -100
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                cond = cond_lm(input_ids, attn, l1_idx, cefr_idx)  # (B,T,V), grad only into head
                loss = F.cross_entropy(
                    cond[:, :-1].reshape(-1, cond.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # grad norm BEFORE the step, on unscaled grads (max_norm=inf ⇒ measure,
            # don't clip) — the progress plot's divergence signal.
            scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in cond_lm.head.parameters() if p.requires_grad], float("inf")))
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))
            rows_consumed += len(batch)
            step += 1
            history.append({"step": step, "epoch": epoch, "loss": losses[-1],
                            "lr": float(opt.param_groups[0]["lr"]), "grad_norm": grad_norm})
            # ── held-out eval at CADENCE, recorded ONTO this step's record so the
            # eval curve is aligned to the train curve by step (plot train-vs-eval
            # without re-running). Disjoint by construction: eval_examples come
            # from a different split than `examples` (the loader separates them).
            if eval_examples and eval_every and step % eval_every == 0:
                ev = _eval_loss(cond_lm, tok, eval_examples, cfg, device)
                history[-1].update(ev)
                print(f"  [eval] step {step} eval_loss {ev['eval_loss']:.3f} "
                      f"over {ev['eval_n_rows']} held-out rows")
            meta = {"epoch": epoch, "global_step": step, "batch_in_epoch": b_i + 1,
                    "rows_consumed": rows_consumed, "loss": losses[-1],
                    "optimizer": opt, "history": history}
            if cfg.checkpoint_every_steps and on_checkpoint and step % cfg.checkpoint_every_steps == 0:
                on_checkpoint(f"step-{step}", meta)
            if cfg.max_steps is not None and step >= cfg.max_steps:
                print(f"  [train] max_steps={cfg.max_steps} reached — STOPPING EARLY "
                      f"(this run does NOT cover the corpus)")
                stop = True
                break
            if step % 200 == 0:
                print(f"  [train] epoch {epoch + 1}/{cfg.epochs} step {step} "
                      f"rows {rows_consumed} loss {losses[-1]:.3f}")
        if on_checkpoint and not stop:
            # per-epoch checkpoint: batch_in_epoch resets so a resume starts the
            # NEXT epoch cleanly.
            on_checkpoint(f"epoch-{epoch}", {"epoch": epoch + 1, "global_step": step,
                                             "batch_in_epoch": 0, "rows_consumed": rows_consumed,
                                             "loss": losses[-1], "optimizer": opt,
                                             "history": history})
        if stop:
            break

    max_mem = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    curve = f"loss {losses[0]:.3f} -> {losses[-1]:.3f}" if losses else "no steps ran (already complete)"
    print(f"  [train] {step} steps over {cfg.epochs} epoch(s); rows_consumed={rows_consumed} "
          f"(corpus={n}); {curve}"
          + (f"; max_mem={max_mem / 2**30:.2f} GiB" if max_mem else ""))
    return {
        "steps": step,
        "epochs": cfg.epochs,
        "rows_in_split": n,
        "rows_consumed": rows_consumed,
        "covered_corpus": (cfg.max_steps is None),
        "shuffled": True,
        "fp16": use_amp,
        "max_memory_allocated": max_mem,
        "first_loss": losses[0] if losses else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        # The full curve (loss/lr/grad_norm per step), so run_grid can PERSIST it
        # (trainer_state.json). Before this, the per-step losses died with this
        # frame and the artifact carried two scalars — useless for "did it
        # diverge at step 3100".
        "history": history,
    }


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    keep = (torch.cumsum(sp, dim=-1) - sp) <= top_p  # keep the nucleus
    sp = torch.where(keep, sp, torch.zeros_like(sp))
    sp = sp / sp.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sp, num_samples=1, generator=gen)
    return si.gather(-1, choice)


@torch.no_grad()
def generate_cell(cond_lm: ConditionedLMHead, tok, prompt: str, l1_idx: int, cefr_idx: int,
                  cfg: RunConfig, device: str, gen: torch.Generator) -> Dict[str, object]:
    """Sample a conditioned continuation and RECORD why it stopped.

    Generation is manual token-by-token (NOT model.generate), so no
    generation_config can override the stop token: the tokenizer's eos_token_id
    is read here and used directly (a model may expose a LIST of eos ids —
    handled). stop_reason ∈ {eos, max_new_tokens, other}; a generation that
    stops at the cap is a PREFIX, flagged so downstream length analysis is not
    fooled by clipping."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = int(ids.shape[1])
    l1t = torch.tensor([l1_idx], device=device)
    ct = torch.tensor([cefr_idx], device=device)
    eos = tok.eos_token_id
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    stop_reason = "max_new_tokens"   # the loop running to the cap IS the truncation case
    for _ in range(cfg.max_new_tokens):
        logits = cond_lm(ids, None, l1t, ct)[:, -1, :]  # (1,V) conditioned
        nxt = _sample_next(logits, cfg.temperature, cfg.top_p, gen)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) in eos_set:
            stop_reason = "eos"
            break
    n_new = int(ids.shape[1]) - prompt_len
    if n_new == 0:
        stop_reason = "other"
    return {"text": tok.decode(ids[0], skip_special_tokens=True),
            "stop_reason": stop_reason, "n_new_tokens": n_new}


def run_grid(cfg: RunConfig) -> Dict[str, object]:
    device = _resolve_device(cfg.device)
    set_seed(cfg.seed)
    spec = MODEL_REGISTRY[cfg.model_key]
    print(f"== run_grid strategy={cfg.strategy} model={cfg.model_key} hf={spec.hf_model_id} device={device} ==")

    tok, model = load_backbone(spec.hf_model_id, device, spec.revision)
    cond_lm = build_conditioned(model, tok, cfg.hidden, device, cfg.strategy)

    # ── invariant 1+2: frozen backbone, head-only trainable, identity at init ──
    cond_lm.assert_frozen_head_only()
    bb, hd = cond_lm.trainable_param_report()
    probe = tok("Hi, my name is", return_tensors="pt").input_ids.to(device)
    l1_probe, c_probe = torch.tensor([0], device=device), torch.tensor([0], device=device)
    atol = _IDENTITY_ATOL.get(cfg.strategy, 0.0)
    with torch.no_grad():
        d0 = (cond_lm(probe, None, l1_probe, c_probe) - cond_lm.base_logits(probe)).abs().max().item()
    cond_lm.assert_identity_at_init(probe, l1_probe, c_probe, atol=atol)
    kind = "bit-exact" if atol == 0.0 else f"within atol={atol} (uniform shift ⇒ KL(base‖cond)=0)"
    print(f"  ✅ identity-at-init holds [{kind}]; max|Δlogit|={d0:.2e}; trainable: backbone={bb} head={hd}")

    # ── fit on the WRITER-level train split ──
    examples = load_corpus(Path(cfg.corpus_path), split=cfg.train_split)
    # PREFLIGHT, before any gradient step: a corpus below the floor stops the run
    # here, where it costs seconds, instead of producing a beautiful artifact of
    # nothing an hour later.
    assert_corpus_floor(len(examples), cfg, cfg.corpus_path)

    # ── held-out eval split (disjoint from train by the loader's split filter,
    # never touched during the fit). Empty when the corpus carries no such split
    # ⇒ eval is silently skipped. Row-disjoint, NOT writer-disjoint (Q-LEAK):
    # this makes checkpoint selection DECIDABLE, not the L1-recoverability claim
    # VALID. ──
    eval_examples: List[Example] = []
    if cfg.eval_every and cfg.eval_split:
        eval_examples = load_corpus(Path(cfg.corpus_path), split=cfg.eval_split)
        if not eval_examples:
            print(f"  [eval] no rows in split={cfg.eval_split!r} — eval DISABLED for this run")
        elif not set(eval_examples).isdisjoint(examples):
            # A corpus WITHOUT explicit per-row splits returns the same rows for
            # every split query, so eval would land on training rows — selecting
            # on data the head fit, which is not model selection. Refuse it rather
            # than publish a misleading eval curve. Row-level disjointness is thus
            # DECIDED here, not assumed.
            print(f"  [eval] ⚠️ split={cfg.eval_split!r} OVERLAPS train by row — eval DISABLED "
                  f"(corpus has no disjoint {cfg.eval_split} split)")
            eval_examples = []
        else:
            print(f"  [eval] {len(eval_examples)} held-out rows (split={cfg.eval_split!r}), "
                  f"row-disjoint from train, eval every {cfg.eval_every} steps")

    out = Path(cfg.out_dir) / cfg.model_key
    ck_root = out / "checkpoints"
    ck_root.mkdir(parents=True, exist_ok=True)

    # ── resume (holes a+g): if the run publishes via rclone, PULL the
    # strategy-keyed remote checkpoints first (a fresh Colab VM has an empty
    # disk — the whole point of periodic remote upload), then load the LATEST
    # periodic state found locally. cfg.resume_from still overrides. ──
    from l1cefr_cond.lmhead.ckpt_upload import (make_ckpt_uploader,
                                                pull_run_checkpoints,
                                                resolve_ckpt_dest)
    ckpt_dest = resolve_ckpt_dest(cfg.strategy)
    uploader = make_ckpt_uploader()
    if ckpt_dest and uploader is not None:
        pull_run_checkpoints(ckpt_dest, cfg.model_key, ck_root)
    if cfg.resume_from:
        resume_state: Optional[Dict[str, object]] = torch.load(cfg.resume_from, map_location=device)
        src = cfg.resume_from
    else:
        resume_state, src = _latest_state(ck_root, device)
    if resume_state:
        cond_lm.head.load_state_dict(
            resume_state["head"] if "head" in resume_state else resume_state)  # type: ignore[index]
        print(f"  ✅ resumed head state from {src}")
    prior_history: List[Dict[str, float]] = list(resume_state.get("history") or []) if resume_state else []  # type: ignore[union-attr]

    def _write_trainer_state(history: List[Dict[str, float]]) -> Path:
        full = prior_history + history
        loss_vals = [float(h["loss"]) for h in full]
        by_epoch: Dict[int, List[float]] = {}
        for h in full:
            by_epoch.setdefault(int(h.get("epoch", 0)), []).append(float(h["loss"]))
        # eval curve, promoted to its own array so a downstream plot reads it
        # directly (it is also inline on the per-step records, aligned by step).
        eval_log = [{"step": h["step"], "epoch": h.get("epoch"), "eval_loss": h["eval_loss"],
                     "eval_n_rows": h.get("eval_n_rows")}
                    for h in full if h.get("eval_loss") is not None]
        state_path = ck_root / "trainer_state.json"
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "log_history": full,
            "epochs_log": [{"epoch": e, "steps": len(v), "mean_loss": sum(v) / len(v),
                            "final_loss": v[-1]} for e, v in sorted(by_epoch.items())],
            "eval_log": eval_log,
            "first_loss": loss_vals[0] if loss_vals else None,
            "final_loss": loss_vals[-1] if loss_vals else None,
            **health_flags(full),   # has_nan_loss / has_nan_grad / n_nonfinite_* / has_nan(aggregate)
            "n_steps": len(loss_vals),
            "epochs": cfg.epochs,
        }, indent=2))
        tmp.replace(state_path)
        return state_path

    def _save_head_safetensors(dir_: Path) -> Path:
        from safetensors.torch import save_file
        dir_.mkdir(parents=True, exist_ok=True)
        p = dir_ / "head.safetensors"
        tmp = p.with_suffix(".safetensors.tmp")
        save_file({k: v.detach().cpu().contiguous() for k, v in cond_lm.head.state_dict().items()}, tmp)
        tmp.replace(p)
        return p

    def _checkpoint(tag: str, meta: Dict[str, object]) -> None:
        """Periodic/per-epoch save: checkpoints/<tag>/{head.safetensors,head-state.pt}
        + refreshed trainer_state.json — then IMMEDIATELY uploaded to the run's
        gdrive DEST (readback-verified), so a Colab preemption loses at most one
        interval, never the run."""
        ck = ck_root / tag
        _save_head_safetensors(ck)
        opt = meta["optimizer"]
        state = {
            "head": cond_lm.head.state_dict(),
            "optimizer": opt.state_dict(),  # type: ignore[union-attr]
            "scheduler": None,              # no scheduler yet; slot reserved
            # The shuffle is DERIVED (per-epoch Random(seed*100003+epoch), see
            # _epoch_order), so replay needs no captured rng state — recording
            # the scheme + seed keeps that auditable from the artifact.
            "rng": {"scheme": "per-epoch Random(seed*100003+epoch)", "seed": cfg.seed},
            "global_step": meta["global_step"],
            "epoch": meta["epoch"],
            "batch_in_epoch": meta["batch_in_epoch"],
            "history": prior_history + list(meta["history"]),  # type: ignore[operator]
        }
        # Write-then-rename: a checkpoint interrupted mid-write must not replace a
        # good one — on a preemptible T4 that is the difference between resuming
        # and starting over.
        sp = ck / "head-state.pt"
        tmp = sp.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(sp)
        ts_path = _write_trainer_state(list(meta["history"]))  # type: ignore[arg-type]
        print(f"  [ckpt] {tag} -> {ck} (state {sp.stat().st_size} bytes)")
        if uploader is not None and ckpt_dest:
            # hole (c)/(f): upload NOW and verify by remote readback. Failure is
            # LOUD but non-fatal — training continues; the end-of-run sentinel
            # still refuses green if the FINAL checkpoint is not on the remote.
            ok = uploader(ck, f"{ckpt_dest}/{cfg.model_key}/checkpoints/{tag}")
            ok = uploader(ts_path, f"{ckpt_dest}/{cfg.model_key}/checkpoints") and ok
            if not ok:
                print(f"  ⚠️  [ckpt] REMOTE UPLOAD FAILED for {tag} — progress is local-only "
                      f"until the next interval; the run sentinel will catch a missing final.")

    train_stats = train_head(cond_lm, tok, examples, cfg, device,
                             on_checkpoint=_checkpoint, resume_state=resume_state,
                             eval_examples=eval_examples, eval_every=cfg.eval_every)
    history = list(train_stats.pop("history", []))

    # ── UNCONDITIONAL end-of-run save (before generation, so per-model isolation
    # keeps each model its own dir): the flat checkpoints/head.safetensors is the
    # canonical reloadable artifact verify_trained_model checks, independent of
    # which periodic tags exist. ──
    head_path = _save_head_safetensors(ck_root)
    _write_trainer_state(history)
    losses = [float(h["loss"]) for h in prior_history + history]
    print(f"  ✅ saved trained head -> {head_path} ({head_path.stat().st_size} bytes) "
          f"+ loss curve ({len(losses)} steps)")

    # ── progress figure, INSIDE the run dir so the publish carries it to Drive.
    # Best-effort: a missing matplotlib must never kill a finished training run.
    try:
        from l1cefr_cond.lmhead.plotting import plot_progress
        plot_progress(ck_root / "trainer_state.json", out / "figures" / "progress.png")
    except Exception as e:  # noqa: BLE001 — plotting is reporting, not training
        print(f"  ⚠️  progress plot failed (non-fatal): {type(e).__name__}: {e}")

    # ── generate the 6×5 grid (generation only — NOT scored here) ──
    preds_dir = out / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    prompts = DEFAULT_PROMPTS[: cfg.n_prompts_per_cell]
    n_preds = 0
    samples: List[Dict[str, str]] = []
    from l1cefr_cond.lmhead.gen_health import cell_health, degeneracy_flags
    gen_health_per_cell: Dict[str, Dict[str, object]] = {}
    all_gen_rows: List[Dict[str, object]] = []
    for lc, cb, li, ci in cells():
        rows = []
        for pi, prompt in enumerate(prompts):
            g = generate_cell(cond_lm, tok, prompt, li, ci, cfg, device, gen)
            rows.append({"l1": lc, "cefr": cb, "prompt_idx": pi, "prompt": prompt,
                         "generation": g["text"], "stop_reason": g["stop_reason"],
                         "n_new_tokens": g["n_new_tokens"], **degeneracy_flags(g["text"])})
            n_preds += 1
        (preds_dir / f"{lc}__{cb}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        gen_health_per_cell[f"{lc}/{cb}"] = cell_health(rows)
        all_gen_rows.extend(rows)
        if len(samples) < 3:
            samples.append({"cell": f"{lc}/{cb}", "sample": rows[0]["generation"]})
    generation_health = {
        "max_new_tokens": cfg.max_new_tokens,
        "overall": cell_health(all_gen_rows),
        "per_cell": gen_health_per_cell,
    }
    _oh = generation_health["overall"]
    print(f"  [gen] truncation_rate={_oh['truncation_rate']:.3f} "
          f"degenerate_rate={_oh['degenerate_rate']:.3f} "
          f"(stop_reasons={_oh['stop_reasons']}) at cap {cfg.max_new_tokens}")

    run_info = {
        "strategy": cfg.strategy,
        "model_key": cfg.model_key,
        "hf_model_id": spec.hf_model_id,
        "model_revision": spec.revision,  # F22: the PINNED HF-Hub commit SHA (None ⇒ unpinned)
        "model_family": spec.model_family,
        "git_sha": _git_sha(),
        # The SHA of the PUBLIC mirror the Colab VM actually cloned (set by the notebook's
        # clone cell into LMHEAD_MIRROR_SHA); falls back to the local rev on a workstation run.
        # Proves which code ran even if a reused runtime skipped the re-clone (stale-code guard).
        "mirror_sha": os.environ.get("LMHEAD_MIRROR_SHA") or _git_sha(),
        "seed": cfg.seed,
        "device": device,
        "decode": {"max_new_tokens": cfg.max_new_tokens, "temperature": cfg.temperature, "top_p": cfg.top_p},
        # per-cell stop-reason coverage + truncation rate + degeneracy (empty,
        # TTR, repeated-token run). REPORT-ONLY — it does NOT feed the completion
        # sentinel's rc: rc attests the run did what it CLAIMED (trained a head,
        # generated the grid, published + verified the artifacts); a model that
        # generates degenerately has still done that, so degeneracy is a RESULT,
        # not an execution failure. Coupling rc to a quality threshold would make
        # rc unable to tell a broken run from a working run of a bad model, and
        # would smuggle an arbitrary threshold into the sentinel. Quality is the
        # Coordinator's held-out job (specs/96 firewall), not the executor's.
        "generation_health": generation_health,
        "grid": {"n_l1": N_L1, "n_cefr": N_CEFR, "l1_order": list(L1_ORDER), "cefr_order": list(CEFR_ORDER)},
        "trainable_params": {"backbone": bb, "head": hd},
        "identity_at_init": True,
        "train": train_stats,
        # ── coverage, promoted out of `train` so no reader has to dig ──
        "rows_consumed": train_stats.get("rows_consumed", 0),
        "rows_in_split": train_stats.get("rows_in_split", 0),
        "covered_corpus": train_stats.get("covered_corpus", False),
        "max_memory_allocated": train_stats.get("max_memory_allocated", 0),
        "corpus_path": str(cfg.corpus_path),
        "min_rows": cfg.min_rows,
        # ── Q-LEAK, stated in the artifact itself (checklist B4) ──
        # The interim corpus is split by row, not by writer, so the same author can
        # appear in train and test. Every consumer of this file must be able to see
        # that without knowing the backstory, which is why it is a field and not a
        # README note. Flip to True only when a writer-disjoint split lands.
        "writer_disjoint": False,
        "leakage_note": ("interim corpus: row-level split, NOT writer-disjoint (Q-LEAK open). "
                         "Any metric derived from this run is UNVALIDATED for the "
                         "'reads as an L1-X learner' claim."),
        "n_cells": N_L1 * N_CEFR,
        "n_preds": n_preds,
        # ── weights-on-Drive, assertable from run-info alone (checkpoint gate) ──
        "checkpoint": {
            "path": "checkpoints/head.safetensors",
            "sha256": hashlib.sha256(head_path.read_bytes()).hexdigest(),
            "bytes": head_path.stat().st_size,
        },
        "loss_curve": {
            "path": "checkpoints/trainer_state.json",
            "n_steps": len(losses),
            "final_loss": losses[-1] if losses else None,
            # health over BOTH per-step series (loss AND grad_norm), each with a
            # count — a run with non-finite gradients but finite losses is now
            # visible from run-info alone (was invisible: has_nan was loss-only).
            **health_flags(prior_history + history),
            # eval, so a reader knows an eval curve exists and where it ended
            # (row-split only — NOT a generalisation number; see Q-LEAK).
            "eval": {
                "n_evals": sum(1 for h in (prior_history + history) if h.get("eval_loss") is not None),
                "final_eval_loss": next(
                    (h["eval_loss"] for h in reversed(prior_history + history)
                     if h.get("eval_loss") is not None), None),
                "split": cfg.eval_split, "every": cfg.eval_every,
            },
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _pkg("transformers"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scored": False,  # firewall: the Coordinator scores held-out; the notebook only generates
        "config": asdict(cfg),
    }
    (out / "run-info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False))
    print(f"  ✅ wrote {n_preds} generations across {N_L1 * N_CEFR} cells -> {out}")
    return {"model_key": cfg.model_key, "out": str(out), "identity_at_init": True,
            "trainable_backbone": bb, "trainable_head": hd, "n_preds": n_preds,
            "samples": samples, "train": train_stats}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train the LM-head + generate the 6x5 grid (generation only).")
    ap.add_argument("--models", default="gpt2-small", help="comma-separated MODEL_REGISTRY keys")
    ap.add_argument("--strategy", default="additive-bias-mlp",
                    help="lm-head strategy (additive-bias-mlp|film|lora-delta|gated-suppression|bilinear-interaction)")
    ap.add_argument("--dataset-path", default="data/smoke/efcamdat_smoke.jsonl")
    ap.add_argument("--out", default="outputs/lm-head/run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny settings for a fast S0 self-check")
    ap.add_argument("--device", default="")
    args = ap.parse_args(argv)

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    summaries = []
    for k in keys:
        if args.smoke:
            # The S0 self-check runs on the 4-row smoke fixture ON PURPOSE, so it
            # must lower the floor EXPLICITLY. That is the whole design of the
            # floor: a tiny run is possible but never accidental, and `min_rows`
            # lands in run-info.json so the artifact says which kind of run it was.
            cfg = RunConfig(
                model_key=k, corpus_path=args.dataset_path, out_dir=args.out, strategy=args.strategy,
                # batch=2 over the 4-row fixture = 2 steps, so the persisted loss
                # curve has a slope for verify_trained_model's decrease check —
                # at batch=4 the whole fixture was one step and first==final.
                seed=args.seed, hidden=[64], train_split="train", epochs=1, train_batch=2, lr=1e-3,
                max_steps=3, min_rows=1, fp16=False, checkpoint_every_steps=1,
                n_prompts_per_cell=1, max_new_tokens=8, temperature=0.8, top_p=0.95, device=args.device,
            )
        else:
            cfg = RunConfig(model_key=k, corpus_path=args.dataset_path, out_dir=args.out,
                            strategy=args.strategy, seed=args.seed, device=args.device)
        summaries.append(run_grid(cfg))

    print("\n== SELF-CHECK SUMMARY ==")
    ok = True
    for s in summaries:
        good = bool(s["identity_at_init"]) and int(s["trainable_backbone"]) == 0 and int(s["n_preds"]) > 0
        ok = ok and good
        print(f"  {'✅' if good else '❌'} {s['model_key']}: identity={s['identity_at_init']} "
              f"backbone_trainable={s['trainable_backbone']} head_trainable={s['trainable_head']} "
              f"n_preds={s['n_preds']}")
    print(f"LMHEAD_SELFCHECK rc={0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

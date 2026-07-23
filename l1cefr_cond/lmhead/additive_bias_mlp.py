"""LM-Head Additive Vocab-Bias MLP — strategy 1 (specs/96).

Mechanism: ``logits' = logits_base + MLP([e_l1 ; e_cefr])`` — an additive bias
per vocab entry. No LoRA, no attention, no backbone touch. Works on tied OR
untied LM heads unchanged because it never reads/writes the head weight matrix —
pure post-hoc logit arithmetic.

``AdditiveVocabBias`` is GRAFTED VERBATIM from
``docs/lm-head-additive-bias-mlp/sketch.py`` (the verified, runnable seed).
``ConditionedLMHead`` is the thin composition added here: a frozen backbone +
the head module, with the identity-at-init and trainable-params=head-only
invariants exposed as methods the notebook/runner assert on.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class AdditiveVocabBias(nn.Module):
    """One-or-many-layer MLP mapping [e_l1 ; e_cefr] -> a full-vocab additive logit bias."""

    def __init__(
        self,
        n_l1: int,
        n_cefr: int,
        vocab_size: int,
        d_l1: int = 32,
        d_cefr: int = 16,
        hidden: Optional[List[int]] = None,  # None/[] = single linear layer ("1 layer")
    ) -> None:
        super().__init__()
        self.e_l1 = nn.Embedding(n_l1, d_l1)
        self.e_cefr = nn.Embedding(n_cefr, d_cefr)
        dims = [d_l1 + d_cefr, *(hidden or []), vocab_size]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)  # warm-start: bias=0 at init, base model untouched
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, logits_base: torch.Tensor, l1_idx: torch.Tensor, cefr_idx: torch.Tensor) -> torch.Tensor:
        # logits_base: (B, T, V); l1_idx/cefr_idx: (B,)
        e = torch.cat([self.e_l1(l1_idx), self.e_cefr(cefr_idx)], dim=-1)  # (B, d_l1+d_cefr)
        bias = self.mlp(e).unsqueeze(1)  # (B, 1, V) broadcasts over sequence positions
        return logits_base + bias


class ConditionedLMHead(nn.Module):
    """A frozen decoder backbone + the additive-bias head.

    The backbone is frozen at construction (``requires_grad_(False)``) and its
    logits are computed under ``no_grad`` — gradient flows ONLY into the head
    module. ``forward`` returns the conditioned logits ``base + bias``.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, needs_hidden: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        # hidden-state-dependent heads (e.g. lora-delta) also take the last hidden
        # state, not just the base logits — forward surfaces it when set.
        self.needs_hidden = needs_hidden
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def base_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        l1_idx: torch.Tensor,
        cefr_idx: torch.Tensor,
    ) -> torch.Tensor:
        # backbone is frozen ⇒ base logits (and hidden) under no_grad; grad flows only into head.
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=self.needs_hidden)
        base = out.logits
        if self.needs_hidden:
            hidden = out.hidden_states[-1]  # (B, T, d_model) pre-lm_head
            return self.head(hidden, base, l1_idx, cefr_idx)
        return self.head(base, l1_idx, cefr_idx)

    # ── invariants the notebook/runner assert on ──────────────────────────
    def trainable_param_report(self) -> Tuple[int, int]:
        """(backbone_trainable, head_trainable). The frozen-backbone contract is
        ``backbone_trainable == 0`` and ``head_trainable == every head param``."""
        bb = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        hd = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return bb, hd

    def assert_frozen_head_only(self) -> None:
        bb, hd = self.trainable_param_report()
        head_total = sum(p.numel() for p in self.head.parameters())
        if bb != 0:
            raise AssertionError(f"backbone not frozen: {bb} trainable params")
        if hd != head_total:
            raise AssertionError(f"head not fully trainable: {hd} != {head_total}")

    def assert_identity_at_init(
        self, input_ids: torch.Tensor, l1_idx: torch.Tensor, cefr_idx: torch.Tensor, atol: float = 0.0
    ) -> None:
        """Warm-start check: at init the conditioned logits equal the base logits,
        so the modulated model starts as the base. Routed through ``forward`` so
        hidden-dependent heads are covered too.

        ``atol == 0`` (additive / FiLM / LoRA-delta): the modulation is EXACTLY 0,
        asserted **bit-exact** via ``torch.equal``. A strictly suppress-only smooth
        log-gate (gated-suppression) cannot reach ``gate == 1`` with finite params,
        so it warm-starts to a tiny **uniform** per-vocab shift (max|Δ| ≈ 2.5e-3,
        invariant under softmax ⇒ zero KL from base); it asserts within ``atol``."""
        base = self.base_logits(input_ids)
        cond = self.forward(input_ids, None, l1_idx, cefr_idx)
        delta = (cond - base).abs().max().item()
        ok = torch.equal(cond, base) if atol == 0.0 else torch.allclose(cond, base, atol=atol, rtol=0.0)
        if not ok:
            raise AssertionError(f"identity-at-init FAILED: max|cond-base|={delta} (atol={atol})")

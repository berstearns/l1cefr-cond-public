"""LM-Head FiLM (scale + shift) — strategy 2 (specs/96).

Mechanism: ``logits' = γ(e) ⊙ logits_base + β(e)`` — a per-vocab multiplicative
scale γ and additive shift β generated from [e_l1 ; e_cefr]. Unlike the additive
strategy (shift only), FiLM can *sharpen or flatten* the base distribution per
cell, not just offset it. Warm-started to γ=1, β=0 (the last layer emits
``gamma_raw=0, beta=0``) ⇒ **identity at init** (bit-exact: 1.0·x + 0.0 == x).

``LMHeadFiLM`` is GRAFTED VERBATIM from ``docs/lm-head-film/sketch.py`` (only
type annotations tightened). It shares the head interface
``forward(logits_base, l1_idx, cefr_idx) -> logits`` with the additive strategy,
so it composes with the same ``ConditionedLMHead`` wrapper (frozen backbone +
head; identity-at-init + trainable-params=head-only asserts).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class LMHeadFiLM(nn.Module):
    def __init__(
        self,
        n_l1: int,
        n_cefr: int,
        vocab_size: int,
        d_l1: int = 32,
        d_cefr: int = 16,
        hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.e_l1 = nn.Embedding(n_l1, d_l1)
        self.e_cefr = nn.Embedding(n_cefr, d_cefr)
        d_in = d_l1 + d_cefr
        dims = [d_in, *(hidden or []), 2 * vocab_size]  # last layer emits [gamma_raw ; beta]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.gen = nn.Sequential(*layers)
        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)  # gamma_raw=0, beta=0 at init
        self.vocab_size = vocab_size

    def forward(self, logits_base: torch.Tensor, l1_idx: torch.Tensor, cefr_idx: torch.Tensor) -> torch.Tensor:
        e = torch.cat([self.e_l1(l1_idx), self.e_cefr(cefr_idx)], dim=-1)
        out = self.gen(e)  # (B, 2*V)
        gamma_raw, beta = out.split(self.vocab_size, dim=-1)
        gamma = 1.0 + gamma_raw  # warm-start: gamma_raw=0 -> gamma=1 (identity scale)
        return gamma.unsqueeze(1) * logits_base + beta.unsqueeze(1)

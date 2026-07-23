"""LM-Head LoRA Delta — strategy 3 (specs/96).

Mechanism: ``logits' = logits_base + h @ (B_l1 A_l1 + B_cefr A_cefr)ᵀ`` — a
**hidden-state-dependent** low-rank delta applied ONLY at the output projection
(a soft, output-only untying), so it is correct on tied (gpt2/smollm2) AND untied
(pythia) LM heads: it never reads/writes ``W_0``, only adds a separate low-rank
term computed from the last hidden state ``h``. ``B`` is zero-initialized ⇒ ΔW=0
⇒ **identity at init** (bit-exact: base + 0 == base).

Unlike the additive/FiLM strategies (logits-only), this head consumes the
backbone's last hidden state, so ``ConditionedLMHead`` is constructed with
``needs_hidden=True`` and its ``forward(hidden, logits_base, l1_idx, cefr_idx)``
receives ``hidden_states[-1]``.

``LMHeadLoRADelta`` is GRAFTED VERBATIM from ``docs/lm-head-lora-delta/sketch.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LMHeadLoRADelta(nn.Module):
    def __init__(self, n_l1: int, n_cefr: int, d_model: int, vocab_size: int, r: int = 8) -> None:
        super().__init__()
        self.r = r
        # one (A,B) factor pair PER L1 value and PER CEFR band -- C2 composition, same as T03/T06
        self.A_l1 = nn.Parameter(torch.randn(n_l1, r, d_model) * 0.01)
        self.B_l1 = nn.Parameter(torch.zeros(n_l1, vocab_size, r))       # zero-init: warm-start
        self.A_cefr = nn.Parameter(torch.randn(n_cefr, r, d_model) * 0.01)
        self.B_cefr = nn.Parameter(torch.zeros(n_cefr, vocab_size, r))   # zero-init: warm-start

    def forward(self, hidden: torch.Tensor, logits_base: torch.Tensor,
                l1_idx: torch.Tensor, cefr_idx: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, d_model) -- the LAST hidden state, pre-lm_head
        A_l1, B_l1 = self.A_l1[l1_idx], self.B_l1[l1_idx]        # (B, r, d), (B, V, r)
        A_c, B_c = self.A_cefr[cefr_idx], self.B_cefr[cefr_idx]  # (B, r, d), (B, V, r)

        h_l1 = torch.einsum("btd,brd->btr", hidden, A_l1)   # (B, T, r)
        delta_l1 = torch.einsum("btr,bvr->btv", h_l1, B_l1)  # (B, T, V)
        h_c = torch.einsum("btd,brd->btr", hidden, A_c)
        delta_c = torch.einsum("btr,bvr->btv", h_c, B_c)

        return logits_base + delta_l1 + delta_c

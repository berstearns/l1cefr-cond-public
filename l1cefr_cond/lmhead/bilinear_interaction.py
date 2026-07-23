"""LM-Head Bilinear L1×CEFR Interaction — strategy 5 (specs/96).

Mechanism::

    logits' = logits_base + W_out(W_u·e_l1 ⊙ W_v·e_cefr) + b_l1[ℓ] + b_cefr[c]

The elementwise product ``(W_u·e_l1) ⊙ (W_v·e_cefr)`` forces a genuine
*multiplicative* interaction between the L1 and the CEFR factor — a term that a
plain concatenation-MLP (strategy 1) cannot represent — projected to vocab by
``W_out``. Two full-vocab **additive fallback** rows (``b_l1``, ``b_cefr``) let
each factor also shift logits on its own. This is a logits-only head: it never
reads/writes the backbone LM-head weight, so it is correct on tied (gpt2/smollm2)
AND untied (pythia) heads unchanged, and it shares the
``forward(logits_base, l1_idx, cefr_idx) -> logits`` interface with the
additive / FiLM / gated strategies, composing with the same ``ConditionedLMHead``
wrapper (frozen backbone; identity-at-init + trainable-params = head-only asserts).

``BilinearInteraction`` is GRAFTED VERBATIM from
``docs/lm-head-bilinear-interaction/sketch.py`` (only type annotations tightened,
matching ``film.py`` / ``gated_suppression.py``).

Warm-start & identity-at-init (BIT-EXACT, atol = 0.0)
-----------------------------------------------------
The added term is a sum of **three independent contributions**, and warm-start
zeroes all three so their sum is the exact zero vector:

* ``W_out`` — ``nn.init.zeros_(self.W_out.weight)`` with ``bias=False`` ⇒
  ``W_out(z) = z @ 0ᵀ = 0`` **exactly, for any z**. This is the subtle part: the
  interaction *input* ``z = (W_u·e_l1) ⊙ (W_v·e_cefr)`` is itself **non-zero** at
  init (``e_l1``, ``e_cefr``, ``W_u``, ``W_v`` keep their default random init).
  The interaction path is annihilated by the zeroed **output** projection, not by
  a zeroed input — so identity here depends on zeroing ``W_out``, not on the
  embeddings being zero.
* ``b_l1`` — ``nn.init.zeros_(self.b_l1.weight)`` ⇒ the looked-up row is all zeros.
* ``b_cefr`` — ``nn.init.zeros_(self.b_cefr.weight)`` ⇒ the looked-up row is all zeros.

Hence ``bias = 0 + 0 + 0`` is the exact fp32 zero vector, and
``logits_base + 0 == logits_base`` bit-exact (IEEE-754: ``x + 0.0 == x`` for
finite ``x``). So this strategy asserts identity **bit-exact** via ``torch.equal``
— ``_IDENTITY_ATOL`` default (0.0) applies, NO tolerance is needed — the same as
additive / FiLM / LoRA-delta and UNLIKE gated-suppression (whose smooth
suppress-only log-gate cannot reach exactly 1 with finite params, so it only
warm-starts *near* identity). It is "trickiest" only in that **all three** terms
must be zeroed at once and the interaction term's zero comes from ``W_out``, not
from its non-zero inputs; get any one wrong and warm-start breaks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BilinearInteraction(nn.Module):
    """Forced multiplicative L1×CEFR interaction → per-vocab logit bias (+ additive fallbacks)."""

    def __init__(
        self,
        n_l1: int,
        n_cefr: int,
        vocab_size: int,
        d_l1: int = 32,
        d_cefr: int = 16,
        k: int = 64,  # interaction rank
    ) -> None:
        super().__init__()
        self.e_l1 = nn.Embedding(n_l1, d_l1)
        self.e_cefr = nn.Embedding(n_cefr, d_cefr)
        self.W_u = nn.Linear(d_l1, k, bias=False)
        self.W_v = nn.Linear(d_cefr, k, bias=False)
        self.W_out = nn.Linear(k, vocab_size, bias=False)
        nn.init.zeros_(self.W_out.weight)  # warm-start: interaction term contributes 0 at init

        # additive fallback terms -- one full vocab-sized bias row PER L1 value and PER CEFR value
        self.b_l1 = nn.Embedding(n_l1, vocab_size)
        self.b_cefr = nn.Embedding(n_cefr, vocab_size)
        nn.init.zeros_(self.b_l1.weight)
        nn.init.zeros_(self.b_cefr.weight)

    def forward(self, logits_base: torch.Tensor, l1_idx: torch.Tensor, cefr_idx: torch.Tensor) -> torch.Tensor:
        # logits_base: (B, T, V); l1_idx/cefr_idx: (B,)
        u = self.W_u(self.e_l1(l1_idx))       # (B, k)
        v = self.W_v(self.e_cefr(cefr_idx))   # (B, k)
        z = u * v                             # (B, k) -- the forced interaction term
        interaction_bias = self.W_out(z)      # (B, V)
        bias = interaction_bias + self.b_l1(l1_idx) + self.b_cefr(cefr_idx)  # (B, V)
        return logits_base + bias.unsqueeze(1)  # (B,1,V) broadcasts over T

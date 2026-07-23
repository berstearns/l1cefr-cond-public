"""LM-Head Gated Suppression — strategy 4 (specs/96).

Mechanism: ``logits' = logits_base + log(sigmoid(MLP([e_l1;e_cefr])) + eps)`` — a
per-vocab, **one-directional (suppress-only)** log-gate. Because ``gate ∈ (0,1]``,
``log(gate) ≤ 0``: the head can only ever *lower* a token's logit, never raise it.
This is the conditioning-as-inhibition inductive bias — an L1×CEFR cell learns
which tokens a learner at that level would *avoid*, not which to boost.

``GatedSuppression`` is GRAFTED VERBATIM from ``docs/lm-head-gated-suppression/
sketch.py`` (only type annotations tightened, matching ``film.py``). It shares the
logits-only head interface ``forward(logits_base, l1_idx, cefr_idx) -> logits``
with the additive / FiLM strategies, so it composes with the same
``ConditionedLMHead`` wrapper (frozen backbone; identity-at-init +
trainable-params = head-only asserts).

Warm-start & identity-at-init
-----------------------------
The last MLP layer is zero-init on the weight with ``bias = INIT_BIAS = 6.0``, so
at init ``MLP(e) ≡ 6.0`` for **every** vocab entry (the zeroed weight removes all
``e``-dependence) ⇒ ``gate ≡ sigmoid(6.0) ≈ 0.9975`` ⇒ ``log(gate) ≈ −0.00248``
added **uniformly** to every logit. Two consequences:

* A *strictly* suppress-only smooth gate (``gate ≤ 1`` with equality at init)
  places init at an interior maximum of ``log(gate)``, where any smooth gate has
  **zero training gradient** — a dead, untrainable head. So we deliberately keep
  the sketch's near-identity warm-start (``sigmoid(6.0)``, gradient ≈ 2.5e-3 ≠ 0):
  the head is genuinely trainable.
* The init perturbation is a **constant vector** added to all logits, hence
  **invariant under softmax** (``softmax(x + c·1) == softmax(x)``) ⇒ **zero KL
  from the base distribution** ⇒ functionally exact identity for generation.

So this strategy asserts identity-at-init within a tight ``atol`` (max|Δlogit|
≈ 2.5e-3), not bit-exact 0 — see ``ConditionedLMHead.assert_identity_at_init``'s
``atol`` argument and ``runner._IDENTITY_ATOL``.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

INIT_BIAS = 6.0  # sigmoid(6.0) ≈ 0.9975 → log(gate) ≈ −0.0025, a near-no-op (uniform) at init


class GatedSuppression(nn.Module):
    """One-or-many-layer MLP mapping [e_l1 ; e_cefr] -> a per-vocab log-gate ≤ 0."""

    def __init__(
        self,
        n_l1: int,
        n_cefr: int,
        vocab_size: int,
        d_l1: int = 32,
        d_cefr: int = 16,
        hidden: Optional[List[int]] = None,  # None/[] = single linear layer ("1 layer")
        eps: float = 1e-6,
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
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, INIT_BIAS)  # warm-start: gate ≈ 1 everywhere
        self.eps = eps

    def forward(self, logits_base: torch.Tensor, l1_idx: torch.Tensor, cefr_idx: torch.Tensor) -> torch.Tensor:
        # logits_base: (B, T, V); l1_idx/cefr_idx: (B,)
        e = torch.cat([self.e_l1(l1_idx), self.e_cefr(cefr_idx)], dim=-1)
        gate = torch.sigmoid(self.mlp(e))  # (B, V), in (0,1)
        return logits_base + torch.log(gate + self.eps).unsqueeze(1)  # (B,1,V) broadcasts over T

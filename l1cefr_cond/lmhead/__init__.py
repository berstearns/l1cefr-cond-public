"""l1cefr_cond.lmhead — LM-head conditioning strategies (specs/96).

Each strategy modulates a **frozen** decoder's output logits from an
(L1, CEFR) conditioning vector; only the small head module trains. Strategy 1
(this pair) is the additive vocab-bias MLP grafted from
``docs/lm-head-additive-bias-mlp/sketch.py``. The shared grid / data loader /
train-and-generate engine live alongside so every strategy notebook stays thin.
"""
from __future__ import annotations

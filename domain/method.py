"""domain/method.py — the conditioning ``Method`` as a SUM TYPE.

Each method is a frozen dataclass carrying ONLY its own valid fields, so an
illegal config ("a LoRA with a hypernet field") is *unconstructible* — no
``Optional``-field guard soup. ``describe()`` consumes the union with a total
``match`` ending in ``assert_never``, so adding a variant without handling it
is a ``mypy --strict`` error (compile-time exhaustiveness).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

if sys.version_info >= (3, 11):  # 3.10 target: assert_never lives in typing_extensions
    from typing import assert_never
else:  # pragma: no cover
    from typing_extensions import assert_never


@dataclass(frozen=True, slots=True)
class ControlToken:  # [T01] categorical control tokens prepended to the prompt
    scheme: str


@dataclass(frozen=True, slots=True)
class LoRA:  # [T03] low-rank adapters on named projection modules
    rank: int
    alpha: int
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError(f"LoRA rank/alpha must be >= 1, got r={self.rank} a={self.alpha}")
        if not self.target_modules:
            raise ValueError("LoRA needs >= 1 target module")


@dataclass(frozen=True, slots=True)
class Hypernet:  # [T07] generate adapter params from a typology vector [S01]
    typology_dim: int
    generated_rank: int

    def __post_init__(self) -> None:
        if self.typology_dim < 1 or self.generated_rank < 1:
            raise ValueError(
                f"Hypernet dims must be >= 1, got t={self.typology_dim} r={self.generated_rank}"
            )


@dataclass(frozen=True, slots=True)
class DPO:  # [T09] preference optimization
    beta: float
    n_pairs: int


@dataclass(frozen=True, slots=True)
class GRPO:  # [T10] RL with token-level rewards
    group_size: int
    kl_coeff: float


@dataclass(frozen=True, slots=True)
class Steering:  # [T13] activation steering at inference
    layer: int
    coeff: float


@dataclass(frozen=True, slots=True)
class DecodingTime:  # [T14] decoding-time control (FUDGE / GeDi / DExperts / ...)
    guidance_strength: float


@dataclass(frozen=True, slots=True)
class TaskVector:  # [T15] task vectors / model arithmetic
    scaling: float


# The sum type. 3.10-safe runtime union alias (X | Y); on 3.12+ prefer `type Method = ...`.
Method = ControlToken | LoRA | Hypernet | DPO | GRPO | Steering | DecodingTime | TaskVector


def describe(m: Method) -> str:
    """Total, exhaustive one-line description of a ``Method`` (match + assert_never)."""
    match m:
        case ControlToken(scheme=s):
            return f"[T01] control-token scheme={s!r}"
        case LoRA(rank=r, alpha=a, target_modules=tm):
            return f"[T03] LoRA r={r} alpha={a} on {','.join(tm)}"
        case Hypernet(typology_dim=t, generated_rank=r):
            return f"[T07] hypernet typology_dim={t} -> rank-{r} adapter"
        case DPO(beta=b, n_pairs=n):
            return f"[T09] DPO beta={b} over {n} pairs"
        case GRPO(group_size=g, kl_coeff=k):
            return f"[T10] GRPO group={g} kl={k}"
        case Steering(layer=layer, coeff=c):
            return f"[T13] activation-steering layer={layer} coeff={c}"
        case DecodingTime(guidance_strength=g):
            return f"[T14] decoding-time guidance={g}"
        case TaskVector(scaling=sc):
            return f"[T15] task-vector scaling={sc}"
        case _:
            assert_never(m)


if __name__ == "__main__":  # boundary check: describe is total; illegal fields/values raise
    methods: tuple[Method, ...] = (
        ControlToken(scheme="prefix-special-tokens"),
        LoRA(rank=8, alpha=16, target_modules=("q_proj", "v_proj")),
        Hypernet(typology_dim=289, generated_rank=8),
        DPO(beta=0.1, n_pairs=10_000),
        GRPO(group_size=8, kl_coeff=0.04),
        Steering(layer=12, coeff=4.0),
        DecodingTime(guidance_strength=2.0),
        TaskVector(scaling=0.5),
    )
    for method in methods:
        print(describe(method))
    try:
        Hypernet(rank=8)  # type: ignore[call-arg]  # 'rank' is a LoRA field, not a Hypernet one
    except TypeError as e:
        print(f"OK: illegal state unrepresentable -> {e}")
    else:
        raise AssertionError("expected Hypernet(rank=...) to raise TypeError")
    try:
        LoRA(rank=0, alpha=16, target_modules=("q_proj",))  # value invariant
    except ValueError as e:
        print(f"OK: bad LoRA value rejected -> {e}")
    else:
        raise AssertionError("expected LoRA(rank=0) to raise ValueError")
    print("OK: describe() total over all 8 methods")

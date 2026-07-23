"""Render a run's trainer_state.json into figures/progress.png.

One figure per run: loss vs step, lr vs step, grad_norm vs step, and — when the
run spans more than one epoch — per-epoch mean/final loss. Kept import-light:
matplotlib is imported inside the function so the training path never pays for
(or fails on) a plotting dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def plot_progress(trainer_state: Path, out_png: Path) -> Optional[Path]:
    """Write the progress figure; returns the path, or None when there is
    nothing to plot (empty history) or matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [plot] matplotlib unavailable ({e}) — skipping progress.png")
        return None

    state = json.loads(Path(trainer_state).read_text())
    hist = state.get("log_history") or []
    if not hist:
        print("  [plot] empty log_history — nothing to plot")
        return None

    steps = [h["step"] for h in hist]
    loss = [h.get("loss") for h in hist]
    lr = [h.get("lr") for h in hist]
    gn = [h.get("grad_norm") for h in hist]
    epochs_log = state.get("epochs_log") or []
    multi_epoch = len(epochs_log) > 1

    n_rows = 2 if multi_epoch else 1
    fig, axes = plt.subplots(n_rows, 3 if not multi_epoch else 2,
                             figsize=(12, 4 * n_rows), squeeze=False)
    if multi_epoch:
        ax_loss, ax_lr = axes[0]
        ax_gn, ax_ep = axes[1]
    else:
        ax_loss, ax_lr, ax_gn = axes[0]
        ax_ep = None

    ax_loss.plot(steps, loss, lw=0.8, label="train")
    # held-out eval curve overlaid on the SAME axis (train-vs-eval at a glance);
    # only present when the run logged eval at its cadence.
    ev = [(h["step"], h["eval_loss"]) for h in hist if h.get("eval_loss") is not None]
    if ev:
        ax_loss.plot([s for s, _ in ev], [v for _, v in ev], "o-", ms=3,
                     color="tab:orange", label="eval (held-out)")
        ax_loss.legend()
    ax_loss.set(title="loss", xlabel="step", ylabel="loss")
    if any(v is not None for v in lr):
        ax_lr.plot(steps, lr, lw=0.8, color="tab:green")
    ax_lr.set(title="learning rate", xlabel="step", ylabel="lr")
    if any(v is not None for v in gn):
        ax_gn.plot(steps, gn, lw=0.8, color="tab:red")
    ax_gn.set(title="grad norm (head)", xlabel="step", ylabel="||g||2")
    if ax_ep is not None:
        es = [e["epoch"] for e in epochs_log]
        ax_ep.plot(es, [e["mean_loss"] for e in epochs_log], "o-", label="mean")
        ax_ep.plot(es, [e["final_loss"] for e in epochs_log], "s--", label="final")
        ax_ep.set(title="per-epoch loss", xlabel="epoch", ylabel="loss")
        ax_ep.legend()
    for row in axes:
        for ax in row:
            ax.grid(alpha=0.3)
    fig.suptitle(f"{Path(trainer_state).parent.parent.name} — "
                 f"{state.get('n_steps', len(hist))} steps, "
                 f"final loss {state.get('final_loss')}")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  [plot] wrote {out_png} ({out_png.stat().st_size} bytes)")
    return out_png

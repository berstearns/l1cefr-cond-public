# l1cefr-cond-public

Public **code-only mirror** of the `l1cefr-cond` research repo — it exists so the
project's Colab notebooks can `pip install` the package without credentials.

- **Generated, not developed here.** Pushed by `scripts/sync_public_mirror.sh` as a
  single orphan snapshot. Pull requests against this repo will be overwritten by
  the next sync; it carries no history.
- **Contents:** `l1cefr_cond/` (notebook/runner/broker + LM-head conditioning
  strategies) and `domain/` (the algebraic L1 × CEFR domain model).
- Research specs, data, notebooks and experiment outputs are **not** mirrored.

```bash
pip install -e .
python -c "import l1cefr_cond; print(l1cefr_cond.__file__)"
```

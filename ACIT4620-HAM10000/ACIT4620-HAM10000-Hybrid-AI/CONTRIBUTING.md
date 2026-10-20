# Team contribution rules

1. **Issue first:** define goal, owner, evidence and acceptance criteria.
2. **Branch:** create `feature/<topic>`, `fix/<topic>` or `docs/<topic>` from updated `main`.
3. **Small changes:** commit related changes only; use messages like `feat: add manifest overlap validation`.
4. **Local quality:** run `python -m pip install -e '.[dev]'`, `ruff check src tests scripts` and `pytest`.
5. **Pull request:** link an issue, document how work was tested, request **at least one teammate review**.
6. **Merge:** merge after CI passes and review is approved. Avoid direct pushes to protected `main`.
7. **ML reproducibility:** record dataset source/version, lesion-disjoint manifest hash, seed, model weights source, hyperparameters, commit SHA and metrics. Store large model files outside Git.
8. **Evaluation integrity:** tune on training/validation only. Use the held-out test once for final model selection/evaluation reporting, not for repeated adjustment.
9. **Medical data:** do not upload raw images, derived patient-level files or access tokens. A demonstration is **not a clinical diagnosis**.

## Branch examples
- `feature/full-image-audit`
- `feature/efficientnet-baseline`
- `feature/class-weighted-loss`
- `feature/mamdani-rules`
- `feature/evaluation-report`

## Running the checks

```bash
python -m pip install -e '.[dev]'
ruff check src tests scripts
pytest
python scripts/validate_manifest.py /path/to/ham10000_master_manifest.csv
```

The unit tests use **synthetic** image/lesion metadata, so the CI does not need to download HAM10000 or run GPU training.

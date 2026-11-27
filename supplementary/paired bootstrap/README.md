# Supplementary paired lesion-grouped bootstrap

This is a post-experiment analysis of the original saved held-out predictions
from the November 18, 2026 run. The original archive is not modified.
No models are retrained, predictions regenerated, or hyperparameters selected.

## Reproduce

Use Python 3 with NumPy and pandas installed:

```bash
python paired_lesion_bootstrap.py "ACIT4620_09_EVIDENCE_COMPLETE_reports(1).zip" --output-dir .
```

The path to the original evidence ZIP may be absolute or relative. The script
checks the original manifest SHA-256, partition sizes, lesion disjointness,
prediction identity, saved probabilities, and recorded macro-F1 values.
It uses the same randomly resampled lesion groups for all three classifiers,
with replacement (2,000 draws of 1,068 groups, seed 42). Macro-F1 is averaged
over the seven diagnostic classes, with zero contribution for undefined
class F1. Percentile confidence limits are the 2.5th and 97.5th percentiles.

The CSV contains observed differences and confidence limits. The JSON records
the input archive hash and supplementary analysis environment, NOT the
original Kaggle training environment.

Both intervals include zero. This analysis is conditional on the fixed fitted
models and internal held-out dataset; it does not establish superiority,
training-seed variability, or external clinical generalization.

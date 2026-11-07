# Reusable project code

`ham10000_hybrid/data_validation.py` provides manifest contract checks for seven labels, image ID uniqueness, lesion-label consistency and lesion-disjoint splits.

`ham10000_hybrid/metrics.py` computes classification metrics for the seven classes and melanoma recall.

CNN training is provided in `notebooks/ACIT4620_02_EfficientNetB0_7Class_Training.ipynb`,
with `scripts/train_local.py` as a local entry point. See `docs/local_training.md`.
The Mamdani fuzzy inference engine is not yet implemented.

`ham10000_hybrid/evaluation.py` provides probability/calibration summaries,
label-blind review-priority scores, matched-workload referral metrics, external
fuzzy-score alignment and class-stratified lesion bootstrap intervals.
`scripts/evaluate_review.py` evaluates saved validation predictions and exports
CSV, JSON, PNG and SVG artifacts. See [the evaluation guide](../docs/evaluation_guide.md).
Fuzzy inference remains a separate, pending implementation; its real scores can
be connected through the documented CSV contract.

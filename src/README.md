# Reusable project code

`ham10000_hybrid/data_validation.py` provides manifest contract checks for seven labels, image ID uniqueness, lesion-label consistency and lesion-disjoint splits.

`ham10000_hybrid/metrics.py` computes classification metrics for the seven classes and melanoma recall.

CNN training is provided in `notebooks/ACIT4620_02_EfficientNetB0_7Class_Training.ipynb`,
with `scripts/train_local.py` as a local entry point. See `docs/local_training.md`.
The Mamdani fuzzy inference engine is not yet implemented.

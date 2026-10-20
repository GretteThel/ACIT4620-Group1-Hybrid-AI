# Reusable project code

`ham10000_hybrid/data_validation.py` provides manifest contract checks for seven labels, image ID uniqueness, lesion-label consistency and lesion-disjoint splits.

`ham10000_hybrid/metrics.py` computes classification metrics for the seven classes and melanoma recall.

CNN training and the Mamdani fuzzy inference engine are **not yet implemented**; add them as separately reviewed modules and connect them to the same fixed split.

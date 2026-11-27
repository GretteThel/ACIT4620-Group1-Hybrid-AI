# Selected results from the final ACIT4620 Hybrid AI run

These files were **copied without modification** from `ACIT4620_09_EVIDENCE_COMPLETE_reports.zip`, except for the explicitly labelled convenience summary `cv_mean_std_readable.csv`, which was formatted directly from `cv/end_to_end_cv_results.csv`.

The corresponding research notebook is `ACIT4620_Group1_Hybrid_AI.ipynb` in the repository root. The original archive, trained-model weights, exhaustive per-image predictions, raw quality audit and intermediate stage backups should be kept **outside Git**.

## Organization
- `figures/eda`: training-only exploratory analysis and image-quality illustrations.
- `figures/data_centric`: training-only augmentations, ROI audit and short screening comparisons.
- `figures/training`: CNN learning curves and learning-rate schedules.
- `figures/model_comparison`: final validation comparisons, test confusion matrices, ROC/PR and reliability plots for the **three** models.
- `figures/cross_validation`: fold trends and mean ± sample-SD plots.
- `figures/explainability`: qualitative Grad-CAM; heatmaps are **not** clinical interpretations.
- `figures/fuzzy_rules`: first-order Sugeno rule-utilization and rule-count internal screen.
- `tables`: recorded result tables and training histories; `results.csv` contains validation and test rows and should be filtered by the `split` column before comparisons.
- `provenance`: actual experiment configuration, split metadata and one-time held-out-test marker.

## Evaluation boundaries
The test set is a **lesion-disjoint internal HAM10000 holdout**, not external clinical validation or the official hidden challenge test. Do not use these held-out results to choose a different model or hyperparameters. The 4-epoch data-policy and training-only rule-count screens do not establish that an alternative policy/rule count is superior under the full run.

Dataset: Tschandl et al. (2018), *Scientific Data*, https://doi.org/10.1038/sdata.2018.161.

## Supplementary Statistical Analysis

The original experimental results remain unchanged.
A supplementary paired lesion-grouped bootstrap analysis
was conducted using the saved held-out predictions to
quantify uncertainty in macro-F1 differences between
classifiers.

The analysis code, confidence intervals, provenance,
and reproduction instructions are available in
[Supplementary Bootstrap Analysis](../supplementary/paired_bootstrap/README.md).

The original evidence archive,
`ACIT4620_09_EVIDENCE_COMPLETE_reports.zip`,
contains the complete split manifest, prediction records,
training histories, and experimental outputs.

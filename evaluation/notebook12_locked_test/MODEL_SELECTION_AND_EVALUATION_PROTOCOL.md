# ACIT4620 Group 1 — pre-test model-selection and evaluation protocol

**Status:** Prepared from existing *validation-only* experiments. **Not yet fully frozen until `locked_test.py freeze` successfully validates your original checkpoint files and creates the signed-off JSON record on your computer.** No test scores have been produced in this package.

## Original data and provenance

- HAM10000, **10,015 images**, 7 classes; class order `nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`.
- Original **lesion-disjoint** split: training **7,116**, validation **1,441**, held-out test **1,458** images.
- Split identity from earlier experiment metadata: `265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0`.
- Original E2a-50 CNN checkpoint SHA-256: `2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9`.
- Original split manifest is packaged in `data/sealed_split_manifest.zip` with the original E2a-50 output `split_manifest_summary.csv`. It is **not opened during the freeze operation**; the source bytes have a pinned SHA-256. No replacement splits permitted.

## Pre-registered primary test comparison

Selection was completed on 1,441 **validation** images and recorded in Notebook 09 (`validation_selected_checkpoints.csv`). For each architecture or ANFIS rule count, the saved checkpoint with the highest validation macro-F1 was selected. This is **not** a claim of test performance.

| Pre-selected model | Validation macro-F1 | Validation balanced accuracy | Validation melanoma recall |
|---|---:|---:|---:|
| CNN (E2a-50) | 0.5500 | 0.6217 | 0.4315 |
| Linear, seed 42 | 0.5455 | 0.6317 | 0.4452 |
| MLP, seed 42 | 0.5749 | 0.5832 | 0.4247 |
| ANFIS 4 rules, seed 52 | 0.5499 | 0.5748 | 0.4315 |
| ANFIS 8 rules, seed 42 | 0.5588 | 0.5989 | 0.3699 |
| ANFIS 16 rules, seed 52 | 0.5676 | 0.6205 | 0.4521 |

**Primary comparisons:** descriptive metrics across these six models, emphasizing **ANFIS 16 versus MLP** (the strongest validation macro-F1) and **ANFIS 16 versus CNN** (the deep-learning baseline). The 16-rule ANFIS is the strongest *individual ANFIS validation checkpoint*, not necessarily the highest three-seed mean.

**Secondary fixed sensitivity analysis:** one-time evaluation of all 16 original checkpoints: CNN (1), Linear (3 seeds), MLP (3 seeds), ANFIS 4 (3), ANFIS 8 (3), ANFIS 16 (3). This provides three-seed family means and standard deviations, **without selecting a new winner from the test set**. All 16 weights are checked and frozen before the test is opened.

## Model and preprocessing controls

The exact saved, frozen E2a-50 EfficientNet-B0 model extracts 1,280 pooled features. Input processing is `Resize(256) → CenterCrop(224) → RGB tensor → ImageNet Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])`. The CNN head predicts 7 logits directly; Linear, MLP and ANFIS heads operate on the same pooled features standardized using the **original training-only** 1,280-D scaler. E3 and E4 scalers must match numerically. The ANFIS uses a linear `1280→8` projection, Gaussian membership functions, normalized rule firing and first-order multi-output Sugeno consequents.

Test predictions are **original-view uncalibrated softmax**. No TTA, test-derived temperature scaling, changed decision thresholds, augmentation, retraining, retrained scalers, new hyperparameter search, or test-derived model selection is allowed. Inference may run on CPU or CUDA; same pretrained weights and deterministic evaluation mode.

## Metrics and statistical plan

- Classification: macro-F1 (**primary headline metric**), balanced accuracy, accuracy, weighted-F1, and per-class precision, recall, F1 and support.
- Melanoma: sensitivity (recall), precision, F1, true positives and false negatives. **Seven-class melanoma recall is not a binary clinical diagnostic sensitivity estimate**.
- Probabilistic: cross-entropy/NLL, multiclass Brier score, macro one-vs-rest AUROC and mean per-class average precision; undefined scores reported as unavailable.
- Figures: six confusion matrices, macro-F1 bar chart, per-class recall heatmap.
- Statistical uncertainty: **2,000 paired lesion-cluster bootstrap resamples**, fixed seed **20261110**, 95% percentile intervals for primary model macro-F1 and melanoma recall; paired differences *ANFIS16–MLP* and *ANFIS16–CNN*. Lesions, not individual images, are the resampling units. These CIs are descriptive; no unadjusted claim of significance or superiority.
- Explanations: cite separately completed validation-only Grad-CAM, stability, and occlusion experiments. **The human verification of the 50-image acquisition-artifact subset is deferred; do not claim artifact-specific results.**

## Integrity and freeze gate

1. Validate the source Notebook09 selection file using its pinned SHA-256; check that all six pre-selected models are exactly as listed.
2. Verify the 16 original checkpoint files and the training-only E3/E4 scalers: saved metadata for class order, split hash, E2a-50 source CNN hash, rule count and seed must match. The CNN file hash is pinned.
3. Hash every checkpoint and both scalers. Hash the sealed manifest *archive bytes without opening the test labels*.
4. Write `outputs/pretest_freeze.json` containing the model-selection policy, all hashes, original absolute file paths, bootstrap plan and date. **Do not edit this file afterward.**
5. Team members review and approve the freeze record before running the test command. A readable sign-off template is in `TEAM_SIGNOFF_BEFORE_TEST.md`.
6. Only after sign-off, run `evaluate --confirm-locked-test`. The evaluator verifies the freeze record and all hashes, then opens the sealed manifest, checks lesion disjointness and 1,458 original JPEGs and evaluates the entire predefined model list **once**.
7. Save the original full 16-model outputs, six-model selected report, three-seed sensitivity table, confidence intervals, matrices, plots, file integrity hashes and metadata. No test-driven iteration afterward.

## Known limits

- Test performance from a single HAM10000 partition is not an external-clinic validation and should not be used for diagnosis.
- Validation-selected best checkpoints can overstate relative validation performance; the locked-test stage provides the held-out estimate.
- HAM10000 images, class distribution and artifacts are not representative of all clinical presentations.
- Same-CNN frozen-feature head comparisons isolate head choices **conditional on the selected CNN**; they are not equivalent to end-to-end retraining each architecture.
- The AI-assisted acquisition-artifact screen is *not human-verified* and its flags must not be described as observed artifact prevalence or explanation causality.

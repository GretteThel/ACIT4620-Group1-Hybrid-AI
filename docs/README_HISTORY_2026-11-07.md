# ACIT4620 — HAM10000 Hybrid AI

**Research prototype: Data-Centric Seven-Class Skin Lesion Classification using Pretrained EfficientNet-B0 and planned Neuro-Fuzzy Classification Correction.**

## Current status (7 November 2026)

- Dataset: HAM10000 / ISIC 2018 **Task 3 — Lesion Diagnosis** (10,015 dermoscopic images, 7 diagnosis categories).
- EDA has been run on Kaggle, and a **lesion-disjoint** train/validation/test assignment has been verified.
- Source notebook is provided in `notebooks/01_HAM10000_Seven_Class_Data_Audit.ipynb` (outputs cleared for version control; rerun on Kaggle to reproduce figures).
- **Local full audit completed:** all 10,015 images decoded without failures; two identical-content pairs stay within their original lesion groups, with zero cross-lesion identical-content hashes. See `docs/dataset_audit_findings.md` for provenance and split-reproduction limits.
- **Local CNN baseline completed:** EfficientNet-B0 completed 3 head epochs and 8 partial fine-tuning epochs on the fixed local split. Validation accuracy is 75.29%, macro-F1 0.4896, balanced accuracy 46.69%, and melanoma recall 21.23%. Checkpoint, predictions, plots and comparison with the initial 3-epoch run are saved locally in `reports/local_runs/20261107_cnn_baseline_full/` (excluded from Git). See the [local runner guide](docs/local_training.md). These are validation/model-selection results; final test evaluation remains pending.
- **Additional fine-tuning completed:** 8 more epochs from the baseline weights, with a fresh optimizer, reached validation accuracy 77.65%, macro-F1 0.5328, balanced accuracy 51.35%, and melanoma recall 30.82%. The new run is preserved in `reports/local_runs/20261107_cnn_finetune_continued/`. Three-run comparison CSVs and PNG/SVG figures are in `reports/comparisons/20261107_cnn_three_runs/`. Existing run folders were retained; test evaluation remains pending.
- **Class-weight M2 trial completed:** seed 42, same maximum 3+8 epoch budget and stopping rule. The run stopped after 7 epochs and selected head epoch 3. Melanoma recall improved from 21.23% to 34.25%, while accuracy fell from 75.29% to 53.57% and macro-F1 from 0.4896 to 0.3778. This setting is not an overall improvement under the primary macro-F1 criterion. See `reports/classifier_comparisons/20261107_weighted_vs_ce_seed42/` locally.
- **Primary hybrid target clarified:** a neuro-fuzzy stage should output corrected final seven-class predictions. The [classifier comparison guide](docs/neuro_fuzzy_evaluation.md) and evaluator support actual final decisions and paired lesion difference intervals. The neuro-fuzzy classifier itself and its real predictions are still pending; no hybrid improvement has been established.
- **Longer local training completed:** both CNN branches ran 16 additional epochs with patience 12. The selected ordinary CNN reached accuracy 78.28%, macro-F1 0.5402 and melanoma recall 32.88%; its macro-F1 increase over the previous best is small (+0.00739; conditional paired 95% interval -0.01441 to +0.02814). The weighted branch improved over its own source to macro-F1 0.4412, accuracy 63.64%, balanced accuracy 62.43% and melanoma recall 38.36%, but remains below the ordinary CNN on the primary macro-F1 criterion. Five-run CSVs and PNG/SVG plots are in `reports/comparisons/20261107_extended_training_five_runs/`. These are single-seed validation results; no final-test or neuro-fuzzy gain has been established.
- **Not yet completed:** repeated-training evidence, final model/hybrid selection and held-out test evaluation. Earlier review-priority analyses remain a secondary experiment.
- **Evaluation code ready:** [validation/review evaluation guide](docs/evaluation_guide.md). Saved predictions support melanoma PR/ROC analysis, calibration summaries, lesion bootstrap intervals and matched-workload review comparisons. Real fuzzy scores can be connected by image ID plus checkpoint/split hashes; no fuzzy results are fabricated. The first report is saved locally in `reports/evaluation/20261107_cnn_review_baselines/`.

## Classification labels

| Index | Code | Diagnosis | Total images |
|---:|---|---|---:|
| 0 | `nv` | Melanocytic nevi | 6,705 |
| 1 | `mel` | Melanoma | 1,113 |
| 2 | `bkl` | Benign keratosis-like lesions | 1,099 |
| 3 | `bcc` | Basal cell carcinoma | 514 |
| 4 | `akiec` | Actinic keratoses / intraepithelial carcinoma | 327 |
| 5 | `vasc` | Vascular lesions | 142 |
| 6 | `df` | Dermatofibroma | 115 |

These are **seven diagnoses**, not seven cancers. This is not a clinically validated diagnostic tool.

## Dataset split (already verified on Kaggle)

| Set | Images | Distinct lesions |
|---|---:|---:|
| Train | 7,116 | 5,335 |
| Validation | 1,441 | 1,067 |
| Test | 1,458 | 1,068 |

Split method: `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)` grouping by `lesion_id`; fold 0 is validation, fold 1 is testing, other folds form training. There is **zero overlap in lesion IDs**. Patient-level disjointness cannot be verified because HAM10000 does not provide reliable patient IDs. All model variants must use the **same split**.

## Proposed experiment sequence

1. Complete data-centric audit: image integrity, repeated-lesion grouping, metadata quality, sample and full-image QC, leakage analysis, and exact-content duplicate checks.
2. Train seven-output pretrained EfficientNet-B0 with training-only augmentations. Run baseline with regular cross-entropy.
3. Compare data-centric alternatives, e.g. careful minority-class augmentation/resampling; also test class-weighted cross-entropy **as an optimization choice**. Never change validation/test distributions.
4. Select/configure model using **validation only** and assess per-class errors, calibration, and subgroup effects.
5. Implement the neuro-fuzzy final-class corrector and compare the same frozen CNN with and without it. Measure actual classification differences and both corrected/new errors. Optional Mamdani review-priority comparisons remain separate.
6. Freeze the full configuration before final held-out test evaluation; report classification metrics, paired comparisons and limitations. Never choose further model settings from test results.

Neuro-fuzzy classification improvement is a hypothesis to test. A review-priority score alone does **not** demonstrate classification improvement; it needs separate review-workload evaluation.

## Repo layout

```text
notebooks/01_HAM10000_Seven_Class_Data_Audit.ipynb  EDA + seven-class manifests
src/                                             Future training and fuzzy modules
docs/dataset_audit_findings.md                   Verified EDA observations
docs/kaggle_setup.md                            Kaggle run guide
docs/methodology.md                             Experiment and evaluation protocol
docs/team_workflow.md                           Collaboration rules
requirements.txt                                Python packages
.gitignore                                      Excludes images, checkpoints and local data
```

## Reproduce the inspection

1. Open Kaggle dataset: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Create/import the notebook in `notebooks/` and attach that Kaggle dataset via **Add Input**.
3. In Section 1, set `RUN_FULL_IMAGE_AUDIT = True` and `RUN_EXACT_HASH_AUDIT = True` for the final complete audit (may take time).
4. Run cells from top to bottom. Confirm that `ham10000_audit/` and `ham10000_7class/` exist under `/kaggle/working`.
5. Save a completed Kaggle version and verify its outputs persist.
6. Record the run's version/URL and any QC exceptions in `docs/dataset_audit_findings.md`.

## Data and licensing

**Do not commit raw HAM10000 images or Kaggle CSV mirror data to GitHub.** Each contributor obtains the dataset directly from its official source and follows the relevant terms, including attribution and non-commercial restrictions where applicable. Repository content is intended for coursework; review project and dataset licensing before any public release.

- Original dataset: Tschandl, P., Rosendahl, C., & Kittler, H. (2018), *The HAM10000 dataset*, Scientific Data. https://doi.org/10.1038/sdata.2018.161
- ISIC challenge: Codella et al., *Skin Lesion Analysis Toward Melanoma Detection 2018* (2019 preprint): https://arxiv.org/abs/1902.03368
- ISIC challenge portal: https://challenge.isic-archive.com/

## Team contributions

Create a GitHub issue for each concrete task and submit changes on a branch via pull request. See `docs/team_workflow.md`.

## Software engineering practices

This repository also includes a lightweight, tested Python package (`src/ham10000_hybrid/`), synthetic-data unit tests, GitHub Actions CI, issue and pull-request templates, and an architecture decision record. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

**Start locally:**

```bash
python -m pip install -e '.[dev]'
ruff check src tests scripts
pytest
```

**GitHub placement:** `.github/workflows/ci.yml` works as-is only when this project is its **own GitHub repository root**. If the project lives inside a larger `Masters/` repository, move this file to that repository's root `.github/workflows/` and point its job to this subdirectory.

The validation and metric modules are reusable helpers. EfficientNet training code is
available in Notebook 02 and the local runner; successful dataset training requires
the saved manifests, original images and completed audit. Mamdani inference remains planned.

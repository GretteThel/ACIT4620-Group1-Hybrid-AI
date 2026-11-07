# ACIT4620 — HAM10000 Hybrid AI

**Research prototype: Data-Centric Seven-Class Skin Lesion Classification using Pretrained EfficientNet-B0 and Mamdani Fuzzy Inference.**

## Current status (7 November 2026)

- Dataset: HAM10000 / ISIC 2018 **Task 3 — Lesion Diagnosis** (10,015 dermoscopic images, 7 diagnosis categories).
- EDA has been run on Kaggle, and a **lesion-disjoint** train/validation/test assignment has been verified.
- Source notebook is provided in `notebooks/01_HAM10000_Seven_Class_Data_Audit.ipynb` (outputs cleared for version control; rerun on Kaggle to reproduce figures).
- **Local full audit completed:** all 10,015 images decoded without failures; two identical-content pairs stay within their original lesion groups, with zero cross-lesion identical-content hashes. See `docs/dataset_audit_findings.md` for provenance and split-reproduction limits.
- **Local CNN training verified:** EfficientNet-B0 completed an initial 3-epoch run on the locally reproduced split. Checkpoint, validation metrics, predictions, plots and a result summary are saved locally in `reports/local_runs/20261107_initial_cnn/` (excluded from Git). See the [local runner guide](docs/local_training.md). This is an initial run; final test evaluation remains pending.
- **Not yet implemented:** fuzzy inference. Hyperparameter selection and final test-set evaluation remain pending experiments.

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
5. Implement Mamdani fuzzy review-priority layer using defensible uncertainty/quality measures. Compare against basic probability thresholds at matched review budget.
6. Evaluate final configuration **once on held-out test data**; report balanced accuracy, macro-F1, confusion matrix, per-class precision/recall, melanoma sensitivity, ROC/PR analyses where appropriate, and fuzzy review sensitivity/workload.

The Mamdani layer does **not necessarily improve CNN classification accuracy**; its review-decision value must be tested separately.

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



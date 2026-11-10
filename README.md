# ACIT4620 — Group 1 Hybrid AI | HAM10000 (7 classes)

**Research project:** Lesion-disjoint seven-class dermoscopic classification with a pretrained **EfficientNet-B0** backbone, data-centric CNN experiments and a **first-order Takagi–Sugeno CNN–ANFIS** classifier. The project includes baselines, matched validation comparison, explainability, efficiency benchmarking, a frozen locked-test protocol and a non-clinical Streamlit demonstration.

> **Research use only.** The seven output categories are not all cancers. Model probabilities are uncalibrated softmax scores, not clinical cancer-risk estimates. Neither Grad-CAM nor fuzzy-rule firing is diagnostic or causal evidence. The held-out test evaluation is **not reported** in this repository snapshot.

## Current integration status — 10 November 2026

| Workstream | Status | Repository location |
|---|---|---|
| Seven-class data audit and split | Implemented, verified lesion-disjoint | [`notebooks/`](notebooks/), [`docs/dataset_audit_findings.md`](docs/dataset_audit_findings.md) |
| Initial CNN and data-centric variants | Implemented; multiple separate CNN training runs | [`notebooks/`](notebooks/), [`scripts/`](scripts/) |
| E3 Linear / MLP frozen-feature baselines | Completed (3 seeds per head) | [`experiments/e3_linear_mlp/`](experiments/e3_linear_mlp/) |
| E4 multi-output Sugeno ANFIS | Completed (4, 8, 16 rules × 3 seeds) | [`experiments/e4_sugeno_anfis/`](experiments/e4_sugeno_anfis/) |
| E5 matched validation comparison | Completed; **no demonstrated ANFIS superiority** | [`experiments/e5_model_comparison/`](experiments/e5_model_comparison/) |
| E6 CNN–ANFIS explanations | Completed image-level validation analysis; artifact-label review pending | [`experiments/e6_cnn_anfis_explainability/`](experiments/e6_cnn_anfis_explainability/) |
| E7 computational efficiency | Completed benchmark outputs | [`experiments/e7_computational_efficiency/`](experiments/e7_computational_efficiency/) |
| Notebook 12 final locked-test evaluator | Code and local pre-test freeze prepared; **test pending group sign-off** | [`evaluation/notebook12_locked_test/`](evaluation/notebook12_locked_test/) |
| Streamlit demo | Implemented and validated on selected validation JPEGs | [`artifact/cnn_anfis_streamlit/`](artifact/cnn_anfis_streamlit/) |
| IEEE report | In progress | [`reports/`](reports/) |

**Integration note:** The repository contains both earlier local CNN training runs documented by teammates and the later E2a-50 frozen-backbone experiments. They are separate experimental histories; do **not** combine their metrics or imply that a model from one run produced predictions from the other. The former root README, preserved as [`docs/README_HISTORY_2026-11-07.md`](docs/README_HISTORY_2026-11-07.md), is historical and should not be used as the current status.

## Dataset and provenance

- **Dataset:** HAM10000, 10,015 dermoscopic images; ISIC 2018 Task 3 **diagnosis task** (not segmentation).
- **Seven-class index order:** `nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`.
- **Fixed split:** train **7,116**, validation **1,441**, internally held-out test **1,458** images; `lesion_id` groups do not overlap between partitions. Patient-level disjointness cannot be demonstrated from the available metadata.
- **Split SHA-256:** `265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0`.
- **Selected E2a-50 CNN checkpoint SHA-256:** `2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9`.
- The test set here is an internal lesion-disjoint holdout of HAM10000, **not the official hidden ISIC Challenge test set**.

See [`docs/dataset_audit_findings.md`](docs/dataset_audit_findings.md) and individual experiment READMEs for methods and exceptions.

## Notebook sequence and repository organization

| Notebook | What it covers | Location |
|---|---|---|
| 01 | Dataset audit and split | [`notebooks/`](notebooks/) |
| 02 | Original EfficientNet-B0 CNN training | [`notebooks/`](notebooks/) |
| 03, 03B, 03C, 03D | Class weighting, longer fine-tuning, E2a-50, recovery | [`notebooks/`](notebooks/) |
| 04 | Shared validation metrics | [`notebooks/`](notebooks/) |
| 05 | Rotation ablation (E2b) | [`notebooks/`](notebooks/) |
| 06 | Shared frozen EfficientNet features | [`notebooks/`](notebooks/) |
| 07 / E3 | Linear and MLP head baselines | [`experiments/e3_linear_mlp/`](experiments/e3_linear_mlp/) |
| 08 / E4 | Multi-output Sugeno ANFIS | [`experiments/e4_sugeno_anfis/`](experiments/e4_sugeno_anfis/) |
| 09 / E5 | Matched validation comparison and statistical analysis | [`experiments/e5_model_comparison/`](experiments/e5_model_comparison/) |
| 10 / E6 | ANFIS-aware Grad-CAM, stability, occlusion | [`experiments/e6_cnn_anfis_explainability/`](experiments/e6_cnn_anfis_explainability/) |
| 11 / E7 | Parameter, latency, training-step and sampled-RAM analysis | [`experiments/e7_computational_efficiency/`](experiments/e7_computational_efficiency/) |
| 12 | Frozen final test evaluation (not executed here) | [`evaluation/notebook12_locked_test/`](evaluation/notebook12_locked_test/) |

**E3–E7 are experiment IDs, not notebook numbers.** The repository additionally includes reusable package code in [`src/`](src/), a teammate's local training scripts in [`scripts/`](scripts/), shared tests in [`tests/`](tests/), configuration templates in [`configs/`](configs/) and team/project documents in [`docs/`](docs/).

## Selected-model architecture and validation anchors

**Dermoscopic image → EfficientNet-B0 → pooled 1,280-D features → train-only feature scaling → trainable 8-D projection → Gaussian fuzzy premises → 4/8/16 first-order Sugeno rules → seven logits → softmax.** The original CNN and classical Linear/MLP heads serve as comparators. No clinical calibration is claimed.

The six validation-selected *primary* configurations in Notebook 12 were fixed using validation macro-F1:

| Configuration | Validation macro-F1 |
|---|---:|
| CNN E2a-50 | 0.5500 |
| Linear seed 42 | 0.5455 |
| MLP seed 42 | **0.5749** |
| ANFIS R4 seed 52 | 0.5499 |
| ANFIS R8 seed 42 | 0.5588 |
| ANFIS R16 seed 52 | 0.5676 |

These are **validation** scores from [`E5`](experiments/e5_model_comparison/); they are neither locked-test outcomes nor evidence of ANFIS superiority. All 16 seed/family variants are included as predefined secondary sensitivity comparisons. See [`MODEL_SELECTION_AND_EVALUATION_PROTOCOL.md`](evaluation/notebook12_locked_test/MODEL_SELECTION_AND_EVALUATION_PROTOCOL.md).

## Reproducibility and run instructions

**General Python package and root tests** (Python 3.11+ recommended):

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests
```

Individual experiments have separate requirements, saved inputs and instructions; consult their own READMEs before running them. Results in separate training histories should not be assumed directly comparable. Do not rerun the locked test to improve scores.

**Streamlit research app:** follow [`artifact/cnn_anfis_streamlit/README.md`](artifact/cnn_anfis_streamlit/README.md). The repository contains its source and integrity checks, but **not** the locally restored checkpoints inside the app folder. Seven validation class examples have passed prediction-parity checks; this is not a seven-image model-accuracy study.

**Notebook 12:** code is published, but the repository snapshot **is not a self-contained fresh-clone reproduction of the freeze**. Original pinned validation input files under `evaluation/notebook12_locked_test/data/`, the matching ANFIS implementation dependency and an authorized local sealed manifest must be restored and integrity-checked. Do not fabricate replacement files, modify the already created local freeze, or expose sealed test labels. See [`docs/REPOSITORY_AUDIT_2026-11-10.md`](docs/REPOSITORY_AUDIT_2026-11-10.md).

## GitHub housekeeping and data use

- Keep the research code, validation metrics, figure captions and provenance; do not remove assets that affect a frozen checkpoint or result without a separately reviewed cleanup plan.
- This snapshot **does contain tracked model weights and some image-derived Grad-CAM overlays** inside experiment directories, despite exclusions for *new* raw-image or weight files in `.gitignore`. Review publication rights and model-asset policy before copying the repository to a public location. Do not claim it is source-code-only.
- No raw HAM10000 JPG dataset was included in the audited snapshot. Obtain authorized original images independently from [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and respect dataset licensing and attribution.
- Leave the original local pre-test selection freeze unchanged and obtain genuine team sign-off before any held-out evaluation. The acquisition-artifact human-review subgroup remains pending; automated artifact suggestions are not ground truth.

**Further guidance:** [`docs/REPOSITORY_AUDIT_2026-11-10.md`](docs/REPOSITORY_AUDIT_2026-11-10.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/team_workflow.md`](docs/team_workflow.md).

**Dataset reference:** Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset*. *Scientific Data*. https://doi.org/10.1038/sdata.2018.161

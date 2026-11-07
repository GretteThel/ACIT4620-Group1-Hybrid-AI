# Local EfficientNet-B0 training

The local runner executes the training cells from
`notebooks/ACIT4620_02_EfficientNetB0_7Class_Training.ipynb` with local paths.
It retains the notebook's ordinary cross-entropy, three head epochs and eight
partial fine-tuning epochs, audit gate, class order and saved split checks.
Checkpoint selection uses validation macro-F1. Test evaluation stays disabled.

## Environment (Windows PowerShell)

Use a project virtual environment with Python 3.10 or newer. For the RTX 5070 Ti
environment used here, CUDA 12.8 PyTorch wheels were selected from the
[official PyTorch index](https://download.pytorch.org/whl/cu128/).

```powershell
uv venv .venv --python 3.12
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e '.[eda,dev]' -c configs/local_training_constraints.txt
```

If `python` is unavailable, install it first. For a project-contained uv setup:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.python'
uv python install 3.12
```

Check pretrained-weight loading and a GPU forward/backward/optimizer step:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_local.py --check-environment
```

This uses synthetic tensors solely to check the environment. It does not train
on HAM10000 or produce model performance measurements. The first invocation
downloads ImageNet weights into `data/model_cache/` if they are not cached.

## Inputs

Download the original HAM10000 JPEGs and metadata, plus the completed saved
Notebook 01 output. Preserve the existing split assignments rather than making
a new split. A suitable local layout is:

```text
data/
  ham10000/
    HAM10000_metadata.csv
    HAM10000_images_part_1/*.jpg
    HAM10000_images_part_2/*.jpg
  notebook01/
    ham10000_7class/seven_class_master_manifest.csv
    ham10000_audit/audit_summary.json
```

If saved Kaggle outputs are unavailable, the following command downloads the
public original dataset (~5.2 GiB archive), extracts one original JPEG per image
ID, and reruns **the original Notebook 01** locally with full image and hash
audits enabled:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 scripts/prepare_local_data.py
```

This saves new Notebook 01 outputs to `data/notebook01/`. It records that the
assignments were reproduced locally rather than imported from Kaggle. A final
count check requires every class/split count to match the project record. The
local constraints pin scikit-learn 1.5.2: this reproduces the recorded counts,
whereas 1.9.1 changes the fold assignments. Matching counts alone cannot prove
identity to a missing original manifest; retain and reuse the newly saved
manifest and its fingerprint for all subsequent local runs.

An interrupted archive download is resumed on rerun. Extraction refuses to
overwrite existing image files. If extraction is already complete, use
`--skip-extraction` with a new `--audit-dir` to repeat only the audit.

The manifest must include `image_id`, `lesion_id`, `dx`, and `split`. The audit
must confirm full image/hash checks for all 10,015 images, zero decode failures
and zero lesion overlap, as required by the training notebook. Investigate any
duplicate-content findings before using the audit for training.

## Run and saved outputs

With the above layout, run:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_local.py
```

Or specify paths explicitly:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_local.py `
  --data-dir 'D:\datasets\ham10000' `
  --manifest 'D:\notebook01\ham10000_7class\seven_class_master_manifest.csv' `
  --audit-summary 'D:\notebook01\ham10000_audit\audit_summary.json'
```

Every invocation creates a new UTC timestamp folder under `reports/local_runs/`.
`--output-dir PATH` chooses a new explicit folder; existing folders are refused
to prevent overwriting results. The runner uses zero loader workers for Windows
compatibility with notebook-defined dataset classes.

All attempts save `run.log`, `run_request.json`, and `run_status.json`. Missing
inputs also produce `input_check.json` and a failed status without training.
Successful training additionally saves:

- `best_efficientnet_b0.pth`: best checkpoint and split fingerprint.
- `training_history.csv` and training plots.
- `validation_metrics.json`, classification report and confusion matrix plot.
- Validation predictions (`.csv`, `.npz`) and error analysis.
- Calibration settings and `experiment_config.json`.

Training data, pretrained weights, checkpoints and local output folders are
ignored by Git. An environment-check success is not a dataset-training success;
check `run_status.json` and `mode` before interpreting a run.

The first local run on 7 November 2026 used `--head-epochs 1 --finetune-epochs 2`
and saved outputs plus a Korean `RESULTS.md` summary to
`reports/local_runs/20261107_initial_cnn/`. It completed successfully; its
validation metrics are initial results, with no held-out test evaluation.

The subsequent baseline on the same date completed **3 head + 8 partial
fine-tuning epochs** using the same split, ordinary cross-entropy, seed,
augmentation and learning rates. It started again from ImageNet weights rather
than resuming the initial checkpoint. Outputs are in
`reports/local_runs/20261107_cnn_baseline_full/`; `RESULTS.md` contains the report
and `comparison_with_initial.csv` contains the comparison below.

| Validation metric | Initial 3 epochs | Full 11 epochs |
|---|---:|---:|
| Accuracy | 73.77% | 75.29% |
| Balanced accuracy | 40.36% | 46.69% |
| Macro-F1 | 0.4114 | 0.4896 |
| Melanoma recall | 15.07% | 21.23% |

The selected checkpoint is fine-tuning epoch 8, chosen by validation macro-F1.
Saved predictions were checked against the manifest and their recomputed metrics
matched the saved metrics. Melanoma recall remains low (31/146 validation images);
this baseline provides a reference for subsequent imbalance/loss experiments.
These are single-seed validation results, with no held-out test evaluation.

## Continue fine-tuning from a saved checkpoint

To run additional fine-tuning from a completed local experiment:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 scripts/train_local.py `
  --init-checkpoint reports/local_runs/20261107_cnn_baseline_full/best_efficientnet_b0.pth `
  --finetune-epochs 8 `
  --output-dir reports/local_runs/20261107_cnn_finetune_continued
```

With `--init-checkpoint`, head-only epochs default to zero. The runner validates
class order, split fingerprint and relevant settings against the completed source
run. It loads the saved **model weights** and creates fresh optimizer and RNG
states; this is a warm start, not exact optimizer-state resumption. The same
last two feature blocks and classifier are trained. The source checkpoint remains
untouched. Its validation score initializes checkpoint selection so further
training cannot silently discard the starting model in favor of a worse one.

`warm_start.json` records the parent path/hash, source epoch, cumulative epoch
offset and whether the final checkpoint came from the parent or the current run.
`initial_validation_metrics.json` saves the starting model's measured scores.
`runtime_environment.json` records the actual GPU and CUDA environment, and
`runner_source.py` snapshots the runner used for a new invocation.
On Windows, the process prevents automatic system sleep while training and
releases that request when it exits. The display can still sleep.

## Compare runs and redraw figures later

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_runs.py `
  reports/local_runs/20261107_initial_cnn `
  reports/local_runs/20261107_cnn_baseline_full `
  reports/local_runs/20261107_cnn_finetune_continued `
  --labels 'Initial 3 epochs' 'Baseline 11 epochs' 'Continued fine-tuning' `
  --output-dir reports/comparisons/20261107_cnn_three_runs
```

Comparison refuses unfinished runs, mismatched class order/split fingerprints,
different validation IDs/labels, invalid probabilities or metrics inconsistent
with saved predictions. Choose a new output folder for each comparison.

- `run_metrics.csv`: one row per run; original metric values on a 0–1 scale.
- `epoch_history.csv`: every recorded epoch, with local and checkpoint-lineage
  epoch numbers. A continuation's local epoch 1 is distinct from a fresh run's.
- `class_metrics.csv`: precision, recall, F1 and support per class and run.
- `comparison_metadata.json`: source paths, checkpoint hashes and configurations.
- PNG and editable SVG figures: aggregate metrics, local learning curves,
  class-wise recall and cumulative parent/continuation curves.
- `RESULTS.md`: linked comparison summary.

The lineage plot shows the parent only up to its selected checkpoint and marks
the new optimizer boundary. Different loss configurations should be considered
when comparing loss curves. All summaries remain validation/model-selection
results; the held-out test is left for final evaluation.

The 7 November continuation completed all **8 additional epochs**, starting from
the baseline's selected epoch 11 weights. Its selected checkpoint is local
fine-tuning epoch 8 (cumulative epoch 19 along this lineage). The source checkpoint
was unchanged, and checkpoint reload and saved-prediction metric checks passed.

| Validation metric | Baseline 11 epochs | After 8 additional epochs |
|---|---:|---:|
| Accuracy | 75.29% | 77.65% |
| Balanced accuracy | 46.69% | 51.35% |
| Macro-F1 | 0.4896 | 0.5328 |
| Melanoma recall | 21.23% | 30.82% |

The continuation report is at
`reports/local_runs/20261107_cnn_finetune_continued/RESULTS.md`. The three-run
comparison report, CSVs and PNG/SVG figures are at
`reports/comparisons/20261107_cnn_three_runs/`. The original 3-epoch and 11-epoch
run folders remain available. All reported scores are validation results.

## Controlled class-weight experiment

`--class-weights` enables the existing training-only inverse-frequency weights.
`--no-class-weights` keeps ordinary cross-entropy. `--seed` controls training RNG
only; the saved split is never regenerated. Run both arms from ImageNet weights
with the same maximum budget/settings, rather than warm-starting only one arm.
Validation cross-entropy is unweighted for both arms, so validation-loss plots
have a common definition. Training losses have different objectives and should
not be compared directly. `loss_configuration.json` records counts and weights.

```powershell
.\.venv\Scripts\python.exe -u -X utf8 scripts/train_local.py `
  --class-weights --seed 42 --head-epochs 3 --finetune-epochs 8 `
  --manifest data/notebook01/ham10000_7class/seven_class_master_manifest.csv `
  --audit-summary data/notebook01/ham10000_audit/audit_summary.json `
  --output-dir reports/local_runs/20261107_cnn_weighted_seed42
```

The 7 November M2 trial completed with the existing patience-4 rule: 3 head and
4 fine-tuning epochs ran, then head epoch 3 was selected by validation macro-F1.
The ordinary baseline ran all 11 epochs. The maximum budget and stopping rule
matched, but the actual executed epoch counts differ.

| Validation metric | Ordinary CE baseline | Inverse-frequency weighted CE |
|---|---:|---:|
| Accuracy | 75.29% | 53.57% |
| Balanced accuracy | 46.69% | 54.82% |
| Macro-F1 | 0.4896 | 0.3778 |
| Melanoma recall | 21.23% | 34.25% |
| Melanoma precision | 41.89% | 25.13% |

The weighting setting increases recall/balanced accuracy but is **not an overall
improvement under the predefined macro-F1 selection criterion**. It corrected
107 baseline errors and introduced 420; melanoma misses corrected/introduced
were 32/13. This is one training seed and does not establish a general verdict
on all class-weighting approaches. Do not replace the stronger saved CNN with
this model merely because one metric rose.

The run and its loss/weight provenance are in
`reports/local_runs/20261107_cnn_weighted_seed42/`. Learning-curve comparisons are
in `reports/comparisons/20261107_weighted_vs_ce_seed42/`, and paired difference
intervals are in `reports/classifier_comparisons/20261107_weighted_vs_ce_seed42/`.
Seed 43/44 repeats have not been run. This experiment does not test neuro-fuzzy
classification; see `docs/neuro_fuzzy_evaluation.md` for that distinct comparison.

## Longer local fine-tuning

`--patience` controls the number of consecutive non-improving epochs before a
stage stops (default 4, positive integer). Increasing `--finetune-epochs` alone
does not increase that stopping tolerance. The local runner records both values
in its request and saves the effective patience in `experiment_config.json`.

The 7 November follow-up uses up to 16 additional fine-tuning epochs and patience
12 for each of the previous strongest unweighted model and the weighted model.
Both keep the same saved split, seed, learning rates and augmentations. Source
weights are loaded with fresh optimizer/RNG states; existing run folders are
preserved. This is a continuation study with different starting checkpoints,
not a controlled estimate of the weighting treatment effect.

```powershell
.\.venv\Scripts\python.exe -u -X utf8 scripts/train_local.py `
  --init-checkpoint reports/local_runs/20261107_cnn_finetune_continued/best_efficientnet_b0.pth `
  --finetune-epochs 16 --patience 12 --seed 42 --no-class-weights `
  --output-dir reports/local_runs/20261107_cnn_extended_unweighted_16ep

.\.venv\Scripts\python.exe -u -X utf8 scripts/train_local.py `
  --init-checkpoint reports/local_runs/20261107_cnn_weighted_seed42/best_efficientnet_b0.pth `
  --finetune-epochs 16 --patience 12 --seed 42 --class-weights `
  --output-dir reports/local_runs/20261107_cnn_extended_weighted_16ep
```

The comparison uses macro-F1 as the selection criterion and reports other
metrics and error transitions as well. Retaining the original checkpoint means
no improvement, even though the selection rule protects its previous score.
Final testing remains disabled. The plan is saved in
`reports/experiment_plans/20261107_extended_training.json`.

Both extensions completed all 16 additional epochs on the RTX 5070 Ti. The
unweighted run selected local epoch 10 (checkpoint-lineage epoch 29); the weighted
run selected local epoch 13 (checkpoint-lineage epoch 16). The weighted source
was its selected head epoch 3, not the discarded last fine-tuning epoch. Both
source checkpoint hashes remain unchanged, selected weights reloaded, and saved
validation predictions reproduced the reported metrics.

| Validation metric | Previous best ordinary CNN | Extended ordinary CNN | Original weighted CNN | Extended weighted CNN |
|---|---:|---:|---:|---:|
| Accuracy | 77.65% | 78.28% | 53.57% | 63.64% |
| Balanced accuracy | 51.35% | 52.30% | 54.82% | 62.43% |
| Macro-F1 | 0.5328 | 0.5402 | 0.3778 | 0.4412 |
| Melanoma recall | 30.82% | 32.88% | 34.25% | 38.36% |

The ordinary extension's macro-F1 difference is +0.00739 with conditional paired
lesion bootstrap 95% interval [-0.01441, +0.02814]. This modest validation gain
does not establish a clear generalization improvement. The weighted extension
improves over its own source by +0.06338 [0.03116, 0.09967], but is below the
extended ordinary model by -0.09901 [-0.15044, -0.04506]. Keep the ordinary model
as the primary macro-F1 candidate. These intervals do not include training-seed
variance or remove optimism from selecting models on validation.

Five-run CSVs and PNG/SVG plots are in
`reports/comparisons/20261107_extended_training_five_runs/`; the run folders have
their own `RESULTS.md` and `artifact_verification.json`. The paired comparisons
are in `reports/classifier_comparisons/20261107_extended_*`. Neither branch
evaluated the final test set or implemented a neuro-fuzzy classifier.

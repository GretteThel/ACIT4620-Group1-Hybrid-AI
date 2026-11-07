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

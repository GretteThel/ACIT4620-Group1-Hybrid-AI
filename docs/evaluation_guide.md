# CNN and fuzzy review evaluation

**Scope clarification, 7 November 2026:** the primary hybrid now targets corrected
final seven-class predictions. Use [the classifier comparison guide](neuro_fuzzy_evaluation.md)
for CNN-versus-neuro-fuzzy classification and paired metric differences. This
page describes the separate, secondary review-priority analysis.

`scripts/evaluate_review.py` evaluates saved **validation** predictions. It does
not train a model, change class predictions, invent fuzzy results, or read test
predictions. The reusable calculations are in `src/ham10000_hybrid/evaluation.py`.

## Current melanoma findings and next training comparison

The continued CNN correctly classifies 45/146 validation melanoma images.
Its other melanoma predictions are `nv=61`, `bkl=26`, `bcc=7`, `akiec=5`, `df=2`.
This motivates inspecting the melanoma-to-nv/bkl errors and testing imbalance
handling; the confusion counts alone do not establish the cause of the errors.

The first controlled training comparison should use ordinary versus
class-weighted cross-entropy, with the **same initialization, split, seed,
augmentation and epoch budget**. The existing training notebook contains
`USE_CLASS_WEIGHTS`; weights are calculated from training counts only as
`N_train / (7 * class_count)`. Evaluate melanoma precision, recall and F1 together
with macro-F1/balanced accuracy; an increase in recall is not a guaranteed outcome.
Do not simultaneously add oversampling to the first weighting experiment, since
that makes the effect difficult to identify. No new weighted model was trained
as part of preparing the original review evaluation code. A subsequent seed-42
weighted-loss trial is documented in `docs/local_training.md`; it improved
melanoma recall but reduced macro-F1/accuracy and was not adopted as an overall
improvement.

Separately, a melanoma-probability threshold can flag cases for review. Its
precision/recall/workload trade-off is exported, but it is not automatically
chosen and does not replace the seven-class argmax classifier. More flagged
melanoma images are not automatically more correctly diagnosed images.

## Run the evaluation

Use the project environment with the EDA extras installed for plotting:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_review.py `
  reports/local_runs/20261107_initial_cnn `
  reports/local_runs/20261107_cnn_baseline_full `
  reports/local_runs/20261107_cnn_finetune_continued `
  --bootstrap-resamples 200 `
  --output-dir reports/evaluation/20261107_cnn_review_baselines
```

This invocation has been run successfully. Choose a new output folder for a new
invocation; existing folders are refused. Omit `--output-dir` for an automatically
timestamped folder. `--bootstrap-resamples 0` disables intervals. `--budgets`
accepts review fractions, for example `--budgets 0 .1 .2 .3 .5 1`.

Inputs are completed local-run folders with `experiment_config.json`,
`validation_predictions.npz`, `split_manifest_summary.csv`, saved metrics and
training history. The adapter reuses the existing run validation, checks class
order/split/IDs/labels, and recomputes metrics from saved probabilities. Earlier
NPZ files may contain object-typed lesion IDs: the evaluator never enables
pickle and instead obtains lesion IDs from the validated CSV manifest.

## Classification and probability metrics

- Existing seven-class accuracy, balanced accuracy, macro-F1 and per-class metrics.
- Melanoma precision, recall, F1, support and correctly classified count.
- One-vs-rest ROC-AUC and average precision (AP) for every class, with PR and ROC
  coordinates for melanoma. AP is scikit-learn average precision, not trapezoidal
  PR area. Undefined metrics are reported as missing, not zero successes.
- Multiclass NLL; Brier **sum** over seven classes averaged over images (0–2).
- Top-label ECE with 15 equally spaced bins and a reliability plot. This measures
  confidence versus argmax correctness, not melanoma-specific calibration.
- Optional 95% percentile intervals for the four main classification metrics,
  resampling lesions within each observed diagnosis class. All photos of a
  lesion receive the same resampling multiplicity. This is not patient-level
  resampling, does not estimate review-metric intervals, and does not remove the
  optimism from using validation for model selection/calibration.

Saved probabilities have already been temperature-scaled using the same
validation labels in these runs. Calibration metrics here are descriptive and
must not be presented as independent calibration-test evidence.

## Matched review workloads

Available baseline priority scores, all higher-is-more-review:

| Strategy | Score |
|---|---|
| `low_confidence` | `1 - max(p)` |
| `entropy` | Normalized seven-class predictive entropy |
| `small_margin` | `1 - (largest p - second-largest p)` |
| `mel_probability` | The CNN's melanoma probability |
| `tta_disagreement` | Normalized Jensen–Shannon disagreement across saved views |
| `fuzzy` | Optional real Mamdani review score supplied by the team |

For every requested fraction `b`, every strategy reviews exactly `floor(N*b)`
images. Actual fractions are included. Rank by descending score, then ascending
`image_id` to resolve ties **without using labels**. The cutoff and tie-group
size are exported, since coarse fuzzy scores may contain many ties. This is a
matched-rank budget comparison, not a claim that a single score threshold always
produces exactly the requested fraction on unseen data.

Key output definitions:

| Output | Definition |
|---|---|
| `error_capture_rate` | Existing CNN errors flagged / all existing CNN errors |
| `review_error_precision` | Existing CNN errors flagged / reviewed images |
| `melanoma_flag_rate` | True melanoma flagged / all true melanoma |
| `missed_melanoma_capture_rate` | CNN-missed melanoma flagged / all CNN-missed melanoma |
| `n_missed_melanoma_unreviewed` | CNN-missed melanoma still outside the review set |
| `missed_melanoma_fraction_of_unreviewed` | Those remaining missed melanoma / all unreviewed images |

A review flag never changes the CNN's predicted diagnosis or assumes that a
human reviewer successfully corrects it. Zero denominators produce undefined
rates; 0% and 100% review provide endpoint checks.

## Connect the real fuzzy output

Supply a CSV with these columns:

```text
run_id,image_id,review_score,checkpoint_sha256,split_sha256
```

`run_id` is the saved CNN run folder name. Include exactly one row for every
validation image of that run. Every row must use its selected checkpoint hash
and fixed split fingerprint; these are available in comparison metadata and
the local-run artifact/config files. Scores must be finite numbers; larger
values must mean higher review priority. The score need not be a probability.

The CSV can cover only the CNN run for which fuzzy results exist, even when
several CNN runs are evaluated. Missing/extra/duplicate IDs, stale checkpoint
hashes and wrong split fingerprints are rejected. The producer must also record
its rule/membership-function version and must not use ground truth as runtime
input; the CSV schema alone cannot verify how a score was constructed.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_review.py `
  reports/local_runs/20261107_cnn_finetune_continued `
  --fuzzy-scores path/to/real_validation_fuzzy_scores.csv
```

Without that CSV, metadata and the report explicitly mark fuzzy evaluation as
pending. Synthetic fuzzy scores appear only in isolated unit tests.

## Artifacts and final evaluation

The output folder contains classification/probability CSVs, melanoma threshold
sweeps and PR/ROC coordinates, review scores with exact ranks, matched-workload
curves, metric/interval JSON, provenance/configuration, a linked `RESULTS.md`,
and PNG/SVG plots. All generated artifacts are excluded from Git.

Before final testing, freeze the selected CNN, calibration, fuzzy rules and
review policy on validation. If deploying a fixed threshold, freeze that actual
threshold rather than selecting a test-label-dependent workload. The current
CLI intentionally reads validation exports only. Final test evaluation will be
a separate step after the full configuration is fixed.

References: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html),
[scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[scikit-learn precision–recall curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html).

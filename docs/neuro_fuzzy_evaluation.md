# Primary target: CNN + neuro-fuzzy final classification

The user clarified on 7 November 2026 that the hybrid should correct the CNN's
predictions and output a final one-of-seven diagnosis. Classification improvement
therefore needs a direct **CNN versus CNN + neuro-fuzzy** comparison. The earlier
Mamdani review-score evaluation is a secondary analysis; selecting cases for
review does not itself improve classification accuracy or melanoma recall.

The classifier architecture and fitting method are not yet implemented or
specified. This evaluation interface does not substitute a fabricated fuzzy
classifier. It accepts real final predictions from the eventual implementation.

## Fair comparison

1. Fix the dataset, saved lesion-disjoint assignments and class order.
2. Select and freeze the source CNN using validation only; record its checkpoint hash.
3. Fit any learned fuzzy correction parameters using training data only. A learned
   feature head can use frozen training features; probability stacking may need
   lesion-grouped training OOF predictions or an internal training-only holdout.
   Never use validation/test labels as the fitting set or runtime input.
4. Select architecture/hyperparameters on validation, retaining model/rule versions.
5. Compare final diagnoses from CNN alone and the same CNN plus the corrector on
   exactly the same images. A simple non-fuzzy correction head with the same inputs
   is a useful additional control.
6. Repeat training seeds where feasible, then freeze the full setup before final testing.

Primary metrics are macro-F1 and balanced accuracy; also report accuracy and all
class-wise precision/recall/F1/support, especially melanoma. Report degradation
as well as improvement. No universal performance threshold has been agreed.

## Final-prediction export contract

Create a CSV containing **exactly one row per validation image**, with `image_id`
and one of these alternatives:

- `predicted_dx`: a hard final label in the fixed seven diagnosis codes.
- All seven final probability columns, in order:
  `prob_nv,prob_mel,prob_bkl,prob_bcc,prob_akiec,prob_vasc,prob_df`.

If both are supplied, labels must equal the final probability argmax. Do not
attach the original CNN probabilities to changed hard labels and call those
the hybrid probabilities. Hard-label output is supported without inventing
one-hot confidence; probability/calibration metrics are then omitted.

Copy `configs/neuro_fuzzy_output_metadata_template.json` and fill its placeholders
with real provenance. The JSON identifies the source CNN run, checkpoint hash,
split fingerprint, class order, validation partition, training-fit partition
(`train` or `train_oof`) and a model/rule version. The evaluator checks these
declarations against the source run. This is not an independent audit of how the
producer actually fitted the hybrid.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_classifiers.py `
  --baseline-run reports/local_runs/SELECTED_CNN_RUN `
  --hybrid-predictions path/to/real_hybrid_validation_predictions.csv `
  --hybrid-metadata path/to/real_hybrid_metadata.json `
  --bootstrap-resamples 1000
```

No real hybrid CSV has been supplied yet, so no CNN-plus-neuro-fuzzy performance
claim has been made. Synthetic hybrid predictions exist only inside unit tests.

## Difference estimates and outputs

The same class-stratified lesion samples/multiplicities are applied to both
classifiers. The tool reports candidate-minus-baseline point differences and
95% percentile bootstrap intervals for accuracy, balanced accuracy, macro-F1
and melanoma recall. These are paired intervals for **differences**, unlike the
earlier evaluator's separate per-model intervals.

The resampling unit is a lesion, not an individual photograph or patient.
Intervals are conditional on these fitted models and validation cases, do not
include training-seed variance, and do not remove selection/calibration optimism.
An interval above zero is not a substitute for independent final-test evidence.

Exports include:

- `metric_differences.csv` and `paired_comparison.json`.
- `paired_image_predictions.csv`, including errors corrected and introduced.
- `class_metrics.csv`, full classification reports and optional probability metrics.
- Source/provenance metadata and PNG/SVG metric/difference and class-recall plots.

Use `--no-plots` for a core-dependency-only evaluation. Existing output folders
are refused. The CLI intentionally reads only saved validation exports.

## Reuse for CNN training ablations

The same evaluator can compare two real completed CNN runs:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/compare_classifiers.py `
  --baseline-run reports/local_runs/20261107_cnn_baseline_full `
  --candidate-run reports/local_runs/20261107_cnn_weighted_seed42 `
  --bootstrap-resamples 1000 `
  --output-dir reports/classifier_comparisons/20261107_weighted_vs_ce_seed42
```

This comparison is explicitly marked `cnn_training_variant`; it cannot be
presented as neuro-fuzzy improvement. Controlled-setting matches are recorded.
The first inverse-frequency weighting trial improved melanoma recall/balanced
accuracy but reduced accuracy/macro-F1. Its selected checkpoint was head epoch 3;
the run completed 3 head + 4 fine-tuning epochs before the pre-existing stopping
rule fired. Both arms had the same maximum budget and stopping rule, but not
the same actual executed epoch count. See the local-training guide for numbers.

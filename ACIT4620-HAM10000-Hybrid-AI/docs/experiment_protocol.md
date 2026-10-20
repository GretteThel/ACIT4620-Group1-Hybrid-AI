# Experiment protocol — agree before model training

## Scope
- HAM10000, ISIC 2018 Task 3 **training dataset**; seven diagnosis labels.
- Existing lesion-disjoint split: 7,116 train; 1,441 validation; 1,458 test.
- This is **not** the original official challenge test set.
- Patient-disjointness cannot be guaranteed from this metadata.

## Quality gates
- Record full-image-readability report and cross-image exact-content audit before final training.
- Audit any byte-identical images that cross different `image_id` or `lesion_id` values.
- Never augment validation/test images for training or resplit the test set after inspecting results.

## Fixed comparisons
- CNN-0: pretrained EfficientNet-B0, ordinary cross-entropy baseline.
- CNN-1: same CNN and split, controlled class-weight / data-centric intervention.
- CNN-2: model selection and confidence calibration on validation only.
- Hybrid: Mamdani review priority; compare to simple confidence/entropy thresholds at matched review workloads.

## Evaluation
- Dataset-wide: balanced accuracy, macro-F1, confusion matrix.
- Per-class: recall, precision, F1 and support; separately highlight melanoma recall.
- Probability estimates: multiclass calibration and ROC/PR summaries if suitable.
- Review system: review fraction, error capture, melanoma flagged, missed melanoma among non-referred cases.
- Report class/slice supports. Report variability across random seeds where feasible.

## Artifacts per experiment
Record: issue number, run name, date, owner, Git commit, dataset version, manifest SHA-256, seed, configuration, pretrained weight source, Kaggle notebook URL/version, metrics, known problems. Avoid committing image data or trained weight files.

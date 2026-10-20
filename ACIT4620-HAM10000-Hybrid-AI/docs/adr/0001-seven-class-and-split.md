# ADR 0001: Seven-class HAM10000 classification with fixed lesion-disjoint splits

**Status:** Accepted for project development, 2026-10-20

**Context:** The group considered binary melanoma-vs-nevus prediction, then selected the full seven-class ISIC 2018 Task 3 training dataset to support broader diagnosis analysis and stronger class-imbalance evaluation.

**Decision:** Train a seven-output pretrained EfficientNet-B0 on all 10,015 images, using the existing `lesion_id`-grouped 7,116/1,441/1,458 train/validation/test manifests. Keep evaluation focused on balanced accuracy, macro-F1, class-wise recall and melanoma sensitivity. Evaluate the Mamdani component as a distinct uncertainty/review-priority decision layer, not as an assumed boost in CNN accuracy.

**Consequences:** Greater imbalance and harder model selection than binary classification; a stronger need for per-class metrics. No claims of patient disjointness or clinical validity.

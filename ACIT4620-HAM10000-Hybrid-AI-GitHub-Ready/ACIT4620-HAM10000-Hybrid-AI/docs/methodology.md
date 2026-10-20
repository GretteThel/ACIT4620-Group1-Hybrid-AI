# Proposed methods and evaluation — seven-class project

## Goal
Investigate how *data-centric* image preparation and a *Mamdani fuzzy uncertainty-review decision layer* affect seven-class dermoscopic classification with ImageNet-pretrained EfficientNet-B0.

## Data-centric protocol
1. Keep data and diagnosis labels immutable until documented review.
2. Complete full image and content-duplicate audits. Repeated images of the same lesion are kept in a single partition.
3. Inspect class imbalance (majority `nv` vs underrepresented `df`, `vasc`, etc.).
4. Apply training-only image augmentations and compare candidate balancing approaches.
5. Analyze class- and subgroup-specific validation errors; record intervention rationale and version of training data.
6. Freeze the test set throughout development and comparisons.

## CNN
- Start from ImageNet-pretrained EfficientNet-B0.
- Replace original classifier with 7 output logits; use label order in Section 18's `class_mapping.json`.
- Start with cross-entropy; compare a fixed baseline against class-weighted cross-entropy and carefully justified targeted augmentations.
- Define transforms consistent with chosen pretrained weight API; no aggressive medically unjustified perturbations.
- Select checkpoint and decision settings on **validation data only**.

## Mamdani fuzzy system
- Design a separate **review-priority** output based on a defensible subset of: calibrated predictive uncertainty, top-class margin, prediction disagreement under test-time transformations, image-quality indicators.
- Note that entropy and top-two margin are derived from the same class-probability vector; do not claim they are statistically independent.
- Describe membership functions, rule base, min/max or chosen t-norm/s-norm, aggregation and centroid defuzzification.
- A fuzzy review score is **not equivalent to clinical risk probability** and does not necessarily improve diagnosis accuracy.

## Controlled comparison
- CNN baseline (same fixed splits).
- CNN + data-centric training alternative(s).
- Best selected CNN + simple confidence/uncertainty threshold for review.
- Best selected CNN + Mamdani fuzzy review rules.
- Compare review strategies at matched review workload or matched referral sensitivity.

## Metrics
**Classification:** balanced accuracy (consistent with ISIC 2018 Task 3), macro-F1, accuracy as a secondary measure, all 7 class precision/recall/F1, especially melanoma recall, multiclass confusion matrix, one-vs-rest ROC-AUC/PR-AUC when meaningful, calibrated reliability (ECE/Brier) with carefully described multiclass conventions, subgroup/slice results with counts.

**Review decision support:** proportion referred, false reassurance (missed melanoma among low-priority cases), sensitivity for cases with true errors/melanoma, and workload vs coverage curves.

**Not appropriate without segmentation masks/predictions:** Task 1 thresholded Jaccard, Task 2 attribute Jaccard or pixel Dice scores. These concern image masking tasks, not seven-class diagnosis.

## Reporting limitations
- HAM10000 may not reflect real-world clinical prevalence; no external validity guarantee.
- Patient IDs are unavailable, so patient-disjointness cannot be verified.
- Small validation classes (`df`=9) make per-class estimates noisy; report supports/uncertainty.
- Study is a research prototype, not a clinical medical device.

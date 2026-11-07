# Proposed methods and evaluation — seven-class project

## Goal
Investigate data-centric training and a **neuro-fuzzy correction stage that outputs
the final seven-class diagnosis**, using ImageNet-pretrained EfficientNet-B0.
This classification target was clarified on 7 November 2026. Earlier Mamdani
review-priority analysis remains a separate secondary experiment.

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

## Primary hybrid: final-class neuro-fuzzy correction
- Freeze and identify the source CNN checkpoint. Produce the final seven-class
  probabilities or hard diagnosis labels from the correction stage.
- Specify the actual architecture, inputs, membership functions, rules, inference,
  parameter-learning procedure and output conversion. Whether this is ANFIS,
  another learned fuzzy head or a different classifier remains to be decided;
  no such classifier is implemented yet.
- Learn correction parameters on training data only. For probability stacking,
  consider lesion-grouped out-of-fold training predictions or an internal
  training-only holdout; do not train the corrector on validation/test labels.
- Select hyperparameters on validation and freeze before final testing.
- Compare CNN alone with the same frozen CNN plus correction on identical images.
  A simple learned correction head using the same inputs is an additional useful
  control for separating fuzzy-structure effects from adding model capacity.
- Measure actual classification changes, including corrected and introduced
  errors. Do not assume the hybrid improves performance.
- See `docs/neuro_fuzzy_evaluation.md` for the final-prediction export contract
  and paired lesion comparison code.

## Secondary Mamdani review-priority analysis
- Design a separate **review-priority** output based on a defensible subset of: calibrated predictive uncertainty, top-class margin, prediction disagreement under test-time transformations, image-quality indicators.
- Note that entropy and top-two margin are derived from the same class-probability vector; do not claim they are statistically independent.
- Describe membership functions, rule base, min/max or chosen t-norm/s-norm, aggregation and centroid defuzzification.
- A fuzzy review score is **not equivalent to clinical risk probability** and does not necessarily improve diagnosis accuracy.

## Controlled comparison
- CNN baseline (same fixed splits).
- CNN + data-centric training alternative(s).
- Selected CNN alone versus the same CNN plus a real neuro-fuzzy final classifier.
- Report paired metric differences and repeated-training variability where feasible.
- Secondary: compare confidence/uncertainty review baselines with Mamdani review
  scores at matched workloads. These flags do not change diagnosis accuracy.

## Metrics
**Classification:** balanced accuracy (consistent with ISIC 2018 Task 3), macro-F1, accuracy as a secondary measure, all 7 class precision/recall/F1, especially melanoma recall, multiclass confusion matrix, one-vs-rest ROC-AUC/PR-AUC when meaningful, calibrated reliability (ECE/Brier) with carefully described multiclass conventions, subgroup/slice results with counts.

**Review decision support:** proportion referred, false reassurance (missed melanoma among low-priority cases), sensitivity for cases with true errors/melanoma, and workload vs coverage curves.

**Not appropriate without segmentation masks/predictions:** Task 1 thresholded Jaccard, Task 2 attribute Jaccard or pixel Dice scores. These concern image masking tasks, not seven-class diagnosis.

## Reporting limitations
- HAM10000 may not reflect real-world clinical prevalence; no external validity guarantee.
- Patient IDs are unavailable, so patient-disjointness cannot be verified.
- Small validation classes (`df`=9) make per-class estimates noisy; report supports/uncertainty.
- Study is a research prototype, not a clinical medical device.

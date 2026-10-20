# HAM10000 — verified data audit observations

*Source: Kaggle notebook results shared by team, 20 October 2026. These are dataset findings, not model performance claims.*

## Source and image reconciliation
- Metadata contains 10,015 distinct `image_id` entries and 7,470 distinct `lesion_id` values.
- Kaggle dataset directory contained 20,030 JPEG **files** representing **10,015 unique IDs** (duplicate upper-/lower-case image folders).
- Repeated ID copies had **zero differing file hashes** when compared across the corresponding folders.
- Missing image references: **0**; image IDs with no metadata: **0**; conflicting diagnoses within the same `lesion_id`: **0**.

## Repeated lesions / leakage
- 1,956 lesions have more than one photograph; 4,501 images correspond to those repeated lesions.
- Up to 6 images represent a single lesion.
- Stratified grouped train (7,116), validation (1,441), test (1,458) has **zero lesion-ID overlap**.
- **Important limit:** patient-disjointness not verifiable from source metadata.

## Missing and unknown metadata
- Missing `age`: 57.
- `sex`: 57 `unknown` entries; `localization`: 234 `unknown` entries.
- `dx_type`: histo 5,340; follow_up 3,704; consensus 902; confocal 69.

## Seven class counts
`nv=6705, mel=1113, bkl=1099, bcc=514, akiec=327, vasc=142, df=115`

## Image quality
- **Preview audit:** 600/10,015 images; 0 loading failures among these 600, all observed as 600 x 450 RGB.
- **Remaining:** enable `RUN_FULL_IMAGE_AUDIT=True` and verify all images.
- **Remaining:** enable `RUN_EXACT_HASH_AUDIT=True` to look for same-content images **across distinct image IDs / lesion IDs**. Folder-copy hash verification does not replace this check.
- Dark/bright images are **flagged for inspection**, not automatically deleted.

## Seven-class split counts

| Diagnosis | Train | Validation | Test |
|---|---:|---:|---:|
| nv | 4796 | 974 | 935 |
| mel | 782 | 146 | 185 |
| bkl | 772 | 168 | 159 |
| bcc | 358 | 89 | 67 |
| akiec | 230 | 34 | 63 |
| vasc | 89 | 21 | 32 |
| df | 89 | 9 | 17 |

## Follow-up decisions
- Keep raw source immutable; record any exclusions or changes in versioned manifests.
- Apply augmentations and resampling to **training only**.
- Audit false negatives, class-wise errors and subgroup performance on validation; do not use test examples to tune.
- Record exact Kaggle notebook version/commit used for all final models.

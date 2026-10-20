"""Data contract checks independent of Kaggle paths or training libraries.

These checks do NOT replace image-decode, visual quality, duplicate-content,
or patient-level leakage audits; those require access to the underlying images.
"""

from collections import Counter

import pandas as pd

CLASS_CODES = ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
SPLITS = ("train", "validation", "test")
REQUIRED = {"image_id", "lesion_id", "dx", "split"}


def validate_manifest(manifest: pd.DataFrame, *, require_all_classes: bool = True) -> dict:
    """Validate seven-class labels and the non-overlap of lesion IDs across splits.

    Raises ValueError with actionable details on a violated data contract.
    Does not modify the supplied DataFrame.
    """
    missing = REQUIRED - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Manifest is empty")
    if manifest[list(REQUIRED)].isna().any().any():
        raise ValueError("Required manifest columns contain null values")
    if manifest["image_id"].duplicated().any():
        examples = manifest.loc[manifest["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"Duplicate image_id records: {examples}")
    labels = set(manifest["dx"].unique())
    unsupported = labels - set(CLASS_CODES)
    if unsupported:
        raise ValueError(f"Unknown diagnosis labels: {sorted(unsupported)}")
    if require_all_classes and labels != set(CLASS_CODES):
        raise ValueError(f"Missing expected classes: {sorted(set(CLASS_CODES) - labels)}")
    splits = set(manifest["split"].unique())
    if splits != set(SPLITS):
        raise ValueError(f"Expected splits {SPLITS}, got {sorted(splits)}")
    overlap = manifest.groupby("lesion_id")["split"].nunique()
    if overlap.gt(1).any():
        raise ValueError("Data leakage: at least one lesion_id appears in multiple splits")
    conflicting_dx = manifest.groupby("lesion_id")["dx"].nunique()
    if conflicting_dx.gt(1).any():
        raise ValueError("A lesion_id has conflicting diagnosis labels")
    by_split = {name: int((manifest["split"] == name).sum()) for name in SPLITS}
    return {
        "images": int(len(manifest)),
        "unique_lesions": int(manifest["lesion_id"].nunique()),
        "class_counts": {key: int(value) for key, value in Counter(manifest["dx"]).items()},
        "split_counts": by_split,
        "lesion_overlap": 0,
    }

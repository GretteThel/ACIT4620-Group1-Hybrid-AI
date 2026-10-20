import pandas as pd
import pytest

from ham10000_hybrid.data_validation import CLASS_CODES, validate_manifest


@pytest.fixture
def manifest():
    return pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(8)],
            "lesion_id": ["lesion_0", "lesion_0"] + [f"lesion_{i}" for i in range(1, 7)],
            "dx": ["nv", "nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"],
            "split": ["train", "train", "validation", "test", "train", "train", "validation", "test"],
        }
    )


def test_valid_manifest(manifest):
    result = validate_manifest(manifest)
    assert result["images"] == 8
    assert result["unique_lesions"] == 7
    assert result["lesion_overlap"] == 0
    assert set(result["class_counts"]) == set(CLASS_CODES)


def test_detects_leakage(manifest):
    manifest.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="Data leakage"):
        validate_manifest(manifest)


def test_detects_duplicated_images(manifest):
    manifest.loc[1, "image_id"] = manifest.loc[0, "image_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(manifest)


def test_detects_conflicting_labels(manifest):
    manifest.loc[1, "dx"] = "mel"
    with pytest.raises(ValueError, match="conflicting"):
        validate_manifest(manifest)


def test_detects_missing_class(manifest):
    manifest.loc[7, "dx"] = "nv"
    with pytest.raises(ValueError, match="Missing expected classes"):
        validate_manifest(manifest)

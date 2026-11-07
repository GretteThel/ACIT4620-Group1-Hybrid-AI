import numpy as np
import pandas as pd
import pytest

from ham10000_hybrid.classifier_comparison import (
    align_hybrid_predictions,
    paired_lesion_comparison,
)
from ham10000_hybrid.data_validation import CLASS_CODES


def test_identical_predictions_have_zero_paired_difference():
    y = np.tile(np.arange(7), 2)
    predicted = y.copy()
    predicted[1] = 0
    result = paired_lesion_comparison(y, predicted, predicted, [f"l{i}" for i in range(14)],
                                      resamples=30, seed=42)
    for values in result["metrics"].values():
        assert values["difference"] == values["lower_95"] == values["upper_95"] == 0
    assert result["transitions"]["corrected_errors"] == 0
    assert result["transitions"]["introduced_errors"] == 0


def test_hand_calculated_gain_and_error_transitions():
    y = np.array([1, 1, 0, 0])
    baseline = np.array([0, 1, 0, 0])
    candidate = np.array([1, 0, 0, 1])
    result = paired_lesion_comparison(y, baseline, candidate, ["a", "b", "c", "d"], resamples=30)
    assert result["metrics"]["accuracy"]["difference"] == -.25
    assert result["metrics"]["melanoma_recall"]["difference"] == 0
    assert result["transitions"]["corrected_errors"] == 1
    assert result["transitions"]["introduced_errors"] == 2
    assert result["transitions"]["corrected_melanoma_misses"] == 1
    assert result["transitions"]["introduced_melanoma_misses"] == 1


def test_paired_bootstrap_preserves_photographs_of_each_lesion():
    y = np.zeros(6, dtype=int)
    baseline = np.ones(6, dtype=int)
    candidate = np.array([0, 0, 0, 1, 1, 1])
    result = paired_lesion_comparison(y, baseline, candidate, ["a"] * 3 + ["b"] * 3,
                                      resamples=1000, seed=42)
    assert result["metrics"]["accuracy"]["difference"] == .5
    assert result["metrics"]["accuracy"]["lower_95"] == 0
    assert result["metrics"]["accuracy"]["upper_95"] == 1


def test_conflicting_lesion_labels_and_misaligned_predictions_rejected():
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="conflicting"):
        paired_lesion_comparison(y, y, y, ["same", "same"], resamples=10)
    with pytest.raises(ValueError, match="same images"):
        paired_lesion_comparison(y, y[:1], y, ["a", "b"], resamples=10)


@pytest.fixture
def metadata():
    return {"source_cnn_run_id": "cnn", "source_checkpoint_sha256": "a" * 64,
            "split_sha256": "b" * 64, "class_codes": list(CLASS_CODES),
            "evaluation_split": "validation", "hybrid_fit_split": "train",
            "hybrid_version": "synthetic-unit-test"}


def align(frame, metadata):
    return align_hybrid_predictions(frame, metadata, ["a", "b"], source_run_id="cnn",
                                    source_checkpoint_sha256="a" * 64, split_sha256="b" * 64)


def test_hard_label_hybrid_does_not_invent_probabilities(metadata):
    frame = pd.DataFrame({"image_id": ["b", "a"], "predicted_dx": ["mel", "nv"]})
    result = align(frame, metadata)
    assert result["predicted"].tolist() == [0, 1]
    assert result["probs"] is None


def test_hybrid_probability_class_order_and_argmax_are_checked(metadata):
    p = np.eye(7)[[1, 0]]
    frame = pd.DataFrame(p, columns=[f"prob_{code}" for code in CLASS_CODES])
    frame["image_id"] = ["b", "a"]
    frame["predicted_dx"] = ["mel", "nv"]
    assert align(frame, metadata)["predicted"].tolist() == [0, 1]
    frame.loc[0, "predicted_dx"] = "nv"
    with pytest.raises(ValueError, match="disagree"):
        align(frame, metadata)
    with pytest.raises(ValueError, match="all seven"):
        align(frame.drop(columns="prob_df"), metadata)


@pytest.mark.parametrize("field,value", [("source_checkpoint_sha256", "wrong"),
                                        ("split_sha256", "wrong"),
                                        ("hybrid_fit_split", "validation"),
                                        ("hybrid_version", "")])
def test_hybrid_provenance_and_training_partition_rejected(metadata, field, value):
    frame = pd.DataFrame({"image_id": ["a", "b"], "predicted_dx": ["nv", "mel"]})
    with pytest.raises(ValueError):
        align(frame, {**metadata, field: value})


def test_hybrid_requires_same_image_ids(metadata):
    frame = pd.DataFrame({"image_id": ["a", "different"], "predicted_dx": ["nv", "mel"]})
    with pytest.raises(ValueError, match="image IDs differ"):
        align(frame, metadata)

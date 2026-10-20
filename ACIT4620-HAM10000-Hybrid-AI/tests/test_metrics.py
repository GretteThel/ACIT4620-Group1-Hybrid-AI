import pytest

from ham10000_hybrid.data_validation import CLASS_CODES
from ham10000_hybrid.metrics import compute_classification_metrics


def test_perfect_predictions():
    report = compute_classification_metrics(CLASS_CODES, CLASS_CODES)
    assert report["balanced_accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["melanoma_recall"] == 1.0
    assert len(report["confusion_matrix"]) == 7


def test_missed_melanoma_is_detected():
    predictions = list(CLASS_CODES)
    predictions[1] = "nv"
    report = compute_classification_metrics(CLASS_CODES, predictions)
    assert report["melanoma_recall"] == 0.0
    assert report["balanced_accuracy"] < 1.0


def test_invalid_input_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        compute_classification_metrics(["nv"], [])

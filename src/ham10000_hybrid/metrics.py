"""Seven-class classification evaluation: report all classes, not accuracy alone."""

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from .data_validation import CLASS_CODES


def compute_classification_metrics(
    true_labels: Sequence[str], predicted_labels: Sequence[str]
) -> dict:
    """Compute dataset-wide and per-class measures for the seven known diagnoses.

    Inputs are diagnosis codes. Probability/calibration summaries and review
    evaluation are implemented separately in evaluation.py.
    """
    y_true, y_pred = list(true_labels), list(predicted_labels)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Nonempty ground truth and prediction arrays must have equal lengths")
    unknown = (set(y_true) | set(y_pred)) - set(CLASS_CODES)
    if unknown:
        raise ValueError(f"Unexpected diagnosis codes: {sorted(unknown)}")
    per_class = classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_CODES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_CODES), average="macro", zero_division=0)),
        "melanoma_recall": float(recall_score(y_true, y_pred, labels=["mel"], average="macro", zero_division=0)),
        "per_class": {code: per_class[code] for code in CLASS_CODES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CLASS_CODES)).tolist(),
        "confusion_labels": list(CLASS_CODES),
    }

"""Compare final seven-class decisions, including a real neuro-fuzzy corrector."""

import numpy as np
import pandas as pd

from .data_validation import CLASS_CODES
from .evaluation import MEL_INDEX, validate_predictions

METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall")


def _indices(values, name):
    array = np.asarray(values)
    if (array.ndim != 1 or not len(array) or not np.issubdtype(array.dtype, np.integer)
            or (array < 0).any() or (array >= len(CLASS_CODES)).any()):
        raise ValueError(f"{name} must be a nonempty vector of seven-class integer indices")
    return array


def align_hybrid_predictions(frame, metadata, image_ids, *, source_run_id,
                             source_checkpoint_sha256, split_sha256):
    """Load final predictions by ID, bound to their frozen source CNN.

    Accept final probabilities or hard labels; never invent probabilities for
    hard-label classifiers. Ground truth is obtained separately from the manifest.
    Fit-split metadata is the producer's declaration, not an independent fit audit.
    """
    expected = {
        "source_cnn_run_id": source_run_id,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "split_sha256": split_sha256,
        "class_codes": list(CLASS_CODES), "evaluation_split": "validation",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Hybrid metadata has a different or missing {key}")
    if metadata.get("hybrid_fit_split") not in {"train", "train_oof"}:
        raise ValueError("Hybrid parameters must be fitted on train/train_oof, not validation/test")
    if not isinstance(metadata.get("hybrid_version"), str) or not metadata["hybrid_version"].strip():
        raise ValueError("Hybrid metadata requires a nonempty hybrid_version")
    if "image_id" not in frame or frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise ValueError("Hybrid predictions require unique, nonmissing image IDs")
    ids = list(image_ids)
    if len(set(ids)) != len(ids) or set(frame["image_id"]) != set(ids):
        raise ValueError("Hybrid image IDs differ from the source CNN validation images")
    aligned = frame.set_index("image_id").loc[ids]
    columns = [f"prob_{code}" for code in CLASS_CODES]
    present = [column for column in columns if column in aligned]
    probabilities = None
    labels = None
    if present:
        if len(present) != len(columns):
            raise ValueError("Supply all seven hybrid probability columns, or hard labels only")
        probabilities = aligned[columns].to_numpy(dtype=float)
        validate_predictions(np.zeros(len(ids), dtype=int), probabilities, ids)
        labels = probabilities.argmax(axis=1)
    if "predicted_dx" in aligned:
        mapped = aligned["predicted_dx"].map({code: i for i, code in enumerate(CLASS_CODES)})
        if mapped.isna().any():
            raise ValueError("Unknown or missing hybrid predicted_dx")
        supplied = mapped.to_numpy(dtype=int)
        if labels is not None and not np.array_equal(labels, supplied):
            raise ValueError("Hybrid hard labels disagree with its final probability argmax")
        labels = supplied
    if labels is None:
        raise ValueError("Supply predicted_dx or all seven final hybrid probability columns")
    return {"predicted": labels, "probs": probabilities}


def _scores(y, predicted, weights=None):
    k = len(CLASS_CODES)
    cm = np.bincount(y * k + predicted, weights=weights, minlength=k * k).reshape(k, k)
    diagonal, support = np.diag(cm), cm.sum(axis=1)
    recall = np.divide(diagonal, support, out=np.zeros(k), where=support > 0)
    denominator = support + cm.sum(axis=0)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(k), where=denominator > 0)
    return np.array([diagonal.sum() / cm.sum(), recall[support > 0].mean(), f1.mean(),
                     recall[MEL_INDEX] if support[MEL_INDEX] else np.nan])


def paired_lesion_comparison(y_true, baseline_predicted, candidate_predicted, lesion_ids,
                             *, resamples=1000, seed=42):
    """Class-stratified paired lesion bootstrap for candidate-minus-baseline.

    Both classifiers receive identical sampled lesions and multiplicities. These
    conditional validation intervals do not establish held-out generalization or
    include training-seed variation. Positive improvement is never assumed.
    """
    y = _indices(y_true, "Ground truth")
    base = _indices(baseline_predicted, "Baseline predictions")
    candidate = _indices(candidate_predicted, "Candidate predictions")
    groups = np.asarray(lesion_ids)
    if len(base) != len(y) or len(candidate) != len(y) or groups.shape != y.shape:
        raise ValueError("Both classifiers and lesion IDs must align to the same images")
    if any(not isinstance(value, str) or not value.strip() for value in groups.tolist()):
        raise ValueError("Lesion IDs must be nonmissing strings")
    table = pd.DataFrame({"lesion": groups, "label": y})
    if table.groupby("lesion")["label"].nunique().gt(1).any():
        raise ValueError("A lesion has conflicting ground truth")
    if not isinstance(resamples, int) or resamples < 2:
        raise ValueError("Paired bootstrap requires at least two resamples")
    unique, inverse = np.unique(groups, return_inverse=True)
    labels = np.array([y[np.flatnonzero(inverse == i)[0]] for i in range(len(unique))])
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        multiplicity = np.zeros(len(unique), dtype=int)
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            np.add.at(multiplicity, rng.choice(indices, len(indices), replace=True), 1)
        weights = multiplicity[inverse]
        differences.append(_scores(y, candidate, weights) - _scores(y, base, weights))
    values = np.asarray(differences)
    base_scores, candidate_scores = _scores(y, base), _scores(y, candidate)
    result = {}
    for i, metric in enumerate(METRIC_NAMES):
        finite = values[:, i][np.isfinite(values[:, i])]
        result[metric] = {
            "baseline": float(base_scores[i]) if np.isfinite(base_scores[i]) else None,
            "candidate": float(candidate_scores[i]) if np.isfinite(candidate_scores[i]) else None,
            "difference": float(candidate_scores[i] - base_scores[i])
            if np.isfinite(candidate_scores[i] - base_scores[i]) else None,
            "lower_95": float(np.quantile(finite, .025)) if len(finite) else None,
            "upper_95": float(np.quantile(finite, .975)) if len(finite) else None,
        }
    baseline_wrong, candidate_wrong = base != y, candidate != y
    melanoma = y == MEL_INDEX
    return {
        "evaluation_split": "validation", "method": "class-stratified paired lesion bootstrap",
        "resamples": resamples, "seed": seed, "images": len(y), "lesions": len(unique),
        "metrics": result,
        "transitions": {
            "changed_predictions": int((base != candidate).sum()),
            "corrected_errors": int((baseline_wrong & ~candidate_wrong).sum()),
            "introduced_errors": int((~baseline_wrong & candidate_wrong).sum()),
            "corrected_melanoma_misses": int((melanoma & baseline_wrong & ~candidate_wrong).sum()),
            "introduced_melanoma_misses": int((melanoma & ~baseline_wrong & candidate_wrong).sum()),
        },
        "limitations": "Conditional on this validation set and fitted models; not patient bootstrap or independent final-test evidence",
    }

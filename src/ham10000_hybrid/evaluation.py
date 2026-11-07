"""Validation probability metrics and label-blind review-priority evaluation.

Referral is a flag for review, never an assumed corrected diagnosis. Scores are
ranked without ground-truth labels; equal scores are ordered by image ID.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .data_validation import CLASS_CODES

MEL_INDEX = CLASS_CODES.index("mel")
CLASS_CODES_INDICES = tuple(range(len(CLASS_CODES)))


def validate_predictions(y_true, probs, image_ids=None, tta_probs=None):
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Ground truth must be a nonempty integer vector")
    if (y < 0).any() or (y >= len(CLASS_CODES)).any():
        raise ValueError("Ground-truth indices must follow the seven-class project order")
    if p.shape != (len(y), len(CLASS_CODES)):
        raise ValueError("Expected one seven-class probability row per image")
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid class probabilities")
    if image_ids is not None:
        ids = np.asarray(image_ids)
        if ids.ndim != 1 or len(ids) != len(y):
            raise ValueError("Image IDs must match prediction rows")
        if any(not isinstance(value, str) or not value.strip() for value in ids.tolist()):
            raise ValueError("Image IDs must be nonempty strings")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("Image IDs must be unique")
    if tta_probs is not None:
        tta = np.asarray(tta_probs, dtype=float)
        if tta.ndim != 3 or tta.shape[0] != len(y) or tta.shape[2] != len(CLASS_CODES):
            raise ValueError("Expected TTA probabilities with shape (images, views, seven classes)")
        if tta.shape[1] < 2:
            raise ValueError("TTA disagreement requires at least two views")
        if (not np.isfinite(tta).all() or (tta < 0).any() or (tta > 1).any()
                or not np.allclose(tta.sum(axis=2), 1, atol=1e-5)):
            raise ValueError("Invalid TTA probabilities")
    return y, p


def _entropy(p):
    return -(p * np.log(np.clip(p, 1e-15, 1))).sum(axis=-1)


def probability_metrics(y_true, probs, *, n_bins=15):
    """OVR ROC-AUC/AP, multiclass NLL/Brier, and fixed-bin top-label ECE.

    Brier is mean sum of squared errors across seven classes (range 0..2).
    ECE compares top-label confidence with argmax correctness, not melanoma
    calibration. Undefined class ROC-AUC/AP values are None, never fabricated.
    """
    y, p = validate_predictions(y_true, probs)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    one_hot = np.eye(len(CLASS_CODES))[y]
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean_confidence - accuracy)
        bins.append({
            "bin": index, "lower": index / n_bins, "upper": (index + 1) / n_bins,
            "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy,
        })
    per_class = {}
    for index, code in enumerate(CLASS_CODES):
        positive = y == index
        support = int(positive.sum())
        per_class[code] = {
            "support": support, "prevalence": support / len(y),
            "roc_auc": float(roc_auc_score(positive, p[:, index]))
            if 0 < support < len(y) else None,
            "average_precision": float(average_precision_score(positive, p[:, index]))
            if support else None,
        }
    return {
        "nll": float(log_loss(y, p / p.sum(axis=1, keepdims=True), labels=list(CLASS_CODES_INDICES))),
        "multiclass_brier_sum": float(np.square(p - one_hot).sum(axis=1).mean()),
        "top_label_ece": float(ece), "ece_bins": bins, "per_class": per_class,
    }


def review_priority_scores(probs, tta_probs=None):
    """Produce higher-is-more-review-priority scores without using labels."""
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("Expected a probability matrix")
    validate_predictions(np.zeros(len(p), dtype=int), p, tta_probs=tta_probs)
    ordered = np.sort(p, axis=1)
    scores = {
        "low_confidence": 1 - p.max(axis=1),
        "entropy": _entropy(p) / np.log(len(CLASS_CODES)),
        "small_margin": 1 - (ordered[:, -1] - ordered[:, -2]),
        "mel_probability": p[:, MEL_INDEX].copy(),
    }
    if tta_probs is not None:
        tta = np.asarray(tta_probs, dtype=float)
        js = _entropy(tta.mean(axis=1)) - _entropy(tta).mean(axis=1)
        scores["tta_disagreement"] = np.clip(js / np.log(len(CLASS_CODES)), 0, 1)
    return scores


def align_review_scores(frame, image_ids):
    """Align external fuzzy scores by ID; reject omissions, duplicates and extras."""
    if not {"image_id", "review_score"}.issubset(frame.columns):
        raise ValueError("Fuzzy scores require image_id and review_score columns")
    if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise ValueError("Fuzzy score IDs must be nonmissing and unique")
    ids = list(image_ids)
    if len(set(ids)) != len(ids) or set(frame["image_id"]) != set(ids):
        raise ValueError("Fuzzy score IDs differ from the validation predictions")
    values = pd.to_numeric(frame.set_index("image_id").loc[ids, "review_score"], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Fuzzy review scores must be finite")
    return values.to_numpy(dtype=float)


def priority_order(scores, image_ids):
    values = np.asarray(scores, dtype=float)
    ids = np.asarray(image_ids)
    if values.shape != ids.shape or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Priority scores must be a finite vector matching image IDs")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Image IDs must be unique for deterministic tie-breaking")
    return np.lexsort((ids, -values))


def review_workload_curves(
    y_true, probs, image_ids, scores: Mapping[str, Sequence[float]], budgets,
):
    """Use floor(N*budget) reviewed images for every strategy at each budget.

    Error capture counts existing CNN errors flagged for review. Melanoma capture
    is reported both for all true melanoma and for melanoma missed by the CNN.
    No flagged case is assumed correctly diagnosed by a reviewer.
    """
    y, p = validate_predictions(y_true, probs, image_ids)
    budgets = np.asarray(list(budgets), dtype=float)
    if (budgets.ndim != 1 or not len(budgets) or not np.isfinite(budgets).all()
            or (budgets < 0).any() or (budgets > 1).any()):
        raise ValueError("Review budgets must be finite fractions in [0, 1]")
    if not scores:
        raise ValueError("At least one priority strategy is required")
    errors = p.argmax(axis=1) != y
    melanoma = y == MEL_INDEX
    missed = melanoma & errors
    n_errors, n_mel, n_missed = int(errors.sum()), int(melanoma.sum()), int(missed.sum())

    def fraction(a, b):
        return float(a / b) if b else None

    rows = []
    for name, raw in scores.items():
        values = np.asarray(raw, dtype=float)
        order = priority_order(values, image_ids)
        error_prefix = np.r_[0, np.cumsum(errors[order])]
        mel_prefix = np.r_[0, np.cumsum(melanoma[order])]
        missed_prefix = np.r_[0, np.cumsum(missed[order])]
        for budget in np.unique(budgets):
            count = int(np.floor(len(y) * budget + 1e-12))
            flagged_errors = int(error_prefix[count])
            flagged_mel = int(mel_prefix[count])
            flagged_missed = int(missed_prefix[count])
            remaining_errors, remaining_missed = n_errors - flagged_errors, n_missed - flagged_missed
            threshold = float(values[order[count - 1]]) if count else None
            rows.append({
                "strategy": name, "requested_review_fraction": float(budget),
                "n_images": len(y), "n_reviewed": count, "review_fraction": count / len(y),
                "score_at_cutoff": threshold,
                "cutoff_tie_size": int((values == threshold).sum()) if count else 0,
                "n_errors": n_errors, "n_errors_reviewed": flagged_errors,
                "error_capture_rate": fraction(flagged_errors, n_errors),
                "review_error_precision": fraction(flagged_errors, count),
                "n_unreviewed": len(y) - count, "n_errors_unreviewed": remaining_errors,
                "unreviewed_error_rate": fraction(remaining_errors, len(y) - count),
                "n_melanoma": n_mel, "n_melanoma_reviewed": flagged_mel,
                "melanoma_flag_rate": fraction(flagged_mel, n_mel),
                "n_melanoma_missed_by_cnn": n_missed,
                "n_missed_melanoma_reviewed": flagged_missed,
                "missed_melanoma_capture_rate": fraction(flagged_missed, n_missed),
                "n_missed_melanoma_unreviewed": remaining_missed,
                "missed_melanoma_fraction_of_unreviewed": fraction(remaining_missed, len(y) - count),
            })
    return pd.DataFrame(rows)


def lesion_bootstrap_intervals(y_true, probs, lesion_ids, *, resamples=200, seed=42):
    """Stratified lesion bootstrap; resample lesions within each observed class.

    All photographs of a sampled lesion receive the same multiplicity. This is
    conditional on observed class strata and is not a patient-level bootstrap.
    Intervals do not undo validation model-selection or calibration reuse.
    """
    y, p = validate_predictions(y_true, probs)
    groups = np.asarray(lesion_ids)
    if groups.ndim != 1 or len(groups) != len(y) or pd.isna(groups).any():
        raise ValueError("Nonmissing lesion IDs must match prediction rows")
    if any(not isinstance(value, str) or not value.strip() for value in groups.tolist()):
        raise ValueError("Lesion IDs must be nonempty strings")
    if not isinstance(resamples, int) or resamples < 2:
        raise ValueError("Bootstrap requires at least two resamples")
    group_table = pd.DataFrame({"lesion": groups, "label": y})
    if group_table.groupby("lesion")["label"].nunique().gt(1).any():
        raise ValueError("A lesion has conflicting labels")
    unique_groups, inverse = np.unique(groups.astype(str), return_inverse=True)
    group_labels = np.array([y[np.flatnonzero(inverse == i)[0]] for i in range(len(unique_groups))])
    rng = np.random.default_rng(seed)
    pred = p.argmax(axis=1)
    results = []
    for _ in range(resamples):
        multiplicity = np.zeros(len(unique_groups), dtype=int)
        for label in np.unique(group_labels):
            indices = np.flatnonzero(group_labels == label)
            np.add.at(multiplicity, rng.choice(indices, size=len(indices), replace=True), 1)
        cm = np.bincount(y * 7 + pred, weights=multiplicity[inverse], minlength=49).reshape(7, 7)
        diagonal, support = np.diag(cm), cm.sum(axis=1)
        recalls = np.divide(diagonal, support, out=np.zeros(7), where=support > 0)
        denom = support + cm.sum(axis=0)
        f1 = np.divide(2 * diagonal, denom, out=np.zeros(7), where=denom > 0)
        results.append([diagonal.sum() / cm.sum(), recalls[support > 0].mean(),
                        f1.mean(), recalls[MEL_INDEX] if support[MEL_INDEX] else np.nan])
    values = np.asarray(results)
    intervals = {}
    for index, key in enumerate(["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall"]):
        finite = values[:, index][np.isfinite(values[:, index])]
        intervals[key] = {"lower_95": float(np.quantile(finite, .025)) if len(finite) else None,
                          "upper_95": float(np.quantile(finite, .975)) if len(finite) else None}
    return {"method": "class-stratified lesion bootstrap percentile intervals",
            "resamples": resamples, "seed": seed, "intervals": intervals}

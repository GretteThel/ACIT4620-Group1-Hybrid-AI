"""Mamdani uncertainty-review prototype for seven-class HAM10000.

Not a diagnostic device. The review score is NOT a melanoma probability
or a clinically validated triage priority. Fuzzy membership breakpoints
are illustrative; choose/freeze them using validation data only.
"""
from __future__ import annotations
import numpy as np

CLASS_CODES = ('nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df')
GRID = np.linspace(0.0, 100.0, 501)


def trapezoid(x, a, b, c, d):
    """Trapezoidal membership; supports shoulder functions when a=b or c=d."""
    x = np.asarray(x, dtype=float)
    if not (a <= b <= c <= d):
        raise ValueError('membership vertices must be sorted')
    left = np.ones_like(x) if a == b else np.clip((x-a)/(b-a), 0., 1.)
    right = np.ones_like(x) if c == d else np.clip((d-x)/(d-c), 0., 1.)
    return np.maximum(0.0, np.minimum(left, right))


def triangle(x, a, b, c):
    return trapezoid(x, a, b, b, c)


def scores_from_probs(probs: np.ndarray, tta_probs: np.ndarray):
    """Return normalized entropy and TTA mean total-variation disagreement.

    probs: [n, 7] calibrated probabilities for the original image.
    tta_probs: [n, t, 7] matching calibrated probabilities from t views.
    """
    probs = np.asarray(probs, dtype=float)
    tta_probs = np.asarray(tta_probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 7:
        raise ValueError('probs shape must be [n,7]')
    if tta_probs.ndim != 3 or tta_probs.shape[0] != probs.shape[0] or tta_probs.shape[2] != 7:
        raise ValueError('tta_probs shape must be [n,t,7]')
    if tta_probs.shape[1] < 2:
        raise ValueError('use at least two TTA views')
    for arr, label in [(probs,'probs'),(tta_probs,'tta_probs')]:
        if not np.isfinite(arr).all() or (arr < -1e-8).any() or (arr > 1.00000001).any():
            raise ValueError(label + ': expected finite probabilities in [0,1]')
        if not np.allclose(arr.sum(axis=-1), 1.0, atol=1e-4):
            raise ValueError(label + ': class probabilities must sum to one')
    # entropy ranges from 0 to 1; high entropy implies diffuse class scores
    safe_p = np.clip(probs, 1e-12, 1.)
    entropy = -np.sum(probs * np.log(safe_p), axis=1) / np.log(probs.shape[1])
    # The mean total variation of each TTA view from the mean TTA distribution
    avg = tta_probs.mean(axis=1, keepdims=True)
    disagreement = np.abs(tta_probs - avg).sum(axis=2).mean(axis=1) / 2.
    return np.clip(entropy, 0, 1), np.clip(disagreement, 0, 1)


def mamdani_score(entropy: float, disagreement: float) -> float:
    """Six compact Mamdani min/max rules + centroid defuzzification.

    Output domain: 0-100. High = 'needs uncertainty review' (research only).
    """
    h = float(entropy)
    d = float(disagreement)
    if not (0 <= h <= 1 and 0 <= d <= 1):
        raise ValueError('inputs must be in [0,1]')
    h_low = float(trapezoid(h, 0, 0, .25, .45))
    h_mid = float(triangle(h, .30, .52, .75))
    h_high = float(trapezoid(h, .65, .82, 1, 1))
    d_low = float(trapezoid(d, 0, 0, .06, .12))
    d_mid = float(triangle(d, .07, .18, .30))
    d_high = float(trapezoid(d, .22, .35, 1, 1))

    low_out = trapezoid(GRID, 0, 0, 20, 40)
    mid_out = triangle(GRID, 30, 50, 70)
    high_out = trapezoid(GRID, 60, 80, 100, 100)

    # Rule activations; high uncertainty or instability raises priority
    low_activation = min(h_low, d_low)
    mid_activation = max(min(h_low, d_mid), min(h_mid, d_low))
    high_activation = max(h_high, d_high, min(h_mid, d_mid))
    combined = np.maximum.reduce([
        np.minimum(low_activation, low_out),
        np.minimum(mid_activation, mid_out),
        np.minimum(high_activation, high_out),
    ])
    if combined.sum() < 1e-10:
        # Conservative fallback where unusual membership gaps would leave no rule
        return 100.
    return float(np.sum(GRID * combined) / np.sum(combined))


def batch_fuzzy_scores(probs: np.ndarray, tta_probs: np.ndarray):
    entropy, disagreement = scores_from_probs(probs, tta_probs)
    return np.array([mamdani_score(float(h), float(d)) for h,d in zip(entropy, disagreement)])


def review_metrics(y_true, probs, review, mel_idx=1):
    """Evaluate uncertainty review, not a replacement diagnosis classifier."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    review = np.asarray(review, dtype=bool)
    y_pred = probs.argmax(axis=1)
    if not (len(y_true) == len(review) == len(y_pred)):
        raise ValueError('length mismatch')
    error = y_pred != y_true
    accepted = ~review
    missed_mel = (y_true == mel_idx) & (y_pred != mel_idx)
    return {
        'review_fraction': float(review.mean()),
        'errors_captured_fraction': float((review & error).sum()/error.sum()) if error.sum() else None,
        'error_fraction_among_unreviewed': float((accepted & error).sum()/accepted.sum()) if accepted.sum() else None,
        'missed_melanomas_not_flagged_count': int((accepted & missed_mel).sum()),
        'missed_melanomas_not_flagged_fraction_of_all_melanomas': float((accepted & missed_mel).sum()/(y_true==mel_idx).sum()) if (y_true==mel_idx).sum() else None,
        'class_predictions_unchanged_by_review': True,
    }


def cutoff_for_review_budget(validation_scores, review_fraction):
    """Set cutoff on validation only; ties can lead to a different actual fraction."""
    if not 0 < review_fraction < 1:
        raise ValueError('budget must be strictly between 0 and 1')
    vals = np.asarray(validation_scores, dtype=float)
    if not np.isfinite(vals).all() or len(vals) == 0:
        raise ValueError('invalid validation scores')
    return float(np.quantile(vals, 1-review_fraction))


def topk_review_mask(scores, review_fraction, tie_breaker=None):
    """Exact-budget batch ranking (uses scores only, NEVER labels).

    If fuzzy rules yield equal scores, favor larger prespecified tie_breaker
    (e.g. entropy); remaining ties use stable row order. This is a batch
    prioritization comparison, NOT a production fixed-score threshold.
    """
    scores = np.asarray(scores, dtype=float)
    if not (0 < review_fraction < 1) or len(scores) == 0:
        raise ValueError('invalid budget or scores')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    if tie_breaker is None:
        tie_breaker = np.zeros(len(scores), dtype=float)
    tie_breaker = np.asarray(tie_breaker, dtype=float)
    if tie_breaker.shape != scores.shape or not np.isfinite(tie_breaker).all():
        raise ValueError('invalid tie_breaker')
    n_select = min(len(scores), max(1, int(np.ceil(review_fraction*len(scores)))))
    order = np.lexsort((np.arange(len(scores)), -tie_breaker, -scores))
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:n_select]] = True
    return mask

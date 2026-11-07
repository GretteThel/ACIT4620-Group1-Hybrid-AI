"""Check probability metrics and referral accounting against small hand-worked examples."""

import numpy as np
import pandas as pd
import pytest

from ham10000_hybrid.evaluation import (
    align_review_scores,
    lesion_bootstrap_intervals,
    probability_metrics,
    review_priority_scores,
    review_workload_curves,
    validate_predictions,
)


def example():
    # a: missed melanoma, b: correct melanoma, c: wrong non-melanoma, d: correct nv.
    y = np.array([1, 1, 0, 0])
    probs = np.eye(7)[[0, 1, 1, 0]] * .93 + .01
    return y, probs, np.array(["a", "b", "c", "d"])


def test_review_counts_at_equal_workload_are_hand_calculated():
    y, p, ids = example()
    curves = review_workload_curves(y, p, ids, {
        "missed_first": [4, 1, 3, 2], "missed_last": [1, 4, 2, 3],
    }, [0, .5, 1])
    half = curves[curves["requested_review_fraction"] == .5].set_index("strategy")
    assert half["n_reviewed"].tolist() == [2, 2]
    assert half.loc["missed_first", "error_capture_rate"] == 1
    assert half.loc["missed_last", "error_capture_rate"] == 0
    assert half.loc["missed_first", "n_missed_melanoma_unreviewed"] == 0
    assert half.loc["missed_last", "n_missed_melanoma_unreviewed"] == 1
    assert half.loc["missed_first", "melanoma_flag_rate"] == .5
    assert half.loc["missed_first", "missed_melanoma_capture_rate"] == 1
    assert half.loc["missed_last", "missed_melanoma_capture_rate"] == 0
    assert curves.loc[curves["requested_review_fraction"] == 0, "n_reviewed"].eq(0).all()
    assert curves.loc[curves["requested_review_fraction"] == 1, "n_errors_unreviewed"].eq(0).all()


def test_ties_use_id_not_labels_or_input_row_order():
    y, p, ids = example()
    a = review_workload_curves(y, p, ids, {"tied": [1, 1, 1, 1]}, [.25]).iloc[0]
    order = np.array([3, 2, 1, 0])
    b = review_workload_curves(y[order], p[order], ids[order], {"tied": [1, 1, 1, 1]}, [.25]).iloc[0]
    assert a["n_missed_melanoma_reviewed"] == b["n_missed_melanoma_reviewed"] == 1
    assert a["cutoff_tie_size"] == 4
    assert a["n_reviewed"] == 1


def test_perfect_classifier_has_undefined_error_capture_not_fake_success():
    y = np.arange(7)
    p = np.eye(7)
    curve = review_workload_curves(y, p, [str(i) for i in y], {"s": np.arange(7)}, [0]).iloc[0]
    assert curve["error_capture_rate"] is None
    assert curve["missed_melanoma_capture_rate"] is None
    assert curve["n_missed_melanoma_unreviewed"] == 0


def test_fuzzy_scores_align_by_id_and_reject_bad_inputs():
    scores = pd.DataFrame({"image_id": ["b", "a"], "review_score": [.2, .9]})
    assert align_review_scores(scores, ["a", "b"]).tolist() == [.9, .2]
    for invalid in [scores.iloc[:1], pd.concat([scores, scores.iloc[:1]]),
                    scores.assign(review_score=[np.nan, .9])]:
        with pytest.raises(ValueError):
            align_review_scores(invalid, ["a", "b"])


def test_probability_metrics_and_ece_on_perfect_predictions():
    y = np.arange(7)
    metrics = probability_metrics(y, np.eye(7), n_bins=10)
    assert metrics["nll"] < 1e-12
    assert metrics["multiclass_brier_sum"] == 0
    assert metrics["top_label_ece"] == 0
    assert metrics["ece_bins"][-1]["count"] == 7
    assert metrics["per_class"]["mel"]["roc_auc"] == 1
    assert metrics["per_class"]["mel"]["average_precision"] == 1


def test_probability_metrics_uniform_predictions_have_known_values():
    y = np.arange(7)
    metrics = probability_metrics(y, np.full((7, 7), 1 / 7))
    assert metrics["nll"] == pytest.approx(np.log(7))
    assert metrics["multiclass_brier_sum"] == pytest.approx(6 / 7)
    assert metrics["top_label_ece"] == pytest.approx(0)
    absent = probability_metrics(np.array([0, 0]), np.eye(7)[[0, 0]])
    assert absent["per_class"]["mel"]["roc_auc"] is None
    assert absent["per_class"]["mel"]["average_precision"] is None


def test_disagreement_uses_all_views_and_does_not_use_truth():
    p = np.eye(7)[[0, 1]]
    identical = np.stack([p, p], axis=1)
    divergent = np.stack([p, np.eye(7)[[1, 0]]], axis=1)
    assert np.allclose(review_priority_scores(p, identical)["tta_disagreement"], 0)
    assert (review_priority_scores(p, divergent)["tta_disagreement"] > 0).all()
    scores = review_priority_scores(np.full((2, 7), 1 / 7))
    assert np.allclose(scores["entropy"], 1)
    assert np.allclose(scores["small_margin"], 1)


@pytest.mark.parametrize("invalid", [np.full((2, 7), np.nan), np.zeros((2, 7)), np.ones((2, 6))])
def test_invalid_probabilities_rejected(invalid):
    with pytest.raises(ValueError):
        validate_predictions(np.array([0, 1]), invalid)


def test_invalid_tta_and_review_budgets_rejected():
    y, p, ids = example()
    with pytest.raises(ValueError, match="TTA"):
        review_priority_scores(p, np.zeros((4, 2, 7)))
    with pytest.raises(ValueError, match="budgets"):
        review_workload_curves(y, p, ids, {"s": np.arange(4)}, [1.1])
    with pytest.raises(ValueError, match="unique"):
        validate_predictions(y, p, ["a", "a", "b", "c"])


def test_lesion_bootstrap_is_reproducible_and_rejects_label_conflicts():
    y = np.repeat(np.arange(7), 2)
    probs = np.eye(7)[y]
    lesions = np.array([f"lesion_{label}" for label in y])
    a = lesion_bootstrap_intervals(y, probs, lesions, resamples=20, seed=7)
    b = lesion_bootstrap_intervals(y, probs, lesions, resamples=20, seed=7)
    assert a == b
    assert all(value == {"lower_95": 1, "upper_95": 1} for value in a["intervals"].values())
    lesions[2] = lesions[0]
    with pytest.raises(ValueError, match="conflicting labels"):
        lesion_bootstrap_intervals(y, probs, lesions, resamples=20)


def test_lesion_bootstrap_keeps_correlated_photographs_together():
    # Two lesions, three photos each: one always right and one always wrong.
    # Resampling whole lesions permits accuracy 0, .5 or 1; image resampling
    # would instead introduce many intermediate fractions.
    y = np.zeros(6, dtype=int)
    probs = np.eye(7)[[0, 0, 0, 1, 1, 1]]
    lesions = np.array(["good"] * 3 + ["bad"] * 3)
    result = lesion_bootstrap_intervals(y, probs, lesions, resamples=1000, seed=42)
    assert result["intervals"]["accuracy"] == {"lower_95": 0, "upper_95": 1}

"""Lightweight tests for inference mathematics (does not use medical data)."""
import unittest
import numpy as np
from fuzzy_review import (trapezoid, scores_from_probs, batch_fuzzy_scores,
                          topk_review_mask, review_metrics)

class TestFuzzyReview(unittest.TestCase):
    def test_shoulder_memberships(self):
        self.assertAlmostEqual(float(trapezoid(0, 0, 0, .2, .4)), 1.)
        self.assertAlmostEqual(float(trapezoid(1, .6, .8, 1, 1)), 1.)
    def test_agreement_is_zero_for_identical_views(self):
        p=np.array([[.1,.2,.1,.1,.1,.2,.2]])
        tta=np.repeat(p[:,None,:],4,axis=1)
        _,d=scores_from_probs(p,tta)
        self.assertAlmostEqual(float(d[0]),0.)
    def test_ambiguous_higher_than_confident(self):
        confident=np.array([.94,.01,.01,.01,.01,.01,.01])
        diffuse=np.array([.2,.18,.17,.13,.12,.11,.09])
        p=np.stack([confident,diffuse]);tta=np.repeat(p[:,None,:],4,axis=1)
        score=batch_fuzzy_scores(p,tta)
        self.assertGreater(score[1],score[0]);self.assertTrue(np.isfinite(score).all())
    def test_exact_budget_even_when_tied(self):
        s=np.ones(100)
        mask=topk_review_mask(s,.2,tie_breaker=np.arange(100))
        self.assertEqual(int(mask.sum()),20)
    def test_review_does_not_change_class(self):
        p=np.array([[.92,.02,.02,.01,.01,.01,.01],[.02,.86,.03,.03,.02,.02,.02]])
        report=review_metrics(np.array([0,0]),p,np.array([False,True]))
        self.assertTrue(report['class_predictions_unchanged_by_review'])
        self.assertEqual(report['missed_melanomas_not_flagged_count'],0)

if __name__=='__main__':unittest.main()

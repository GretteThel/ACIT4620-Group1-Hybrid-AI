"""No test-partition inputs; visual-data semantics only."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chart_utils import contribution_plot_rows


class FuzzySignedChartTests(unittest.TestCase):
    def make_rows(self, vals):
        return [
            {"Rule": f"Rule {i+1}", "Predicted-logit contribution": val}
            for i, val in enumerate(vals)
        ]

    def test_positive_and_negative_values_preserved(self):
        vals = [0.] * 16
        vals[6] = 5.996848
        vals[8] = -0.0125
        result = contribution_plot_rows(self.make_rows(vals))
        self.assertEqual(len(result), 16)
        self.assertEqual(result[6]['Contribution'], 5.996848)
        self.assertEqual(result[8]['Contribution'], -0.0125)
        self.assertEqual(result[8]['Direction'], 'Negative')
        self.assertEqual(result[6]['Direction'], 'Positive')
        self.assertEqual(sum(x['Contribution'] for x in result), sum(vals))

    def test_near_zero_values_do_not_round_to_zero(self):
        vals = [0.] * 16
        vals[0] = 1.267828452e-6
        result = contribution_plot_rows(self.make_rows(vals))
        self.assertEqual(result[0]['Contribution'], vals[0])

    def test_rejects_missing_or_reordered_rules(self):
        with self.assertRaises(ValueError):
            contribution_plot_rows(self.make_rows([0]*15))
        rows = self.make_rows([0]*16)
        rows[0]['Rule'] = 'Rule 2'
        with self.assertRaises(ValueError):
            contribution_plot_rows(rows)

    def test_rejects_nan_and_infinity(self):
        for val in (math.nan, math.inf, -math.inf):
            rows = self.make_rows([0]*16)
            rows[3]['Predicted-logit contribution'] = val
            with self.assertRaises(ValueError):
                contribution_plot_rows(rows)

if __name__ == '__main__':
    unittest.main()

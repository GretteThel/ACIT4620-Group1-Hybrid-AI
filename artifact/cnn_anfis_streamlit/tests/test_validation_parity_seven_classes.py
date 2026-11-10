"""Synthetic selection tests only; does not load HAM10000 or model weights."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import CLASSES
from validation_parity_seven_classes import select_one_per_class


class ValidationSelectionTests(unittest.TestCase):
    def test_exactly_seven_true_classes_sorted_deterministically(self):
        rows = []
        for label in CLASSES:
            rows.append({'image_id':f'{label}_z', 'true_dx':label})
            rows.append({'image_id':f'{label}_a', 'true_dx':label})
        chosen = select_one_per_class(rows)
        self.assertEqual([r['true_dx'] for r in chosen], list(CLASSES))
        self.assertEqual([r['image_id'] for r in chosen], [f'{c}_a' for c in CLASSES])

    def test_missing_class_rejected(self):
        rows = [{'image_id':f'{label}_a','true_dx':label} for label in CLASSES[:-1]]
        with self.assertRaises(ValueError):
            select_one_per_class(rows)

if __name__ == '__main__':
    unittest.main()

"""Synthetic and dry-run checks; no original features or held-out data are read."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_weight_decay_sweep as sweep


class WeightDecaySweepTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / 'run_weight_decay_sweep.py'), *map(str, arguments)],
            capture_output=True, text=True, check=False,
        )

    def test_dry_run_has_nine_matched_runs_and_no_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'untouched'
            result = self.invoke('--dry-run', '--features', Path(folder) / 'missing.zip',
                                 '--output-dir', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['total_training_runs'], 9)
            self.assertEqual(plan['rule_count'], 16)
            self.assertEqual(plan['seeds'], [42, 52, 62])
            self.assertEqual([job['weight_decay'] for job in plan['jobs']],
                             [0.0001, 0.0003, 0.001])
            self.assertFalse(output.exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            sentinel = output / 'existing.txt'
            sentinel.write_text('original', encoding='utf-8')
            result = self.invoke('--dry-run', '--features', output / 'missing.zip',
                                 '--output-dir', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'original')

    def test_downloaded_login_page_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'features.zip'
            archive.write_text('<html><title>Sign-in</title></html>', encoding='utf-8')
            output = Path(folder) / 'output'
            result = self.invoke('--features', archive, '--output-dir', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not a ZIP', result.stderr)
            self.assertFalse(output.exists())

    def test_summary_preserves_paired_seed_differences(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib-cache'))
            values = ([0.50, 0.55, 0.60], [0.52, 0.56, 0.60], [0.48, 0.53, 0.58])
            jobs = []
            for decay, scores in zip(sweep.WEIGHT_DECAYS, values):
                run_output = output / f'wd_{decay:.0e}'
                run_output.mkdir()
                rows = [dict(n_rules=16, seed=seed, macro_f1=score,
                             balanced_accuracy=0.6, melanoma_recall=0.4, nll=1.0,
                             best_epoch=10, epochs_completed=25, train_seconds=1.0)
                        for seed, score in zip(sweep.SEEDS, scores)]
                pd.DataFrame(rows).to_csv(run_output / 'e4_all_seed_metrics.csv', index=False)
                jobs.append({'weight_decay': decay, 'output_dir': str(run_output)})
            sweep.summarize(jobs, output)
            summary = pd.read_csv(output / 'weight_decay_summary.csv').set_index('weight_decay')
            self.assertAlmostEqual(summary.loc[0.0003, 'macro_f1_mean'], 0.56)
            self.assertAlmostEqual(summary.loc[0.0003, 'macro_f1_delta_vs_baseline_mean'], 0.01)
            self.assertAlmostEqual(summary.loc[0.001, 'macro_f1_delta_vs_baseline_mean'], -0.02)
            self.assertAlmostEqual(summary.loc[0.0001, 'macro_f1_std'], 0.05)
            records = pd.read_csv(output / 'weight_decay_all_seed_metrics.csv')
            self.assertEqual(len(records), 9)
            self.assertEqual((output / 'weight_decay_comparison.png').read_bytes()[:8],
                             b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__':
    unittest.main()

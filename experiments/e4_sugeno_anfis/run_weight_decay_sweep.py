"""Compare three ANFIS weight-decay values using original train/validation features."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


WEIGHT_DECAYS = (1e-4, 3e-4, 1e-3)
SEEDS = (42, 52, 62)
RULE_COUNT = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True,
                        help='Original Notebook 06 feature folder or ZIP')
    parser.add_argument('--output-dir', type=Path,
                        help='New directory for all nine runs and comparison plots')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the fixed experiment plan without reading data or writing outputs')
    args = parser.parse_args()
    experiment_dir = Path(__file__).resolve().parent
    trainer = experiment_dir / 'train_e4_anfis.py'
    features = args.features.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              experiment_dir.parents[1] / 'reports' / f'e4_weight_decay_{stamp}')
    if output.exists():
        parser.error('--output-dir must not already exist; original outputs are never reused')
    if not args.dry_run and not features.exists():
        parser.error('--features must point to the original Notebook 06 feature folder or ZIP')
    if not args.dry_run:
        required = {'train_features.npz', 'validation_features.npz',
                    'feature_extraction_config.json'}
        if features.is_dir():
            candidates = [path for path in features.rglob('feature_extraction_config.json')
                          if all((path.parent / name).is_file() for name in required)]
            if len(candidates) != 1:
                parser.error('--features must contain exactly one complete Notebook 06 feature folder')
        else:
            if not zipfile.is_zipfile(features):
                parser.error('--features is not a ZIP; check for a downloaded sign-in or error page')
            with zipfile.ZipFile(features) as archive:
                folders = {}
                for name in archive.namelist():
                    member = PurePosixPath(name)
                    folders.setdefault(str(member.parent), set()).add(member.name)
            if sum(required.issubset(names) for names in folders.values()) != 1:
                parser.error('ZIP must contain exactly one complete original Notebook 06 feature folder')
    jobs = []
    for decay in WEIGHT_DECAYS:
        label = f'wd_{decay:.0e}'
        run_output = output / label
        command = [sys.executable, str(trainer), '--features', str(features),
                   '--weight-decay', str(decay), '--rule-counts', str(RULE_COUNT),
                   '--seeds', *(str(seed) for seed in SEEDS),
                   '--output-dir', str(run_output)]
        jobs.append({'weight_decay': decay, 'label': label,
                     'output_dir': str(run_output), 'command': command})
    plan = {
        'purpose': 'Exploratory validation-only weight-decay sensitivity; not a new locked-test selection',
        'rule_count': RULE_COUNT, 'seeds': SEEDS, 'weight_decays': WEIGHT_DECAYS,
        'total_training_runs': len(SEEDS) * len(WEIGHT_DECAYS),
        'learning_rate': 1e-3, 'batch_size': 128, 'max_epochs': 100,
        'early_stopping_patience': 15, 'gradient_clip_norm': 5.0,
        'primary_comparison': 'Validation macro-F1 mean and sample standard deviation across matched seeds',
        'test_data_accessed': False,
        'features': str(features), 'output_dir': str(output), 'jobs': jobs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    plan['source_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (trainer, experiment_dir / 'anfis_core.py', Path(__file__).resolve())
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'experiment_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    status_path = output / 'sweep_status.json'
    status = {'status': 'running', 'completed_weight_decays': []}
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    try:
        for job in jobs:
            print(f"Training weight_decay={job['weight_decay']} across seeds {SEEDS}", flush=True)
            with (output / f"{job['label']}.log").open('w', encoding='utf-8') as log:
                subprocess.run(job['command'], cwd=experiment_dir, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            status['completed_weight_decays'].append(job['weight_decay'])
            status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        summarize(jobs, output)
        status['status'] = 'complete'
    except Exception as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(f'Comparison saved to {output}', flush=True)


def summarize(jobs: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    frames = []
    for job in jobs:
        frame = pd.read_csv(Path(job['output_dir']) / 'e4_all_seed_metrics.csv')
        if (len(frame) != len(SEEDS) or set(frame['seed']) != set(SEEDS)
                or not frame['n_rules'].eq(RULE_COUNT).all()):
            raise ValueError('Unexpected rule count or seeds in an experiment result')
        frame['weight_decay'] = job['weight_decay']
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    baseline = results.loc[results.weight_decay.eq(WEIGHT_DECAYS[0]),
                           ['seed', 'macro_f1']].rename(columns={'macro_f1': 'baseline_macro_f1'})
    results = results.merge(baseline, on='seed', validate='many_to_one')
    results['macro_f1_delta_vs_baseline'] = results.macro_f1 - results.baseline_macro_f1
    results.to_csv(output / 'weight_decay_all_seed_metrics.csv', index=False)
    metrics = ['macro_f1', 'balanced_accuracy', 'melanoma_recall', 'nll',
               'best_epoch', 'epochs_completed', 'train_seconds', 'macro_f1_delta_vs_baseline']
    summary = results.groupby('weight_decay')[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(output / 'weight_decay_summary.csv')
    positions = np.arange(len(WEIGHT_DECAYS))
    labels = [f'{value:.0e}' for value in WEIGHT_DECAYS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(positions, summary['macro_f1_mean'],
                yerr=summary['macro_f1_std'], capsize=5)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Validation macro-F1')
    axes[0].set_title('Mean and sample SD across three seeds')
    for seed in SEEDS:
        series = results.loc[results.seed.eq(seed)].set_index('weight_decay').loc[list(WEIGHT_DECAYS)]
        axes[1].plot(positions, series.macro_f1, marker='o', label=f'Seed {seed}')
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Validation macro-F1')
    axes[1].set_title('Matched seed results')
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel('Weight decay')
        axis.grid(axis='y', alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle('ANFIS: 16 rules, fixed training settings; validation only')
    fig.tight_layout()
    fig.savefig(output / 'weight_decay_comparison.png', dpi=180)
    plt.close(fig)
    print(summary[['macro_f1_mean', 'macro_f1_std',
                   'macro_f1_delta_vs_baseline_mean']].to_string(), flush=True)


if __name__ == '__main__':
    main()

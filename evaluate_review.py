"""Compare fuzzy review vs confidence/entropy/TTA threshold baselines.

Run: python evaluate_review.py --validation validation_predictions.npz \
    --test test_predictions.npz --out results

Required NPZ arrays per file:
  y_true: [n] integer in [0,6]
  probs: [n,7] calibrated original-image class probabilities
  tta_probs: [n,t,7] calibrated TTA view probabilities (at least 2 views)
  lesion_id: [n] string IDs (optional but recommended for overlap check)

This script NEVER fits thresholds using the test labels.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from fuzzy_review import (CLASS_CODES, scores_from_probs, batch_fuzzy_scores,
                          review_metrics, cutoff_for_review_budget, topk_review_mask)


def read_bundle(path):
    with np.load(path, allow_pickle=False) as d:
        fields = ('y_true', 'probs', 'tta_probs')
        if any(x not in d for x in fields):
            raise ValueError(f'{path}: requires {fields}')
        out = {x: d[x].copy() for x in fields}
        if 'lesion_id' in d:
            out['lesion_id'] = d['lesion_id'].astype(str)
    y = out['y_true']
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= 7).any():
        raise ValueError('y_true must be integer indices 0 to 6')
    scores_from_probs(out['probs'], out['tta_probs'])
    if len(y) != len(out['probs']):
        raise ValueError('length mismatch')
    return out


def score_dict(bundle):
    h,d = scores_from_probs(bundle['probs'], bundle['tta_probs'])
    return {
        'confidence': 1 - bundle['probs'].max(axis=1),
        'entropy': h,
        'tta_disagreement': d,
        'mamdani_fuzzy': batch_fuzzy_scores(bundle['probs'], bundle['tta_probs']),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validation', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', default='results')
    args = p.parse_args()
    val, test = read_bundle(args.validation), read_bundle(args.test)
    if 'lesion_id' in val and 'lesion_id' in test:
        overlap = set(val['lesion_id']) & set(test['lesion_id'])
        if overlap:
            raise RuntimeError(f'LEAKAGE: {len(overlap)} lesion IDs overlap')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vs, ts = score_dict(val), score_dict(test)
    rows = []
    for budget in (0.10, 0.20, 0.30, 0.40):
        for name in vs:
            threshold = cutoff_for_review_budget(vs[name], budget)
            # PRIMARY: fixed review workload for a fair *ranking* comparison.
            # Only test prediction scores are used, never test labels.
            review = topk_review_mask(ts[name], budget, tie_breaker=ts['entropy'])
            row = {'method':name, 'target_review_fraction':budget,
                   'validation_threshold_for_optional_operating_mode':threshold,
                   'evaluation_mode':'batch_top_k_exact_workload',
                   **review_metrics(test['y_true'],test['probs'],review)}
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'review_comparison.csv', index=False)
    preds = test['probs'].argmax(axis=1)
    report = {
        'class_mapping': dict(enumerate(CLASS_CODES)),
        'n_test': len(test['y_true']),
        'balanced_accuracy': float(balanced_accuracy_score(test['y_true'], preds)),
        'macro_f1': float(f1_score(test['y_true'], preds, labels=list(range(7)), average='macro', zero_division=0)),
        'melanoma_recall': float(recall_score(test['y_true'], preds, labels=[1], average='macro', zero_division=0)),
        'per_class_recall': {c: float(recall_score(test['y_true'], preds, labels=[i], average='macro', zero_division=0)) for i,c in enumerate(CLASS_CODES)},
    }
    try:
        report['macro_ovr_roc_auc'] = float(roc_auc_score(test['y_true'],test['probs'],labels=list(range(7)),multi_class='ovr',average='macro'))
    except ValueError:
        report['macro_ovr_roc_auc'] = None
    (out_dir / 'cnn_metrics.json').write_text(json.dumps(report, indent=2))
    print('Saved:', out_dir / 'review_comparison.csv', out_dir / 'cnn_metrics.json')
    print('\nCNN evaluation (test):', json.dumps(report, indent=2))
    print('\nDecision-layer comparison:')
    print(summary[['method','target_review_fraction','review_fraction','errors_captured_fraction','missed_melanomas_not_flagged_count']].to_string(index=False))

if __name__=='__main__':
    main()

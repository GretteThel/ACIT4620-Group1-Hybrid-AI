"""Read-only validation-set model-reproduction check, stratified by TRUE diagnosis.

This is a parity/engineering check, NOT a new performance study. It never
accesses the sealed test manifest. Run after installing the original model
assets and code. Do not pick models based on these results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from src.inference import CLASSES, load_bundle

REQUIRED_NAME = 'validation_anfis_r16_seed52.csv'
EXPECTED_VALIDATION_COUNT = 1441


def read_original_validation(reference: Path) -> list[dict[str, str]]:
    if reference.name != REQUIRED_NAME:
        raise ValueError(f'Refusing input other than original {REQUIRED_NAME}')
    with reference.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected_columns = {'image_id', 'lesion_id', 'true_dx', 'predicted_dx'} | {
            f'prob_{c}' for c in CLASSES
        }
        if not expected_columns.issubset(reader.fieldnames or []):
            raise ValueError('Missing original validation export columns')
        rows = list(reader)
    if len(rows) != EXPECTED_VALIDATION_COUNT:
        raise ValueError(f'Expected {EXPECTED_VALIDATION_COUNT} validation rows, got {len(rows)}')
    ids = [r['image_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate validation image IDs')
    for row in rows:
        if row['true_dx'] not in CLASSES or row['predicted_dx'] not in CLASSES:
            raise ValueError('Unknown diagnosis class in saved validation reference')
        probs = [float(row[f'prob_{c}']) for c in CLASSES]
        if not all(np.isfinite(p) and 0 <= p <= 1 for p in probs):
            raise ValueError('Invalid reference probabilities')
        if not np.isclose(sum(probs), 1.0, atol=1e-3):
            raise ValueError('Reference probabilities do not sum to one')
    return rows


def select_one_per_class(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select one deterministic validation row per TRUE class, not by result.

    This demonstrates multiple input types; it does not estimate class accuracy.
    """
    chosen: list[dict[str, str]] = []
    for label in CLASSES:
        matches = sorted((r for r in rows if r['true_dx'] == label), key=lambda r: r['image_id'])
        if not matches:
            raise ValueError(f'No validation image available for class {label}')
        chosen.append(matches[0])
    return chosen


def find_selected_images(raw: Path, chosen: list[dict[str, str]]) -> dict[str, Path]:
    if not raw.is_dir():
        raise FileNotFoundError(f'Original JPEG root missing: {raw}')
    targets = {r['image_id'] for r in chosen}
    located: dict[str, Path] = {}
    for folder in ('HAM10000_images_part_1', 'HAM10000_images_part_2'):
        source = raw / folder
        if not source.is_dir():
            raise FileNotFoundError(f'Expected original image directory: {source}')
        for image_id in targets:
            p = source / f'{image_id}.jpg'
            if p.is_file():
                if image_id in located:
                    raise ValueError(f'Duplicate original JPEG: {image_id}')
                located[image_id] = p
    if len(located) != len(targets):
        missing = sorted(targets - set(located))
        raise FileNotFoundError('Missing original validation images: ' + ', '.join(missing))
    return located


def run(args: argparse.Namespace) -> None:
    if not (0 < args.tolerance <= 0.01):
        raise ValueError('Tolerance must be positive and no greater than 0.01')
    rows = read_original_validation(args.reference)
    chosen = select_one_per_class(rows)
    images = find_selected_images(args.raw, chosen)
    torch.set_num_threads(4)
    bundle = load_bundle(args.assets, device='cpu')
    max_delta = 0.0
    for row in chosen:
        with Image.open(images[row['image_id']]) as original:
            got = bundle.predict_pil(original)
        reference_probs = np.asarray([float(row[f'prob_{c}']) for c in CLASSES], dtype=np.float64)
        predicted_probs = np.asarray([got['anfis_probabilities'][c] for c in CLASSES], dtype=np.float64)
        delta = float(np.max(np.abs(reference_probs - predicted_probs)))
        same = got['predicted_class'] == row['predicted_dx']
        print(f"{row['image_id']} | true={row['true_dx']:<5} | "
              f"saved={row['predicted_dx']:<5} | now={got['predicted_class']:<5} | "
              f"max Δ={delta:.8f} | identical predicted class={same}")
        max_delta = max(delta, max_delta)
        if not same or delta > args.tolerance:
            raise RuntimeError('STOP: Saved validation prediction was not reproduced. '
                               'Do not modify original weights or loosen the tolerance.')
    print(f'PASS: 7/7 predefined validation-class examples reproduced (max delta={max_delta:.8f}).')
    print('No held-out test manifest/images/labels were opened. This is NOT a model accuracy estimate.')


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--tolerance', type=float, default=0.01)
    return p


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except Exception as e:
        print(f'FAILED: {type(e).__name__}: {e}', file=sys.stderr)
        raise

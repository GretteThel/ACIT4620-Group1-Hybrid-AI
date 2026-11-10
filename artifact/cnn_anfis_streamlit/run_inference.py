"""CLI entry point for original frozen ACIT4620 CNN–ANFIS validation/inference.

Example:
 python run_inference.py check --assets model_assets
 python run_inference.py verify-validation --assets model_assets --raw <NB10>/data/raw \
   --reference <NB10>/data/reference/validation_anfis_r16_seed52.csv --limit 5
 python run_inference.py predict --assets model_assets --image <validation.jpg> --explain
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from src.inference import CLASSES, load_bundle, prepare_image

ROOT=Path(__file__).resolve().parent


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('check','predict','verify-validation'):
        a=sub.add_parser(name)
        a.add_argument('--assets',type=Path,default=ROOT/'model_assets')
        a.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        a.add_argument('--threads',type=int,default=4)
        if name=='predict':
            a.add_argument('--image',type=Path,required=True,
                help='An explicitly selected validation/demo image; do NOT use sealed test images before sign-off.')
            a.add_argument('--out',type=Path,default=ROOT/'outputs')
            a.add_argument('--explain',action='store_true')
        if name=='verify-validation':
            a.add_argument('--reference',type=Path,required=True)
            a.add_argument('--raw',type=Path,required=True)
            a.add_argument('--limit',type=int,default=5)
            a.add_argument('--tolerance',type=float,default=0.01)
    return p


def _safe_name(stem: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-','_') else '_' for c in stem)[:120]


def check(bundle):
    test=Image.new('RGB',(300,300), (140,109,116))
    x=prepare_image(test)
    result=bundle.predict_tensor(x)
    assert len(result['anfis_probabilities'])==7
    assert np.isclose(sum(result['anfis_probabilities'].values()),1.,atol=1e-5)
    assert np.isclose(sum(result['fuzzy']['normalized_rule_firing']),1.,atol=1e-5)
    assert len(result['fuzzy']['normalized_rule_firing'])==16
    print('PASS: exact frozen checkpoint/scaler fingerprints and provenance verified.')
    print('PASS: synthetic RGB tensor -> 1280-D CNN -> 16-rule ANFIS -> 7 probabilities.')
    print('WARNING: synthetic image is a technical smoke test, NOT validation accuracy evidence.')


def predict(bundle,args):
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    with Image.open(args.image) as img:
        x=prepare_image(img)
    result=bundle.predict_tensor(x)
    args.out.mkdir(exist_ok=True,parents=True)
    filename=_safe_name(args.image.stem)
    dest=args.out/f'{filename}_prediction.json'
    record={'input_filename':args.image.name,'original_model_sha256':bundle.checksums,
            'prediction':result,
            'warning':'Research demonstration only. Not a medical device, clinical diagnosis, or measure of calibrated risk.'}
    dest.write_text(json.dumps(record,indent=2),encoding='utf8')
    print(f"ANFIS prediction: {result['predicted_class']} | softmax = {result['predicted_probability']:.4f}")
    print('Predicted probabilities:')
    for code,p in result['anfis_probabilities'].items():print(f'  {code:<5}: {p:.4f}')
    print('Fuzzy highest firing rule:',result['fuzzy']['highest_firing_rule_index_1_based'])
    print('Saved:',dest)
    if args.explain:
        heat=bundle.gradcam(x,result['predicted_class_index'],head='anfis')
        overlay=bundle.make_overlay(x,heat)
        out=args.out/f'{filename}_anfis_gradcam.png'
        overlay.save(out)
        print('Saved:',out)
        print('Grad-CAM is a model attribution visualization, not a medical or causal explanation.')


def verify_validation(bundle,args):
    # Explicitly refuse other filenames to prevent accidentally browsing held-out test annotations.
    if args.reference.name != 'validation_anfis_r16_seed52.csv':
        raise ValueError('Reference MUST be original Notebook10 validation_anfis_r16_seed52.csv (not test).')
    if not (1<=args.limit<=50):
        raise ValueError('--limit must be between 1 and 50 (small validation-only parity check)')
    if not (0<args.tolerance<=0.01):
        raise ValueError('Tolerance must be positive and <=0.01; do not loosen benchmark agreement.')
    with args.reference.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=1441:
        raise ValueError('Expected the original 1,441-image validation reference, NOT test data.')
    cols=('image_id','predicted_dx')+tuple('prob_'+c for c in CLASSES)
    if not set(cols).issubset(rows[0]):
        raise ValueError('Validation reference missing required columns')
    seen={r['image_id'] for r in rows}
    if len(seen)!=1441:
        raise ValueError('Validation image IDs are not unique')
    chosen=rows[:args.limit]
    ids={r['image_id'] for r in chosen}
    matches={}
    for file in args.raw.rglob('*'):
        if file.is_file() and file.suffix.lower() in ('.jpg','.jpeg','.png') and file.stem in ids:
            if file.stem in matches:
                raise ValueError('Duplicate JPEG for image ID '+file.stem)
            matches[file.stem]=file
    if len(matches)!=len(chosen):
        raise FileNotFoundError(f'Missing {len(chosen)-len(matches)} validation JPEG(s) in {args.raw}.')
    worst=0.0
    for row in chosen:
        with Image.open(matches[row['image_id']]) as img:
            actual=bundle.predict_pil(img)
        expected=np.array([float(row['prob_'+c]) for c in CLASSES],dtype=np.float64)
        got=np.array([actual['anfis_probabilities'][c] for c in CLASSES],dtype=np.float64)
        err=float(np.max(np.abs(expected-got)))
        same=actual['predicted_class']==row['predicted_dx']
        print(f"{row['image_id']}: max probability delta={err:.6f}, class matched={same}")
        worst=max(worst,err)
        if err>args.tolerance or not same:
            raise RuntimeError('STOP: original validation prediction reconstruction failed. '
                'Check transforms, checkpoint authenticity and feature normalization.')
    print(f'PASS: {len(chosen)} original validation predictions reproduced (max delta={worst:.6f}).')
    print('No held-out test images/labels accessed.')


def main():
    a=build_parser().parse_args()
    if a.threads<1:raise ValueError('Use positive torch threads')
    torch.set_num_threads(a.threads)
    bundle=load_bundle(a.assets,device=a.device)
    if a.command=='check':check(bundle)
    elif a.command=='predict':predict(bundle,a)
    elif a.command=='verify-validation':verify_validation(bundle,a)
    else:raise ValueError(a.command)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise

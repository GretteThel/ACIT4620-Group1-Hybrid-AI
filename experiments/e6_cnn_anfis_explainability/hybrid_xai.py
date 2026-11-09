"""ACIT4620 Group 1 / Notebook 10.

Image-level Grad-CAM and exploratory occlusion for the original frozen E2a-50
EfficientNet-B0 CNN head and a first-order Takagi-Sugeno ANFIS head.

NO training. NO access to held-out test labels or images. All target images are
chosen from the previously saved 1,441-image validation prediction manifests.

Research caveat: Grad-CAM is an attribution method, not a clinical diagnosis or
causal explanation. Deletion/occlusion and stability are diagnostic proxies.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from scipy.stats import spearmanr
from torchvision import transforms
from torchvision.models import efficientnet_b0

from anfis_core import MultiOutputSugenoANFIS

CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
SPLIT_SHA = '265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0'
CNN_SHA = '2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
OUTPUT_NAME = 'ACIT4620_10_Explainability_Validation'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for part in iter(lambda: file.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def _load_torch(path: Path):
    # Checkpoints are project artifacts verified by source hashes and metadata.
    return torch.load(path, map_location='cpu', weights_only=True)


@dataclass
class Session:
    root: Path
    out: Path
    device: torch.device
    cnn: torch.nn.Module
    anfis_by_seed: dict[int, MultiOutputSugenoANFIS]
    mean: torch.Tensor
    std: torch.Tensor
    validation: pd.DataFrame
    checkpoint_hashes: dict[str, str]
    image_paths: dict[str, Path] | None = None
    chosen: pd.DataFrame | None = None
    heatmaps: dict[tuple[str, str, str], np.ndarray] | None = None


def verify_and_load(project_root: str | Path, device: str = 'cpu') -> Session:
    """Strictly validate the frozen checkpoint/source and prediction metadata."""
    root = Path(project_root).expanduser().resolve()
    assets = root / 'data' / 'model_assets'
    references = root / 'data' / 'reference'
    out = root / 'outputs' / OUTPUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    model_file = assets / 'e2a50_best_efficientnet_b0.pth'
    assert model_file.is_file(), f'Missing original E2a-50 checkpoint: {model_file}'
    assert sha256(model_file) == CNN_SHA, 'CNN file differs from original E2a-50 source!'
    assert device in ['cpu', 'cuda'], 'Device must be cpu or cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but CUDA is unavailable. Set device="cpu".')
    target_device = torch.device(device)

    cnn_ckpt = _load_torch(model_file)
    assert cnn_ckpt['class_codes'] == CLASSES
    assert cnn_ckpt['split_sha256'] == SPLIT_SHA
    cnn = efficientnet_b0(weights=None)
    n_features = cnn.classifier[1].in_features
    assert n_features == 1280, f'Unexpected EfficientNet-B0 pooled features: {n_features}'
    cnn.classifier[1] = torch.nn.Linear(1280, len(CLASSES))
    cnn.load_state_dict(cnn_ckpt['model_state'], strict=True)
    cnn.to(target_device).eval().requires_grad_(False)

    scale_file = assets / 'train_feature_scaler.npz'
    with np.load(scale_file, allow_pickle=False) as z:
        m = z['mean']; s = z['std']
    assert m.shape == s.shape == (1280,)
    assert np.isfinite(m).all() and np.isfinite(s).all() and (s > 0).all()
    mean = torch.as_tensor(m.astype(np.float32), device=target_device).view(1, -1)
    std = torch.as_tensor(s.astype(np.float32), device=target_device).view(1, -1)
    heads = {}
    checksums = {'e2a50_cnn': CNN_SHA, 'train_feature_scaler': sha256(scale_file)}
    for seed in (42, 52, 62):
        head_path = assets / f'anfis_r16_seed{seed}.pth'
        ckpt = _load_torch(head_path)
        assert ckpt['n_rules'] == 16 and ckpt['seed'] == seed
        assert ckpt['split_sha256'] == SPLIT_SHA
        assert ckpt['source_checkpoint_sha256'] == CNN_SHA
        assert ckpt['class_codes'] == CLASSES
        head = MultiOutputSugenoANFIS(1280, 8, 16, 7)
        head.load_state_dict(ckpt['state_dict'], strict=True)
        head.to(target_device).eval().requires_grad_(False)
        heads[seed] = head
        checksums[f'anfis_r16_seed{seed}'] = sha256(head_path)

    cnn_df = pd.read_csv(references / 'validation_cnn_e2a50.csv')
    anfis_df = pd.read_csv(references / 'validation_anfis_r16_seed52.csv')
    assert len(cnn_df) == len(anfis_df) == 1441
    required_cols = {'image_id', 'lesion_id', 'true_dx', 'predicted_dx', *[f'prob_{c}' for c in CLASSES]}
    assert required_cols.issubset(cnn_df.columns)
    assert required_cols.issubset(anfis_df.columns)
    assert cnn_df.image_id.is_unique and anfis_df.image_id.is_unique
    assert set(cnn_df.image_id) == set(anfis_df.image_id)
    reference = anfis_df.merge(cnn_df[['image_id','lesion_id','true_dx','predicted_dx']],
                               on='image_id', suffixes=('', '_cnn'), validate='one_to_one')
    assert (reference['lesion_id'] == reference['lesion_id_cnn']).all()
    assert (reference['true_dx'] == reference['true_dx_cnn']).all()
    assert reference.lesion_id.nunique() == 1067
    assert set(reference.true_dx) == set(CLASSES)
    for frame in (cnn_df, anfis_df):
        assert np.allclose(frame[[f'prob_{c}' for c in CLASSES]].sum(axis=1), 1.0, atol=1e-5)
        pred = np.asarray(CLASSES)[frame[[f'prob_{c}' for c in CLASSES]].to_numpy().argmax(axis=1)]
        assert np.array_equal(pred, frame.predicted_dx.to_numpy())
    reference['sample_group'] = ''
    selected_fields = ['image_id','lesion_id','true_dx','predicted_dx','predicted_dx_cnn','sample_group']
    print(f'PASS: E2a-50 checkpoint SHA-256 {CNN_SHA}')
    print(f'PASS: three 16-rule ANFIS seeds (42/52/62), same split, class order and feature source')
    print(f'PASS: {len(reference)} identical validation images, {reference.lesion_id.nunique()} lesions')
    print('No held-out test images, labels or predictions are accessed.')
    return Session(root, out, target_device, cnn, heads, mean, std, reference,
                   checksums, heatmaps={})


def choose_validation_images(session: Session, max_images: int = 13, seed: int = 20261109) -> pd.DataFrame:
    """Sample validation images only; favour melanoma TP/FN and rare classes.

    Choose unique lesions; no dependence on model explanations or test labels.
    Artifact status is only used if *independently annotated* by a reviewer.
    """
    if max_images < 2 or max_images > 50:
        raise ValueError('Use 2 <= max_images <= 50')
    frame = session.validation.copy()
    frame['sample_group'] = np.select([
        (frame.true_dx == 'mel') & (frame.predicted_dx != 'mel'),
        (frame.true_dx == 'mel') & (frame.predicted_dx == 'mel'),
        frame.true_dx == 'nv', frame.true_dx == 'bkl',
        frame.true_dx == 'bcc', frame.true_dx == 'akiec',
        frame.true_dx == 'vasc', frame.true_dx == 'df'
    ], ['melanoma_FN','melanoma_TP','nv','bkl','bcc','akiec','vasc','df'], default='other')
    quotas = [('melanoma_FN',4), ('melanoma_TP',3), ('nv',1), ('bkl',1),
              ('bcc',1), ('akiec',1), ('vasc',1), ('df',1)]
    rng = np.random.default_rng(seed)
    picked=[]; lesions=set()
    for group, quota in quotas:
        group_df=frame.loc[frame.sample_group == group]
        order = rng.permutation(len(group_df))
        take=0
        for idx in order:
            row=group_df.iloc[int(idx)]
            if row.lesion_id in lesions:
                continue
            picked.append(row); lesions.add(row.lesion_id);take += 1
            if len(picked) >= max_images or take >= quota:
                break
        if len(picked) >= max_images:
            break
    if len(picked) < max_images:
        for idx in rng.permutation(len(frame)):
            row=frame.iloc[int(idx)]
            if row.lesion_id not in lesions:
                picked.append(row); lesions.add(row.lesion_id)
                if len(picked) >= max_images: break
    selected=pd.DataFrame(picked).reset_index(drop=True)
    assert selected.image_id.is_unique and selected.lesion_id.is_unique
    annotations = session.root / 'data' / 'artifact_annotations.csv'
    selected['artifact_type'] = ''
    if annotations.exists():
        a = pd.read_csv(annotations, dtype=str).fillna('')
        if len(a):
            if not {'image_id','artifact_type','annotation_source'}.issubset(a.columns) or not a.image_id.is_unique:
                raise ValueError('Artifact annotation CSV must have unique image_id plus artifact_type and annotation_source')
            selected['artifact_type'] = selected.image_id.map(a.set_index('image_id')['artifact_type']).fillna('')
    selected.to_csv(session.out / 'sampled_validation_images.csv', index=False)
    session.chosen = selected
    print('Selected', len(selected), 'validation images across', selected.lesion_id.nunique(), 'unique lesions')
    print(selected[['image_id','true_dx','predicted_dx','sample_group','artifact_type']].to_string(index=False))
    if not selected['artifact_type'].astype(bool).any():
        print('NOTE: No image artifact labels available; do not claim an artifact-specific result.')
    return selected


def resolve_raw_image_files(session: Session, chosen: pd.DataFrame | None = None) -> dict[str,Path]:
    """Match only whitelisted validation image IDs to HAM10000 JPEG files."""
    if chosen is None: chosen = session.chosen
    if chosen is None: raise ValueError('Run choose_validation_images first')
    img_root = session.root / 'data' / 'raw'
    wanted=set(chosen.image_id.astype(str))
    mapping={}
    if not img_root.exists():
        img_root.mkdir(parents=True)
    for path in img_root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        if path.stem not in wanted:
            continue
        if path.stem in mapping:
            raise RuntimeError(f'Duplicate raw file for {path.stem}: {mapping[path.stem]} and {path}')
        mapping[path.stem]=path
    missing=wanted-set(mapping)
    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)}/{len(wanted)} selected VALIDATION images under {img_root}. '
            'Download original HAM10000 from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 '
            'and extract the HAM10000_images_part_1 and HAM10000_images_part_2 folders into data/raw. '
            f'Examples missing: {sorted(missing)[:6]} . Do not add these images to GitHub.'
        )
    session.image_paths=mapping
    print('PASS:',len(mapping),'selected validation images found. No held-out test image decoded.')
    return mapping


def load_image(session: Session, image_id: str) -> tuple[torch.Tensor,np.ndarray]:
    if session.image_paths is None or image_id not in session.image_paths:
        raise KeyError(f'Image {image_id} not in selected validation file set')
    with Image.open(session.image_paths[image_id]) as raw:
        corrected=ImageOps.exif_transpose(raw).convert('RGB')
        tensor=IMAGE_TRANSFORM(corrected)
    rgb=to_rgb(tensor)
    return tensor.unsqueeze(0).to(session.device), rgb


def to_rgb(x: torch.Tensor) -> np.ndarray:
    p=x.detach().cpu().squeeze(0).numpy()
    p=np.transpose(p,(1,2,0))
    return np.clip(p*np.asarray(STD)[None,None,:]+np.asarray(MEAN)[None,None,:], 0, 1).astype(np.float32)


def pooled_features(session: Session, x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    spatial=session.cnn.features(x)
    pooled=torch.flatten(session.cnn.avgpool(spatial),1)
    assert spatial.shape[1:] == (1280,7,7), spatial.shape
    return spatial, pooled


def forward_two_heads(session: Session, x: torch.Tensor, seed: int = 52):
    """Return uncalibrated CNN and first-order ANFIS logits, sharing ONE CNN forward."""
    _,pooled=pooled_features(session,x)
    return session.cnn.classifier(pooled), session.anfis_by_seed[seed]((pooled-session.mean)/session.std)


@torch.no_grad()
def check_prediction_reconstruction(session: Session, n_check: int = 3,
                                    max_abs_tolerance: float = 0.01) -> pd.DataFrame:
    """Fail closed if image preprocessing or checkpoint combination does not match E4."""
    if session.chosen is None or session.image_paths is None:
        raise ValueError('Choose images and resolve original JPEGs first')
    examples=session.chosen.iloc[:min(n_check,len(session.chosen))]
    rows=[]
    ref_cnn=pd.read_csv(session.root/'data/reference/validation_cnn_e2a50.csv').set_index('image_id')
    ref_e4=pd.read_csv(session.root/'data/reference/validation_anfis_r16_seed52.csv').set_index('image_id')
    for _,row in examples.iterrows():
        x,_=load_image(session,row.image_id)
        logits_cnn,logits_e4=forward_two_heads(session,x)
        probs={'cnn':torch.softmax(logits_cnn,1).cpu().numpy()[0],
               'e4':torch.softmax(logits_e4,1).cpu().numpy()[0]}
        for tag,ref in [('cnn',ref_cnn),('e4',ref_e4)]:
            expected=ref.loc[row.image_id,[f'prob_{c}' for c in CLASSES]].to_numpy(dtype=float)
            delta=float(np.max(np.abs(probs[tag]-expected)))
            rows.append({'image_id':row.image_id,'head':tag,'max_abs_probability_error':delta,
                         'reconstructed_pred':CLASSES[int(probs[tag].argmax())],
                         'saved_pred':ref.loc[row.image_id,'predicted_dx']})
    report=pd.DataFrame(rows)
    report.to_csv(session.out/'checkpoint_prediction_reconstruction.csv',index=False)
    print(report.to_string(index=False))
    worst=float(report.max_abs_probability_error.max())
    if worst > max_abs_tolerance:
        raise RuntimeError(f'FAIL: reconstructed predictions differ from saved exports by max {worst:.6f}, '
                           f'exceeding {max_abs_tolerance}. Check preprocessing, model weights and scaler; '
                           'do not interpret Grad-CAM until this is fixed.')
    assert (report.reconstructed_pred==report.saved_pred).all(), 'Argmax classes differ from saved predictions'
    print(f'PASS: saved CNN and ANFIS validation probabilities reproduced within {max_abs_tolerance}.')
    return report


def gradcam(session: Session, x: torch.Tensor, head: str, class_index: int, seed: int=52) -> np.ndarray:
    """Gradient from selected *pre-Softmax class logit* to last CNN spatial map.

    Uses the exact same EfficientNet spatial maps for CNN/ANFIS. Input requires
    gradient; CNN and ANFIS parameter weights remain frozen.
    """
    if not (0 <= class_index < len(CLASSES)): raise ValueError('Invalid class')
    if head not in ('cnn','anfis'): raise ValueError('Head must be cnn or anfis')
    with torch.enable_grad():
        source=x.detach().clone().requires_grad_(True)
        spatial,pooled=pooled_features(session,source)
        logits=(session.cnn.classifier(pooled) if head=='cnn'
                else session.anfis_by_seed[seed]((pooled-session.mean)/session.std))
        grads=torch.autograd.grad(logits[0,int(class_index)],spatial,retain_graph=False)[0]
        alpha=grads.mean(dim=(2,3),keepdim=True)
        weighted=(spatial*alpha).sum(dim=1,keepdim=True).clamp_min(0)
        up=F.interpolate(weighted,size=source.shape[-2:],mode='bilinear',align_corners=False)
        hm=up.detach().cpu().float().numpy()[0,0]
        lo=float(hm.min()); hi=float(hm.max())
        return ((hm-lo)/(hi-lo+1e-8)).astype(np.float32)


def heat_rgb(rgb: np.ndarray, cam: np.ndarray, alpha: float=0.47) -> np.ndarray:
    colormap=plt.get_cmap('magma')
    return np.clip((1-alpha)*rgb+alpha*colormap(cam)[...,:3],0,1)


def plot_explanation(session: Session, row: pd.Series, rgb: np.ndarray,
                     cnn_cam: np.ndarray, anfis_cam: np.ndarray,
                     pred_cam: np.ndarray|None=None):
    items=[('Original (validation)',rgb),('CNN Grad-CAM',heat_rgb(rgb,cnn_cam)),
           ('CNN → ANFIS Grad-CAM',heat_rgb(rgb,anfis_cam))]
    if pred_cam is not None:
        items.append(('ANFIS predicted-class CAM',heat_rgb(rgb,pred_cam)))
    fig,axes=plt.subplots(1,len(items),figsize=(4.4*len(items),4.1))
    for ax,(title,img) in zip(np.atleast_1d(axes),items):
        ax.imshow(img);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(f"{row.image_id} | true: {row.true_dx} | ANFIS: {row.predicted_dx} | target: "
                 f"{'mel' if row.true_dx=='mel' else row.predicted_dx}",fontsize=11)
    fig.tight_layout()
    fig.savefig(session.out/'gradcam_overlays'/f'{row.image_id}.png',dpi=150,bbox_inches='tight')
    plt.close(fig)


def selected_class(row: pd.Series) -> int:
    # For melanoma cases, inspect melanoma class (especially *false negatives*).
    # Otherwise inspect the first-order ANFIS originally predicted class.
    return CLASSES.index('mel' if row.true_dx=='mel' else row.predicted_dx)


def generate_gradcam_explanations(session:Session) -> pd.DataFrame:
    if session.chosen is None or session.image_paths is None:
        raise ValueError('Choose samples and resolve files first')
    (session.out/'gradcam_overlays').mkdir(exist_ok=True,parents=True)
    (session.out/'rule_contributions').mkdir(exist_ok=True,parents=True)
    metadata=[];heatmaps={}
    for _,row in session.chosen.iterrows():
        x,rgb=load_image(session,row.image_id)
        target=selected_class(row)
        cm_cnn=gradcam(session,x,'cnn',target)
        cm_e4=gradcam(session,x,'anfis',target,52)
        pred_cam=None
        if row.sample_group == 'melanoma_FN':
            pred_cam=gradcam(session,x,'anfis',CLASSES.index(row.predicted_dx),52)
        plot_explanation(session,row,rgb,cm_cnn,cm_e4,pred_cam)
        heatmaps[(row.image_id,'cnn',CLASSES[target])]=cm_cnn
        heatmaps[(row.image_id,'anfis52',CLASSES[target])]=cm_e4
        if pred_cam is not None:
            heatmaps[(row.image_id,'anfis_pred',row.predicted_dx)]=pred_cam

        with torch.no_grad():
            _,pooled=pooled_features(session,x)
            standardized=(pooled-session.mean)/session.std
            logits,projection,weights,consequents,signed,_=session.anfis_by_seed[52].fuzzy_details(standardized)
            contributions=signed[0,:,target].cpu().numpy()
            rule_weights=weights[0].cpu().numpy()
            original_logit=float(logits[0,target].cpu())
        assert np.isclose(contributions.sum(),original_logit,atol=5e-5)
        for i,(w,contrib) in enumerate(zip(rule_weights,contributions)):
            metadata.append({'image_id':row.image_id,'lesion_id':row.lesion_id,'true_dx':row.true_dx,
                             'predicted_dx':row.predicted_dx,'target_class':CLASSES[target],
                             'sample_group':row.sample_group,'artifact_type':row.artifact_type,
                             'rule_number':i+1,'normalized_rule_weight':float(w),
                             'signed_class_logit_contribution':float(contrib),
                             'target_class_logit':original_logit})
        order=np.argsort(np.abs(contributions))[::-1]
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.bar(np.arange(1,17),contributions,color=['steelblue' if z>=0 else 'coral' for z in contributions])
        ax.set_xticks(range(1,17));ax.axhline(0,color='black',lw=.8)
        ax.set_xlabel('Learned fuzzy rule');ax.set_ylabel('Signed contribution to target logit')
        ax.set_title(f"{row.image_id}: {CLASSES[target]} logit | strongest rule R{order[0]+1}")
        fig.tight_layout();fig.savefig(session.out/'rule_contributions'/f'{row.image_id}.png',dpi=160)
        plt.close(fig)
        print('Grad-CAM + signed rules:',row.image_id,'target',CLASSES[target])
    frame=pd.DataFrame(metadata)
    frame.to_csv(session.out/'sample_rule_contributions.csv',index=False)
    session.heatmaps=heatmaps
    print('PASS:',len(session.chosen),'matched validation-image explanations saved.')
    return frame


def _top_mask(cam:np.ndarray, quantile:float=0.2) -> np.ndarray:
    k=max(1,round(cam.size*quantile))
    a=np.zeros(cam.size,dtype=bool)
    a[np.argsort(cam.ravel())[-k:]]=True
    return a.reshape(cam.shape)


def _agreement(a:np.ndarray,b:np.ndarray) -> dict:
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    c=float(spearmanr(a,b).statistic) if a.std()>1e-8 and b.std()>1e-8 else float('nan')
    mask_a=_top_mask(a.reshape(224,224));mask_b=_top_mask(b.reshape(224,224))
    iou=float(np.logical_and(mask_a,mask_b).sum()/max(1,np.logical_or(mask_a,mask_b).sum()))
    return {'spearman':c,'top20_iou':iou,'flat_a':bool(np.ptp(a)<1e-8),'flat_b':bool(np.ptp(b)<1e-8)}


def evaluate_stability(session:Session, max_images:int=6, test_horizontal_flip:bool=True) -> pd.DataFrame:
    """Within-model horizontal flip + across-seed saliency agreement.

    Same target class for ALL maps for each image; rule indices across seeds
    are NOT assumed aligned (rule permutation ambiguity).
    """
    if session.heatmaps is None or not session.heatmaps:
        raise ValueError('Run generate_gradcam_explanations first')
    rows=[]
    for _,row in session.chosen.head(max_images).iterrows():
        x,_=load_image(session,row.image_id)
        target=selected_class(row)
        base=session.heatmaps[(row.image_id,'anfis52',CLASSES[target])]
        for seed in (42,62):
            other=gradcam(session,x,'anfis',target,seed)
            agree=_agreement(base,other)
            rows.append({'image_id':row.image_id,'true_dx':row.true_dx,
                         'comparison':f'ANFIS16 seed52 vs seed{seed}',
                         'target_class':CLASSES[target],**agree})
        if test_horizontal_flip:
            flipped=gradcam(session,torch.flip(x,dims=[3]),'anfis',target,52)
            flipped=np.fliplr(flipped)
            agree=_agreement(base,flipped)
            rows.append({'image_id':row.image_id,'true_dx':row.true_dx,
                         'comparison':'ANFIS16 seed52 original vs hflip-unflipped',
                         'target_class':CLASSES[target],**agree})
    df=pd.DataFrame(rows)
    df.to_csv(session.out/'gradcam_stability_selected_validation.csv',index=False)
    if len(df):
        summary=df.groupby('comparison')[['spearman','top20_iou']].agg(['mean','std','count'])
        summary.to_csv(session.out/'gradcam_stability_summary.csv')
        print(summary.round(3))
    print('NOTE: Rule numbers across random seeds are not comparable without explicit rule alignment.')
    return df


def occlusion_variants(x:torch.Tensor,cam:np.ndarray, fraction:float,
                       rng:np.random.Generator, grid:int=7) -> dict[str,torch.Tensor]:
    """Replace saliency-ranked 224x224 image tiles by a locally blurred input.

    Equal masking budget for top, bottom and random controls. All images are
    normalized with the same ImageNet preprocessing after blur replacement.
    """
    if x.shape!=(1,3,224,224): raise ValueError('Expected one 224x224 normalized image')
    if not (0<fraction<=0.5): raise ValueError('Occluded fraction must be within (0,0.5]')
    display_img=to_rgb(x)
    # Blur the 224x224 transformed view, not the original whole image.
    pil=Image.fromarray(np.uint8(np.clip(display_img,0,1)*255+0.5),'RGB')
    blur=np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=10)),dtype=np.float32)/255.0
    blur_t=torch.from_numpy(((blur-np.asarray(MEAN)[None,None,:])/np.asarray(STD)[None,None,:])
                            .astype(np.float32)).permute(2,0,1).to(x.device)
    # Divide image into 7x7 cells of 32x32 pixels.
    block=224//grid
    saliency=np.asarray(cam).reshape(grid,block,grid,block).mean(axis=(1,3)).reshape(-1)
    k=max(1,round(fraction*grid*grid))
    ranked=np.argsort(saliency)
    choices={'top':ranked[-k:],'bottom':ranked[:k],
             'random':rng.choice(grid*grid,size=k,replace=False)}
    result={}
    for method,inds in choices.items():
        copied=x[0].clone()
        for idx in inds:
            r,c=divmod(int(idx),grid)
            rs=slice(r*block,(r+1)*block);cs=slice(c*block,(c+1)*block)
            copied[:,rs,cs]=blur_t[:,rs,cs]
        result[method]=copied
    return result


@torch.no_grad()
def _forward_targets(session:Session,batch:torch.Tensor,seed:int=52):
    original,hybrid=forward_two_heads(session,batch,seed)
    return {'cnn':original,'anfis':hybrid}


def evaluate_occlusion(session:Session, max_images:int=8,
                       fractions:tuple[float,...]=(0.10,0.20,0.30)) ->pd.DataFrame:
    """Exploratory saliency deletion vs random/bottom controls. No test data."""
    if not session.heatmaps: raise ValueError('Run Grad-CAM section first')
    rows=[]
    for _,row in session.chosen.head(max_images).iterrows():
        x,_=load_image(session,row.image_id)
        target=selected_class(row)
        with torch.no_grad():
            baseline=_forward_targets(session,x)
        for head in ('cnn','anfis'):
            cam=session.heatmaps[(row.image_id,'cnn' if head=='cnn' else 'anfis52',CLASSES[target])]
            old_logit=baseline[head][0,target].item()
            old_prob=baseline[head].softmax(dim=1)[0,target].item()
            for fraction in fractions:
                # Deterministic random control, shared location for CNN/ANFIS in a sample.
                image_seed=int(hashlib.sha256(f'{row.image_id}-{fraction}'.encode()).hexdigest()[:8],16)
                rng=np.random.default_rng(image_seed)
                corrupted=occlusion_variants(x,cam,fraction,rng)
                batched=torch.stack([corrupted[k] for k in ('top','random','bottom')],dim=0)
                with torch.no_grad():
                    logits=_forward_targets(session,batched)[head]
                probs=logits.softmax(dim=1)
                for idx,method in enumerate(('top','random','bottom')):
                    rows.append({'image_id':row.image_id,'lesion_id':row.lesion_id,
                                 'sample_group':row.sample_group,'artifact_type':row.artifact_type,
                                 'true_dx':row.true_dx,'target_class':CLASSES[target],
                                 'head':head,'fraction_tiles_masked':fraction,'masking':method,
                                 'original_target_logit':old_logit,
                                 'occluded_target_logit':float(logits[idx,target].item()),
                                 'target_logit_drop':old_logit-float(logits[idx,target].item()),
                                 'original_target_probability':old_prob,
                                 'occluded_target_probability':float(probs[idx,target].item()),
                                 'target_probability_drop':old_prob-float(probs[idx,target].item())})
        print('Occlusion:',row.image_id)
    frame=pd.DataFrame(rows)
    frame.to_csv(session.out/'image_occlusion_validation.csv',index=False)
    if len(frame):
        summary=frame.groupby(['head','fraction_tiles_masked','masking'],as_index=False).agg(
            mean_probability_drop=('target_probability_drop','mean'),
            mean_logit_drop=('target_logit_drop','mean'),n_images=('image_id','nunique'))
        summary.to_csv(session.out/'image_occlusion_summary.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,head in zip(axes,('cnn','anfis')):
            subset=summary[summary['head']==head]
            for method in ('top','random','bottom'):
                part=subset[subset['masking']==method].sort_values('fraction_tiles_masked')
                ax.plot(part.fraction_tiles_masked,part.mean_probability_drop,
                        marker='o',label=method)
            ax.axhline(0,ls=':',color='grey',lw=1)
            ax.set_title(f'{head.upper()} — validation, exploratory')
            ax.set_xlabel('Fraction of 7×7 tiles blurred')
            ax.set_ylabel('Mean target-class probability drop')
            ax.legend()
        fig.suptitle('Occluding high-saliency vs random/bottom image regions')
        fig.tight_layout()
        fig.savefig(session.out/'image_occlusion_faithfulness.png',dpi=175,bbox_inches='tight')
        plt.close(fig)
        print(summary.round(4).to_string(index=False))
    print('WARNING: Occlusion is a limited proxy, not an independent causal or clinical validation.')
    return frame


def finalize(session:Session) -> Path:
    """Save exact source hashes, intended limitations and portable output ZIP."""
    info={
        'experiment':'ACIT4620_Notebook10_GradCAM_ANFIS_Validation',
        'research_question':'RQ3 - class-specific stability and explanation faithfulness',
        'architecture':'frozen E2a50 EfficientNet-B0 features -> train-only z-score -> 16-rule first-order Sugeno ANFIS',
        'class_order':CLASSES,'split_sha256':SPLIT_SHA,
        'checkpoint_sha256':session.checkpoint_hashes,
        'image_preprocessing':'EXIF-transposed RGB, Resize(256), CenterCrop(224), ToTensor, ImageNet normalization',
        'gradcam_target':'pre-Softmax class logit, last EfficientNet feature map, CAM ReLU and bilinear upsampling',
        'melanoma_false_negative_target':'true mel logit; predicted class also rendered separately',
        'faithfulness_proxy':'7x7 tiled input blur occlusion, top-saliency vs bottom/random controls; equal tile counts',
        'stability':'across ANFIS16 seeds 42/52/62 and unflipped horizontal-flip Grad-CAM, fixed target class',
        'test_labels_or_images_accessed':False,
        'limitations':[
            'Validation subset is small and purposely oversamples melanoma cases; not a population-level estimate.',
            'Bootstrap selection, model choice and Grad-CAM are assessed on reused validation data.',
            'Grad-CAM highlights associations, not clinically verified lesion characteristics or causation.',
            'Input blur masking may create distribution shift; results are sensitivity probes, not proof of faithfulness.',
            'Grad-CAM seed agreement does not establish that individual fuzzy-rule indices correspond across seeds.',
            'Artifact-specific conclusions require independent image-level artifact labels.',
        ],
        'versions':{'python':platform.python_version(),'torch':torch.__version__,
                    'torchvision':__import__('torchvision').__version__,'numpy':np.__version__,
                    'scikit_learn':sklearn.__version__},
        'selected_validation_image_ids':session.chosen.image_id.tolist() if session.chosen is not None else [],
    }
    (session.out/'experiment_provenance.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    archive=shutil.make_archive(str(session.root/'outputs'/f'{OUTPUT_NAME}_Outputs'),
                               'zip',root_dir=session.root/'outputs',base_dir=OUTPUT_NAME)
    print('Saved:',archive, '| MB:',round(Path(archive).stat().st_size/2**20,2))
    return Path(archive)


def smoke_test(project_root: str|Path) -> None:
    """Smoke-test genuine pretrained checkpoints without medical images or validation results."""
    session=verify_and_load(project_root)
    x=torch.randn(1,3,224,224,device=session.device)
    with torch.no_grad():
        cnn,e4=forward_two_heads(session,x)
        _,pooled=pooled_features(session,x)
        weights=session.anfis_by_seed[52].fuzzy_details((pooled-session.mean)/session.std)[2]
        assert cnn.shape==e4.shape==(1,7)
        assert torch.allclose(weights.sum(1),torch.ones(1),atol=1e-5)
    c1=gradcam(session,x,'cnn',1)
    c2=gradcam(session,x,'anfis',1)
    assert c1.shape==c2.shape==(224,224)
    assert np.isfinite(c1).all() and np.isfinite(c2).all()
    occluded=occlusion_variants(x,c2,0.2,np.random.default_rng(11))
    assert len(occluded)==3 and all(t.shape==(3,224,224) for t in occluded.values())
    print('PASS: real checkpoint integration, Gaussian rule normalization, both Grad-CAM paths, and image occlusion shape tests')
    print('This is only a RANDOM-TENSOR CODE SMOKE TEST; it is NOT a HAM10000 result.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--max-images',type=int,default=13)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.smoke_test:
        smoke_test(args.project_root)
    else:
        sess=verify_and_load(args.project_root,args.device)
        choose_validation_images(sess,args.max_images)
        resolve_raw_image_files(sess)
        check_prediction_reconstruction(sess)
        generate_gradcam_explanations(sess)
        evaluate_stability(sess)
        evaluate_occlusion(sess)
        finalize(sess)

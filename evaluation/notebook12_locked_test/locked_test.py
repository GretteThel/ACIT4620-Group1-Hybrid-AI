"""ACIT4620 Group 1 | Notebook 12 | pre-registered locked-test evaluation.

Two separate actions:
  freeze    -- reads validation-selected models and checkpoint metadata only;
               DOES NOT OPEN the sealed test manifest or original JPEGs.
  evaluate  -- requires a verified immutable freeze record + explicit acknowledgement;
               then accesses the predefined 1,458-image lesion-disjoint test set.

This script NEVER trains, changes thresholds, recalibrates, fits a scaler, or chooses
checkpoints from test outcomes. Uncalibrated, original-view softmax probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
from collections import Counter
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image, ImageOps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    average_precision_score, roc_auc_score, precision_recall_fscore_support,
    f1_score, confusion_matrix, log_loss)
from anfis_core import MultiOutputSugenoANFIS

HERE = Path(__file__).resolve().parent
CLASSES = ['nv','mel','bkl','bcc','akiec','vasc','df']
IDX = {c:i for i,c in enumerate(CLASSES)}
SPLIT_SHA = '265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0'
CNN_SHA = '2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9'
MANIFEST_MEMBER_SHA = '369463a5cd77e2de4912d04429bf3799742c9c805486529741df92ca43e8dcd6'
VALIDATION_SELECTION_SHA = 'b8891bf6e8cd41da5e748abe077365995c61fbaec99997d2f948b3fbf4aa0ec0'
E3_METRICS_SHA = '82089f1480b63b3a2b4888f859892c28c73bac86c09d7190c8cbd28d2c81ca85'
E4_METRICS_SHA = 'e367f8ddac1c29bd87e40554ba1f613634455c97f6f5c2b1ea7ef1e56df6e76d'
# Exact hashes archived by Notebook 11 for the original nine ANFIS checkpoints.
E4_PINNED_SHA = {
 'anfis_r16_seed42':'a819db98a4b8adb55d5e83af2ac2ea484ff9c685ac6d3b6497700c400dc31fc9',
 'anfis_r16_seed52':'020663d064b6e7cb5b12b70d84974dd652c7d78c3759d1f696ade08de786dc18',
 'anfis_r16_seed62':'9db9c31f5239f6754561bb4a3eca0f740816d486a8d5ab0d88c7cc76512a01cf',
 'anfis_r4_seed42':'49bbb577c5f95f7d2192c01e1314a8569e0783038082cf980ffa4af85cf9209f',
 'anfis_r4_seed52':'5ea9952f6019f9f4a032596c73e9b26f01ee78aa50cfb8dc0d551a332f586170',
 'anfis_r4_seed62':'866c63c3898a59762fc1233a7d011f61a8379b73d5e008be06032dfbbef0bbd3',
 'anfis_r8_seed42':'a11e178e2914a53c5e6a68f162105216f7dce587c91b0405376c5c3de0dc661d',
 'anfis_r8_seed52':'9d7fad5b066cc1afdd6efbd28517fcf399b68ad950ee1509e3f7e57013318f07',
 'anfis_r8_seed62':'2b8000d5c5632992dfe12289921e3e225380c15dab5a8e8e97e79a0c6dde88af',
}
E4_SCALER_SHA = '290c5e6b418bdfcb8fe5436d313cfecdc8661f5fbc5447a4529b2e6d889fb43f'
PRIMARY = ('cnn_e2a50','linear_seed42','mlp_seed42',
           'anfis_r4_seed52','anfis_r8_seed42','anfis_r16_seed52')
RUNS = ('cnn_e2a50',) + tuple(f'{h}_seed{s}' for h in ('linear','mlp') for s in (42,52,62)) + tuple(
    f'anfis_r{r}_seed{s}' for r in (4,8,16) for s in (42,52,62))
PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def require(condition:bool, message:str):
    if not condition:raise RuntimeError('STOP: '+message)


def safe_load(path:Path):
    return torch.load(path,map_location='cpu',weights_only=True)


def paths_for(repo:Path,nb11:Path)->dict[str,Path]:
    """Use existing E3 and Notebook11 original checkpoint assets; never train."""
    e3=repo/'experiments'/'e3_linear_mlp'/'outputs'/'ACIT4620_E3_Linear_MLP'
    assets=nb11/'data'/'model_assets'
    r={'cnn_e2a50':assets/'e2a50_best_efficientnet_b0.pth',
       'scaler_e3':e3/'train_feature_scaler.npz',
       'scaler_e4':assets/'train_feature_scaler.npz'}
    for model in RUNS:
        if model=='cnn_e2a50':continue
        r[model]=(e3/model/'best_head.pth') if model.startswith(('linear','mlp')) else (assets/f'{model}.pth')
    return r


def validate_assets(repo:Path,nb11:Path):
    paths=paths_for(repo,nb11)
    absent=[str(p) for p in paths.values() if not p.is_file()]
    require(not absent,'Missing original frozen assets. Do not substitute/retrain!\n'+'\n'.join(absent))
    checks={name:sha256_file(path) for name,path in paths.items()}
    require(checks['cnn_e2a50']==CNN_SHA,'CNN checkpoint SHA mismatch with E2a-50 source')
    for name,expected_sha in E4_PINNED_SHA.items():
        require(checks[name]==expected_sha,f'{name}: original E4 checkpoint SHA mismatch with Notebook11 source')
    require(checks['scaler_e4']==E4_SCALER_SHA,
            'Original Notebook11 E4 training-only scaler SHA mismatch')
    e3_path=HERE/'data'/'e3_all_seed_metrics.csv'
    e4_path=HERE/'data'/'e4_all_seed_metrics.csv'
    require(e3_path.is_file() and sha256_file(e3_path)==E3_METRICS_SHA,
            'Notebook09 E3 validation metrics source has been changed')
    require(e4_path.is_file() and sha256_file(e4_path)==E4_METRICS_SHA,
            'Notebook09 E4 validation metrics source has been changed')
    e3_scores=pd.read_csv(e3_path).set_index(['head','seed'])
    e4_scores=pd.read_csv(e4_path).set_index(['n_rules','seed'])
    cnn=safe_load(paths['cnn_e2a50'])
    require(cnn['split_sha256']==SPLIT_SHA and cnn['class_codes']==CLASSES,'CNN provenance mismatch')
    for name in RUNS:
        if name=='cnn_e2a50':continue
        ck=safe_load(paths[name]);seed=int(name.split('seed')[1]);rules=None
        require(ck['split_sha256']==SPLIT_SHA and ck['class_codes']==CLASSES,
                f'{name}: split or class order mismatch')
        if name.startswith(('linear','mlp')):
            kind=name.split('_')[0]
            require(ck['model_name']==kind and int(ck['seed'])==seed and
                    ck['feature_checkpoint_sha256']==CNN_SHA,f'{name}: original E3 checkpoint metadata mismatch')
            expected_val=float(e3_scores.loc[(kind,seed),'macro_f1'])
            require('val_macro_f1' in ck and abs(float(ck['val_macro_f1'])-expected_val)<2e-6,
                    f'{name}: best-checkpoint validation score differs from Notebook09 E3 archive')
            require(int(ck.get('best_epoch',-1))==int(e3_scores.loc[(kind,seed),'best_epoch']),
                    f'{name}: checkpoint epoch differs from Notebook09 source')
        else:
            rules=int(name.split('_')[1][1:])
            require(int(ck['n_rules'])==rules and int(ck['seed'])==seed and
                    ck['source_checkpoint_sha256']==CNN_SHA,f'{name}: original E4 checkpoint metadata mismatch')
            expected_val=float(e4_scores.loc[(rules,seed),'macro_f1'])
            require('val_macro_f1' in ck and abs(float(ck['val_macro_f1'])-expected_val)<2e-6,
                    f'{name}: best-checkpoint validation score differs from Notebook09 E4 archive')
            require(int(ck.get('best_epoch',-1))==int(e4_scores.loc[(rules,seed),'best_epoch']),
                    f'{name}: checkpoint epoch differs from Notebook09 source')
    scalers=[]
    for key in ('scaler_e3','scaler_e4'):
        with np.load(paths[key],allow_pickle=False) as z:
            mean=z['mean'].astype(np.float64);std=z['std'].astype(np.float64)
        require(mean.shape==std.shape==(1280,) and np.isfinite(mean).all() and
                np.isfinite(std).all() and (std>0).all(),f'{key}: invalid scaler')
        scalers.append((mean,std))
    require(np.allclose(scalers[0][0],scalers[1][0],atol=1e-5,rtol=1e-5) and
            np.allclose(scalers[0][1],scalers[1][1],atol=1e-5,rtol=1e-5),
            'Original E3 and E4 train-only feature scaling differ')
    return paths,checks


def selection_table()->pd.DataFrame:
    p=HERE/'data'/'validation_selected_checkpoints.csv'
    require(p.is_file() and sha256_file(p)==VALIDATION_SELECTION_SHA,
            'Notebook09 validation-only selection file was changed')
    df=pd.read_csv(p)
    require(tuple(df.model.astype(str))==PRIMARY,'Primary selected-model IDs differ from Notebook09')
    require(len(df)==6 and df.macro_f1.between(0,1).all(),
            'Notebook09 selection metrics corrupt')
    return df


def canonical_sha(data:dict)->str:
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),
        ensure_ascii=True).encode()).hexdigest()


def make_freeze(args):
    repo=Path(args.repo).expanduser().resolve();nb11=Path(args.nb11).expanduser().resolve()
    require(repo.is_dir(),'Team repository does not exist: '+str(repo))
    require(nb11.is_dir(),'Notebook11 project folder does not exist: '+str(nb11))
    selection=selection_table()
    paths,checks=validate_assets(repo,nb11)
    sealed=HERE/'data'/'sealed_split_manifest.zip'
    require(sealed.is_file(),'Missing sealed manifest package (test untouched)')
    # Deliberately hash ONLY archive bytes. DO NOT OPEN it until evaluation.
    sealed_sha=sha256_file(sealed)
    freeze_path=Path(args.freeze).expanduser().resolve()
    require(not freeze_path.exists(),'Freeze file already exists. Never overwrite or reselect after freezing: '+str(freeze_path))
    payload={
      'experiment':'ACIT4620_12_locked_test','status':'FROZEN_BEFORE_TEST',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'selection_source':'Notebook09 validation_selected_checkpoints.csv',
      'selection_source_sha256':VALIDATION_SELECTION_SHA,
      'selection_criterion':'Best validation macro-F1 within each predefined architecture/rule-count family. No test-based selection.',
      'primary_models':list(PRIMARY),
      'secondary_all_seed_models':list(RUNS),
      'validation_macro_f1':{str(r.model):float(r.macro_f1) for r in selection.itertuples()},
      'split_sha256':SPLIT_SHA,
      'expected_test_images':1458,
      'class_codes':CLASSES,
      'source_cnn_sha256':CNN_SHA,
      'inference_protocol':'Frozen CNN; Resize(256), CenterCrop(224), ImageNet mean/std; pooled 1280-D features; frozen train-only scaler; full precision; original-view uncalibrated softmax; no TTA, threshold tuning or recalibration.',
      'metrics':['macro_f1','balanced_accuracy','accuracy','weighted_f1','melanoma_recall',
                 'melanoma_precision','melanoma_f1','nll','brier','macro_ovr_auroc','macro_ovr_ap',
                 'class_precision_recall_f1','confusion_matrix'],
      'statistical_plan':'2,000 paired lesion-cluster bootstrap resamples (seed 20261110), 95% percentile CIs for macro-F1 and melanoma recall of primary models; paired differences ANFIS16 minus CNN and ANFIS16 minus MLP. Descriptive, not proof of statistical superiority.',
      'freeze_paths':{name:str(p) for name,p in paths.items()},
      'checkpoint_and_scaler_sha256':checks,
      'sealed_manifest_archive':str(sealed),
      'sealed_manifest_archive_sha256':sealed_sha,
      'sealed_manifest_member':'split_manifest_summary.csv',
      'sealed_manifest_member_sha256':MANIFEST_MEMBER_SHA,
      'test_images_or_labels_read_during_freeze':False,
      'test_evaluations_allowed':1,
    }
    record={'payload':payload,'payload_sha256':canonical_sha(payload)}
    freeze_path.parent.mkdir(parents=True,exist_ok=True)
    freeze_path.write_text(json.dumps(record,indent=2),encoding='utf8')
    print('PASS: freeze created; NO TEST LABELS/IMAGES ACCESSED')
    print('Freeze:',freeze_path)
    print('Frozen primary checkpoints:',', '.join(PRIMARY))
    print('All 16 frozen model assets and two train-only scalers were verified.')
    print('Before evaluating, review the freeze record and get team sign-off.')


def load_freeze(freeze_path:Path,repo:Path,nb11:Path)->tuple[dict,dict,dict]:
    require(freeze_path.is_file(),'Freeze file not found; run `freeze` first')
    saved=json.loads(freeze_path.read_text(encoding='utf8'))
    payload=saved.get('payload',{})
    require(saved.get('payload_sha256')==canonical_sha(payload),'Freeze record was modified after creation')
    require(payload.get('status')=='FROZEN_BEFORE_TEST','Pre-test lock not valid')
    require(payload['split_sha256']==SPLIT_SHA and payload['primary_models']==list(PRIMARY) and
            payload['secondary_all_seed_models']==list(RUNS) and
            payload['source_cnn_sha256']==CNN_SHA,'Freeze protocol conflicts with source')
    require(payload['selection_source_sha256']==VALIDATION_SELECTION_SHA,'Selection anchor mismatch')
    selection_table()
    paths,checks=validate_assets(repo,nb11)
    require(checks==payload['checkpoint_and_scaler_sha256'],'Model/scaler weights changed after freeze')
    require({n:str(p) for n,p in paths.items()}==payload['freeze_paths'],
            'Model paths changed after freeze; use the original checkout')
    sealed=HERE/'data'/'sealed_split_manifest.zip'
    require(str(sealed)==payload['sealed_manifest_archive'] and
            sha256_file(sealed)==payload['sealed_manifest_archive_sha256'],
            'Sealed manifest source changed after freeze')
    return payload,paths,checks


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__();self.classifier=nn.Linear(1280,7)
    def forward(self,x):return self.classifier(x)


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(1280,8)
        self.net=nn.Sequential(nn.ReLU(),nn.Linear(8,32),nn.ReLU(),nn.Dropout(0.15),nn.Linear(32,7))
    def forward(self,x):return self.net(self.projection(x))


def load_models(paths:dict,device:torch.device):
    c=safe_load(paths['cnn_e2a50'])
    cnn=efficientnet_b0(weights=None)
    require(cnn.classifier[1].in_features==1280,'Unexpected CNN pooled feature dimension')
    cnn.classifier[1]=nn.Linear(1280,7)
    cnn.load_state_dict(c['model_state'],strict=True)
    cnn=cnn.to(device).eval().requires_grad_(False)
    with np.load(paths['scaler_e4'],allow_pickle=False) as z:
        mean=torch.as_tensor(z['mean'].copy(),dtype=torch.float32,device=device).view(1,1280)
        std=torch.as_tensor(z['std'].copy(),dtype=torch.float32,device=device).view(1,1280)
    heads={}
    for name in RUNS:
        if name=='cnn_e2a50':continue
        ck=safe_load(paths[name])
        if name.startswith('linear'):head=LinearHead()
        elif name.startswith('mlp'):head=MLPHead()
        else:head=MultiOutputSugenoANFIS(1280,8,int(name.split('_')[1][1:]),7)
        head.load_state_dict(ck['state_dict'],strict=True)
        heads[name]=head.to(device).eval().requires_grad_(False)
    return cnn,heads,mean,std


def open_sealed_test_manifest(payload:dict)->pd.DataFrame:
    """Called only AFTER a verified pre-test freeze and explicit acknowledgement."""
    sealed=Path(payload['sealed_manifest_archive'])
    with zipfile.ZipFile(sealed) as zipf:
        members=zipf.namelist()
        require(members==[payload['sealed_manifest_member']],
                'Unexpected sealed manifest contents')
        raw=zipf.read(payload['sealed_manifest_member'])
    require(hashlib.sha256(raw).hexdigest()==MANIFEST_MEMBER_SHA,
            'Original E2a-50 split manifest differs from saved reference')
    manifest=pd.read_csv(io.BytesIO(raw))
    required={'image_id','lesion_id','dx','label','split'}
    require(required.issubset(manifest.columns),'Missing split manifest columns')
    require(len(manifest)==10015 and manifest.image_id.is_unique,'Split manifest image count/uniqueness error')
    require(manifest.split.value_counts().to_dict()=={'train':7116,'validation':1441,'test':1458},
            'Unexpected dataset split sizes')
    require(manifest.dx.isin(CLASSES).all(),'Unexpected diagnostic class')
    require((manifest.dx.map(IDX).to_numpy()==manifest.label.to_numpy(dtype=int)).all(),
            'Class indices differ from frozen order')
    lesion_sets={p:set(manifest.loc[manifest.split==p,'lesion_id']) for p in ['train','validation','test']}
    require(all(lesion_sets[a].isdisjoint(lesion_sets[b]) for a,b in
                [('train','validation'),('train','test'),('validation','test')]),
            'Lesion leakage across held-out split')
    test=manifest.loc[manifest.split=='test'].sort_values('image_id').reset_index(drop=True)
    require(len(test)==1458 and test.image_id.is_unique,'Test subset inconsistent')
    return test


def find_test_images(raw:Path,test:pd.DataFrame)->dict[str,Path]:
    require(raw.exists(),'Original HAM10000 raw image folder not found: '+str(raw))
    wanted=set(test.image_id.astype(str));files={}
    for p in raw.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png') and p.stem in wanted:
            require(p.stem not in files,f'Ambiguous duplicated test image ID: {p.stem}')
            files[p.stem]=p
    missing=sorted(wanted-files.keys())
    require(not missing,f'Missing {len(missing)} original test JPEGs; first missing: {missing[:8]}')
    return files


class TestDataset(Dataset):
    def __init__(self,test:pd.DataFrame,mapping:dict[str,Path]):
        self.test=test.reset_index(drop=True);self.mapping=mapping
    def __len__(self):return len(self.test)
    def __getitem__(self,idx):
        image_id=str(self.test.iloc[idx].image_id)
        with Image.open(self.mapping[image_id]) as f:
            rgb=ImageOps.exif_transpose(f).convert('RGB')
            x=PREDICT_TRANSFORM(rgb)
        return x,idx


def batch_predictions(test,paths,raw,device,batch_size,workers=0,threads=4):
    if device.type=='cpu':torch.set_num_threads(max(1,int(threads)))
    cnn,heads,mean,std=load_models(paths,device)
    mapping=find_test_images(raw,test)
    ds=TestDataset(test,mapping)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=='cuda')
    data={model:np.full((len(test),7),np.nan,dtype=np.float64) for model in RUNS}
    with torch.inference_mode():
        for batch_idx,(x,ids) in enumerate(loader,1):
            x=x.to(device,non_blocking=True)
            feat=torch.flatten(cnn.avgpool(cnn.features(x)),1)
            cnn_logits=cnn.classifier(feat)
            standardized=(feat-mean)/std
            logits={'cnn_e2a50':cnn_logits}
            for name,head in heads.items():logits[name]=head(standardized)
            dest=ids.numpy()
            for name,logit in logits.items():
                proba=torch.softmax(logit,dim=1).cpu().numpy().astype(np.float64)
                require(np.isfinite(proba).all() and np.allclose(proba.sum(axis=1),1,atol=1e-5),
                        f'Nonfinite or invalid softmax in {name} batch {batch_idx}')
                data[name][dest,:]=proba
            if batch_idx==1 or batch_idx%10==0 or batch_idx==len(loader):
                print(f'Original test images inferred: {min(batch_idx*batch_size,len(test))}/{len(test)}')
    for name,p in data.items():require(np.isfinite(p).all(),f'Incomplete output for {name}')
    return data


def performance(y:np.ndarray,p:np.ndarray)->dict:
    pred=np.argmax(p,axis=1)
    pr,re,f,_=precision_recall_fscore_support(y,pred,labels=np.arange(7),zero_division=0)
    onehot=np.eye(7)[y]
    # Some test classes can be rare: if an AUROC/AP is undefined, disclose N/A rather than fabricate.
    aucs=[];aps=[]
    for i in range(7):
        positive=onehot[:,i]
        if len(np.unique(positive))==2:
            aucs.append(float(roc_auc_score(positive,p[:,i])))
            aps.append(float(average_precision_score(positive,p[:,i])))
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=np.arange(7),average='macro',zero_division=0)),
            'balanced_accuracy':float(np.mean(re)),
            'weighted_f1':float(f1_score(y,pred,labels=np.arange(7),average='weighted',zero_division=0)),
            'melanoma_precision':float(pr[1]),'melanoma_recall':float(re[1]),'melanoma_f1':float(f[1]),
            'melanoma_tp':int(((y==1)&(pred==1)).sum()),
            'melanoma_fn':int(((y==1)&(pred!=1)).sum()),
            'nll_raw_original_view':float(log_loss(y,p,labels=np.arange(7))),
            'multiclass_brier':float(np.mean(np.sum((p-onehot)**2,axis=1))),
            'macro_ovr_auroc':float(np.mean(aucs)) if len(aucs)==7 else float('nan'),
            'macro_ovr_ap':float(np.mean(aps)) if len(aps)==7 else float('nan')}


def classwise(y,p,model):
    pred=p.argmax(axis=1)
    pr,re,f,s=precision_recall_fscore_support(y,pred,labels=np.arange(7),zero_division=0)
    return [{'model':model,'class':CLASSES[i],'support':int(s[i]),
             'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f[i])} for i in range(7)]


def cluster_bootstrap(test:pd.DataFrame,preds:dict,n_resamples=2000,seed=20261110)->pd.DataFrame:
    """Paired LESION-level resampling; CI is descriptive (validation-selected models)."""
    y=test['label'].to_numpy(dtype=int)
    groups=list(test.groupby('lesion_id',sort=True).indices.values())
    require(len(groups)>1,'Cannot bootstrap fewer than 2 lesions')
    rng=np.random.default_rng(seed)
    model_keys=list(PRIMARY)
    outcomes={k:[] for k in model_keys}
    differences={'anfis_r16_minus_mlp':[],'anfis_r16_minus_cnn':[]}
    mel_differences={'anfis_r16_minus_mlp':[],'anfis_r16_minus_cnn':[]}
    for _ in range(n_resamples):
        chosen=rng.integers(0,len(groups),size=len(groups))
        ind=np.concatenate([groups[j] for j in chosen])
        truth=y[ind]
        metrics={}
        for m in model_keys:
            p=preds[m][ind]
            pred=p.argmax(axis=1)
            macro=float(f1_score(truth,pred,average='macro',labels=np.arange(7),zero_division=0))
            mask=truth==1
            mel=float((pred[mask]==1).mean()) if mask.any() else float('nan')
            metrics[m]=(macro,mel)
            outcomes[m].append(metrics[m])
        for tag,b in [('anfis_r16_minus_mlp','mlp_seed42'),('anfis_r16_minus_cnn','cnn_e2a50')]:
            differences[tag].append(metrics['anfis_r16_seed52'][0]-metrics[b][0])
            mel_differences[tag].append(metrics['anfis_r16_seed52'][1]-metrics[b][1])
    def bounds(arr):
        a=np.asarray(arr,dtype=float);a=a[np.isfinite(a)]
        require(len(a)>n_resamples*0.95,'Insufficient bootstrap resamples to calculate interval')
        return np.quantile(a,[0.025,0.975]).tolist()
    rows=[]
    for m,values in outcomes.items():
        arr=np.asarray(values)
        for metric,col in [('macro_f1',0),('melanoma_recall',1)]:
            lo,hi=bounds(arr[:,col]);rows.append({'contrast':m,'metric':metric,
                'interval_2p5':lo,'interval_97p5':hi,'n_resamples':n_resamples})
    for tag in differences:
        for metric,v in [('macro_f1',differences[tag]),('melanoma_recall',mel_differences[tag])]:
            lo,hi=bounds(v);rows.append({'contrast':tag,'metric':metric,
                'interval_2p5':lo,'interval_97p5':hi,'n_resamples':n_resamples})
    return pd.DataFrame(rows)


def plot_cm(y,p,name,out:Path):
    cm=confusion_matrix(y,p.argmax(axis=1),labels=np.arange(7))
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(cm,cmap='Blues')
    ax.set_xticks(range(7),CLASSES);ax.set_yticks(range(7),CLASSES)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    ax.set_title(f'{name} — locked test (n=1,458)')
    vmax=int(cm.max())
    for i in range(7):
        for j in range(7):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
            color='white' if cm[i,j]>vmax/2 else 'black')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/f'confusion_{name}.png',dpi=180)
    plt.close(fig)


def plot_comparison(metrics:pd.DataFrame,per_class:pd.DataFrame,out:Path):
    selection=metrics.set_index('model').loc[list(PRIMARY)].reset_index()
    labels=['CNN','Linear','MLP','ANFIS 4','ANFIS 8','ANFIS 16']
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(6)
    vals=selection['macro_f1'].to_numpy()
    bars=ax.bar(x,vals,color='#3168a0')
    ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylim(0,1)
    ax.set_ylabel('Held-out test macro-F1');ax.set_title('Pre-specified six-model comparison')
    fig.tight_layout();fig.savefig(out/'selected_test_macro_f1.png',dpi=180);plt.close(fig)
    m=per_class.pivot(index='model',columns='class',values='recall').loc[list(PRIMARY),CLASSES]
    fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(m.to_numpy(),cmap='Blues',vmin=0,vmax=1)
    ax.set_yticks(range(len(PRIMARY)),labels);ax.set_xticks(range(7),CLASSES)
    for i in range(6):
        for j in range(7):
            val=float(m.iloc[i,j]);ax.text(j,i,f'{val:.2f}',ha='center',va='center',
                color='white' if val>0.55 else 'black',fontsize=9)
    ax.set_title('Per-class recall — untouched test partition')
    fig.colorbar(im,ax=ax,label='Recall');fig.tight_layout()
    fig.savefig(out/'selected_test_recall_heatmap.png',dpi=180);plt.close(fig)


def evaluate(args):
    require(args.confirm_locked_test is True,'Explicit --confirm-locked-test required')
    repo=Path(args.repo).expanduser().resolve();nb11=Path(args.nb11).expanduser().resolve()
    raw=Path(args.raw).expanduser().resolve()
    freeze_path=Path(args.freeze).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    payload,paths,checks=load_freeze(freeze_path,repo,nb11)
    require(not out.exists() or not any(out.iterdir()),
            f'Output directory nonempty; do not overwrite previously reported test results: {out}')
    require(int(args.batch_size)>=1 and int(args.batch_size)<=128,'Choose 1 <= batch-size <= 128')
    require(args.device in ('cpu','cuda'),'Device should be cpu or cuda')
    require(args.device!='cuda' or torch.cuda.is_available(),'CUDA requested but not available')
    # Freeze and output-gate validated before any test manifest or image access.
    out.mkdir(exist_ok=True,parents=True)
    audit={'event':'locked_test_evaluation_started','created_utc':datetime.now(timezone.utc).isoformat(),
           'freeze_file':str(freeze_path),'freeze_sha256':sha256_file(freeze_path),
           'device':args.device,'batch_size':args.batch_size,
           'torch_version':torch.__version__,'python_version':platform.python_version()}
    (out/'evaluation_started.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    # FIRST access to the original held-out test labels begins here.
    test=open_sealed_test_manifest(payload)
    print('PASS: authentic original test manifest opened only AFTER freeze; images:',len(test),
          '| lesions:',test.lesion_id.nunique())
    test[['image_id','lesion_id','dx','label']].rename(columns={'dx':'true_dx'}).to_csv(
        out/'held_out_test_manifest_used.csv',index=False)
    print('Checking original JPEG files...')
    device=torch.device(args.device)
    predicted=batch_predictions(test,paths,raw,device,args.batch_size,args.workers,args.cpu_threads)
    y=test.label.to_numpy(dtype=int)
    table=[];per_class=[]
    for name in RUNS:
        p=predicted[name]
        table.append({'model':name,**performance(y,p)})
        per_class.extend(classwise(y,p,name))
        pred=p.argmax(axis=1)
        export=test[['image_id','lesion_id','dx','label']].copy().rename(columns={'dx':'true_dx'})
        export['predicted_label']=pred
        export['predicted_dx']=[CLASSES[int(j)] for j in pred]
        for j,code in enumerate(CLASSES):export[f'prob_{code}']=p[:,j]
        (out/'all_model_predictions').mkdir(exist_ok=True)
        export.to_csv(out/'all_model_predictions'/f'{name}.csv',index=False)
        if name in PRIMARY:
            plot_cm(y,p,name,out)
    metric_df=pd.DataFrame(table);metric_df.to_csv(out/'all_16_models_test_metrics.csv',index=False)
    selection=metric_df.set_index('model').loc[list(PRIMARY)].reset_index()
    selection.to_csv(out/'six_prespecified_models_test_metrics.csv',index=False)
    perclass=pd.DataFrame(per_class);perclass.to_csv(out/'per_class_test_metrics.csv',index=False)
    family=metric_df.loc[metric_df.model!='cnn_e2a50'].copy()
    family['family']=family.model.str.replace(r'_seed\d+$','',regex=True)
    summary=family.groupby('family').agg(n_seeds=('model','count'),
         macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
         accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean'),
         melanoma_recall_mean=('melanoma_recall','mean'),melanoma_recall_sd=('melanoma_recall','std')).reset_index()
    summary.to_csv(out/'three_seed_family_test_summary.csv',index=False)
    plot_comparison(metric_df,perclass,out)
    bootstrap=cluster_bootstrap(test,predicted,n_resamples=args.bootstrap_repeats)
    bootstrap.to_csv(out/'paired_lesion_bootstrap_intervals.csv',index=False)
    metadata={'experiment':'ACIT4620_12_locked_test',
         'evaluation_completed_utc':datetime.now(timezone.utc).isoformat(),
         'freeze_sha256':sha256_file(freeze_path),
         'source_model_sha256':checks,'split_sha256':SPLIT_SHA,'manifest_member_sha256':MANIFEST_MEMBER_SHA,
         'test_images':len(test),'test_unique_lesions':int(test.lesion_id.nunique()),
         'class_order':CLASSES,'primary_models':list(PRIMARY),'secondary_models':list(RUNS),
         'batch_size':args.batch_size,'device':args.device,'bootstrap_repeats':args.bootstrap_repeats,
         'notes':['No test-based model selection, optimization, threshold tuning, calibration or training',
                  'Primary model checkpoint selection was fixed by validation macro-F1 in Notebook09',
                  'Model-probabilities are raw original-view softmax. 16 pre-registered heads, one frozen CNN.',
                  'Bootstrap resamples lesions, not independent images; confidence intervals descriptive.',
                  'This classification research tool is not for clinical diagnosis.',
                  'Artifact subgroup annotation postponed; no verified artifact-specific claims.']}
    (out/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='file_integrity_manifest.json']
    integrity={str(p.relative_to(out)):sha256_file(p) for p in files}
    (out/'file_integrity_manifest.json').write_text(json.dumps(integrity,indent=2),encoding='utf8')
    print('\nPASS: ONE pre-specified locked-test evaluation COMPLETED')
    print(selection[['model','macro_f1','balanced_accuracy','melanoma_recall']].to_string(index=False))
    print('\nSaved:',out)
    print('Do NOT change models, thresholds, training, or class selection using these test scores.')


def cli():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='cmd',required=True)
    fr=sub.add_parser('freeze',help='Lock 16 original heads and 6 selected models WITHOUT opening test set')
    fr.add_argument('--repo',required=True,help='GitHub repository root containing original E3 outputs')
    fr.add_argument('--nb11',required=True,help='Original Notebook11 project root containing all 9 ANFIS weights')
    fr.add_argument('--freeze',default=str(HERE/'outputs'/'pretest_freeze.json'))
    ev=sub.add_parser('evaluate',help='Evaluate test ONLY after freeze/signoff')
    ev.add_argument('--repo',required=True);ev.add_argument('--nb11',required=True)
    ev.add_argument('--raw',required=True,help='Path to original full HAM10000_images_part_* JPEG directories')
    ev.add_argument('--freeze',default=str(HERE/'outputs'/'pretest_freeze.json'))
    ev.add_argument('--out',default=str(HERE/'outputs'/'ACIT4620_12_Locked_Test_Results'))
    ev.add_argument('--device',choices=['cpu','cuda'],default='cpu');ev.add_argument('--batch-size',type=int,default=16)
    ev.add_argument('--workers',type=int,default=0);ev.add_argument('--cpu-threads',type=int,default=4)
    ev.add_argument('--bootstrap-repeats',type=int,default=2000)
    ev.add_argument('--confirm-locked-test',action='store_true',help='I confirm team sign-off and one-time locked-test unsealing')
    args=parser.parse_args()
    if args.cmd=='freeze':make_freeze(args)
    else:
        require(args.bootstrap_repeats==2000,'Statistical protocol frozen at exactly 2,000 bootstrap replicates')
        evaluate(args)

if __name__=='__main__':cli()

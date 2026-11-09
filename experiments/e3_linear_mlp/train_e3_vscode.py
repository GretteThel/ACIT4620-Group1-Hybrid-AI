"""ACIT4620 Notebook 07 - local Windows/Linux/macOS E3 linear & MLP experiment.
Run from this folder after installing requirements and placing Notebook 06 ZIP in data/.
"""

# --- SECTION 2 ---
import os, json, math, hashlib, random, time, shutil, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix, roc_auc_score,
    average_precision_score, log_loss)

PROJECT_ROOT = Path.cwd().resolve()
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / 'outputs' / 'ACIT4620_E3_Linear_MLP'
OUT.mkdir(parents=True, exist_ok=True)

# Set this only if the ZIP is outside data/ or your Downloads folder:
ZIP_OVERRIDE = None  # Example: Path(r'C:\Users\YourName\Downloads\ACIT4620_06_Shared_Features.zip')
EXPECTED_ZIP = 'ACIT4620_06_Shared_Features.zip'
SEEDS = [42, 52, 62]  # repeat with E4 ANFIS
CLASS_CODES = ['nv','mel','bkl','bcc','akiec','vasc','df']
EXPECTED_SPLIT_SHA256 = '265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0'
EXPECTED_SPLITS = {'train':7116, 'validation':1441}
FEATURE_DIM, BOTTLENECK, N_CLASSES = 1280, 8, 7
MAX_EPOCHS, PATIENCE = 100, 15
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY = 128, 1e-3, 1e-4
DROPOUT = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    torch.set_num_threads(min(4, os.cpu_count() or 2))
print('Project root:', PROJECT_ROOT)
print('Device:',DEVICE,'| Torch:',torch.__version__)
print('Seeds:',SEEDS,'| Output:',OUT)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Locate extracted feature directory first; otherwise unpack the ZIP locally.
def find_feature_configs():
    return sorted(p for p in DATA_DIR.rglob('feature_extraction_config.json')
                  if (p.parent / 'train_features.npz').is_file()
                  and (p.parent / 'validation_features.npz').is_file())

configs = find_feature_configs()
if not configs:
    candidate_zips = ([Path(ZIP_OVERRIDE).expanduser()] if ZIP_OVERRIDE else [
        DATA_DIR / EXPECTED_ZIP,
        PROJECT_ROOT / EXPECTED_ZIP,
        Path.home() / 'Downloads' / EXPECTED_ZIP,
    ])
    found_zips = [p for p in candidate_zips if p.is_file()]
    if not found_zips:
        raise FileNotFoundError(
            'Notebook 06 feature ZIP not found. Download '
            'ACIT4620_06_Shared_Features.zip from your Google Drive link '
            'and place it in the project data folder, then rerun this cell. '
            f'Searched: {candidate_zips}'
        )
    zip_path = found_zips[0]
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f'Invalid ZIP file: {zip_path}')
    extraction = DATA_DIR / 'extracted_notebook06'
    extraction.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            destination = (extraction / member.filename).resolve()
            if not destination.is_relative_to(extraction.resolve()):
                raise ValueError(f'Unsafe archive path: {member.filename}')
        archive.extractall(extraction)
    print('Unpacked Notebook 06 ZIP from:', zip_path)
    configs = find_feature_configs()

if len(configs) != 1:
    raise RuntimeError(f'Expected exactly one feature directory; found {len(configs)}: {configs}')
FEATURE_DIR = configs[0].parent
feature_config = json.loads(configs[0].read_text(encoding='utf-8'))
assert feature_config['split_sha256'] == EXPECTED_SPLIT_SHA256, 'Wrong dataset split'
assert feature_config['class_codes'] == CLASS_CODES, 'Wrong class order'
assert feature_config['feature_dimension'] == FEATURE_DIM, 'Wrong feature dimension'
assert feature_config['backbone_frozen'] is True, 'CNN was not frozen'
assert feature_config['test_set_extracted'] is False, 'The test set must remain locked'
print('PASS — found Notebook 06 feature files:', FEATURE_DIR)
print('Source CNN:', feature_config['source_model'])
print('CNN checkpoint SHA-256:',feature_config['checkpoint_sha256'])


# --- SECTION 4 ---
def read_npz(part):
    with np.load(FEATURE_DIR / f'{part}_features.npz', allow_pickle=False) as z:
        keys=set(z.files)
        assert {'X','y','image_id','lesion_id','dx'}.issubset(keys), keys
        return {k:z[k].copy() for k in ['X','y','image_id','lesion_id','dx']}

train = read_npz('train'); valid = read_npz('validation')
for part,data in [('train',train),('validation',valid)]:
    X,y = data['X'],data['y']
    assert X.shape==(EXPECTED_SPLITS[part],FEATURE_DIM), (part,X.shape)
    assert y.shape==(EXPECTED_SPLITS[part],), (part,y.shape)
    assert np.isfinite(X).all() and set(np.unique(y)).issubset(set(range(N_CLASSES)))
    assert len(set(data['image_id'].tolist()))==len(y)
    assert all(CLASS_CODES[int(label)]==str(dx) for label,dx in zip(y,data['dx']))
    print(part, 'X:',X.shape,'unique lesions:',len(set(data['lesion_id'].tolist())))
assert set(train['image_id'].tolist()).isdisjoint(set(valid['image_id'].tolist()))
assert set(train['lesion_id'].tolist()).isdisjoint(set(valid['lesion_id'].tolist()))

# Standardization fitted on the TRAIN features only: do NOT fit on validation/test.
mu = train['X'].astype(np.float64).mean(axis=0)
sigma = train['X'].astype(np.float64).std(axis=0)
sigma_safe = np.maximum(sigma, 1e-6)
X_train = ((train['X'].astype(np.float64)-mu)/sigma_safe).astype(np.float32)
X_val = ((valid['X'].astype(np.float64)-mu)/sigma_safe).astype(np.float32)
y_train=train['y'].astype(np.int64); y_val=valid['y'].astype(np.int64)
assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
np.savez_compressed(OUT/'train_feature_scaler.npz',mean=mu.astype(np.float32),std=sigma_safe.astype(np.float32))
class_counts=np.bincount(y_train,minlength=N_CLASSES)
weights=len(y_train)/(N_CLASSES*class_counts)
print('Train class counts:', dict(zip(CLASS_CODES,class_counts.tolist())))
print('Train class weights:', dict(zip(CLASS_CODES,np.round(weights,3).tolist())))
print('PASS: same split and labels, no lesion overlap, train-only normalization')


# --- SECTION 6 ---
class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier=nn.Linear(FEATURE_DIM,N_CLASSES)
    def forward(self,x): return self.classifier(x)

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(FEATURE_DIM,BOTTLENECK)
        self.net=nn.Sequential(nn.ReLU(),nn.Linear(BOTTLENECK,32),nn.ReLU(),
                               nn.Dropout(DROPOUT),nn.Linear(32,N_CLASSES))
    def forward(self,x): return self.net(self.projection(x))

def model_factory(name):
    return {'linear':LinearHead, 'mlp':MLPHead}[name]()

def metrics(y, probabilities):
    # Stable valid probability simplex for sklearn metrics
    probs = np.maximum(np.asarray(probabilities,dtype=np.float64),1e-12)
    probs /= probs.sum(axis=1,keepdims=True)
    pred=probs.argmax(axis=1)
    per=precision_recall_fscore_support(y,pred,labels=list(range(N_CLASSES)),zero_division=0)
    Y=np.eye(N_CLASSES,dtype=np.int64)[y]
    output={'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'melanoma_precision':float(per[0][1]),
            'melanoma_recall':float(per[1][1]),
            'melanoma_f1':float(per[2][1]),
            'melanoma_tp':int(((y==1)&(pred==1)).sum()),
            'melanoma_fn':int(((y==1)&(pred!=1)).sum()),
            'nll':float(log_loss(y,probs,labels=list(range(N_CLASSES)))),
            'multiclass_brier':float(np.mean(np.sum((probs-Y)**2,axis=1))),
            'macro_ovr_auroc':float(roc_auc_score(y,probs,labels=list(range(N_CLASSES)),multi_class='ovr',average='macro')),
            'macro_ovr_ap':float(np.mean([average_precision_score(Y[:,i],probs[:,i]) for i in range(N_CLASSES)]))}
    return output

@torch.inference_mode()
def validate(model, features, labels):
    model.eval()
    ds=TensorDataset(torch.from_numpy(features),torch.from_numpy(labels))
    loader=DataLoader(ds,batch_size=512,shuffle=False)
    logits=[]
    for x,_ in loader:
        logits.append(model(x.to(DEVICE)).float().cpu().numpy())
    z=np.concatenate(logits,axis=0).astype(np.float64)
    z-=z.max(axis=1,keepdims=True)
    p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
    return metrics(labels,p),p

for name in ['linear','mlp']:
    candidate=model_factory(name)
    print(name,'parameter count:',sum(p.numel() for p in candidate.parameters()))


# --- SECTION 8 ---
all_rows=[]
run_records=[]
criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=DEVICE))

def train_run(name,seed):
    seed_everything(seed)
    model=model_factory(name).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train)),
                      batch_size=BATCH_SIZE,shuffle=True,generator=generator,
                      num_workers=0,pin_memory=(DEVICE.type=='cuda'))
    best_f1=-1.0; best_epoch=0; patience_counter=0; history=[]
    run_dir=OUT/f'{name}_seed{seed}'; run_dir.mkdir(exist_ok=True)
    checkpoint=run_dir/'best_head.pth'
    start=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); total_loss=0.; total_n=0
        for batch_x,batch_y in loader:
            batch_x=batch_x.to(DEVICE,non_blocking=True)
            batch_y=batch_y.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(batch_x),batch_y)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*len(batch_y); total_n+=len(batch_y)
        val_metrics,_=validate(model,X_val,y_val)
        score=val_metrics['macro_f1']
        history.append({'epoch':epoch,'train_weighted_ce':total_loss/total_n,
                        **{f'val_{k}':v for k,v in val_metrics.items() if k in ['macro_f1','accuracy','balanced_accuracy','melanoma_recall']}})
        if score>best_f1+1e-8:
            best_f1=score; best_epoch=epoch; patience_counter=0
            torch.save({'state_dict':model.state_dict(),'model_name':name,'seed':seed,
                'best_epoch':best_epoch,'val_macro_f1':best_f1,
                'feature_checkpoint_sha256':feature_config['checkpoint_sha256'],
                'split_sha256':EXPECTED_SPLIT_SHA256,'class_codes':CLASS_CODES,
                'feature_scaler_file':'train_feature_scaler.npz'}, checkpoint)
        else:
            patience_counter+=1
        if epoch==1 or epoch%10==0 or patience_counter>=PATIENCE:
            print(f'{name:6s} seed={seed} epoch={epoch:03d}: loss={total_loss/total_n:.4f} val_macro_f1={score:.4f} best={best_f1:.4f}')
        if patience_counter>=PATIENCE: break
    best_state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(best_state['state_dict'],strict=True)
    model=model.to(DEVICE)
    score,p=validate(model,X_val,y_val)
    assert abs(score['macro_f1']-best_f1)<1e-9
    np.savez_compressed(run_dir/'validation_predictions.npz',
        y_true=y_val,probs=p.astype(np.float32),image_id=valid['image_id'],
        lesion_id=valid['lesion_id'],dx=valid['dx'])
    pred=p.argmax(axis=1)
    csv=pd.DataFrame({'image_id':valid['image_id'],'lesion_id':valid['lesion_id'],
        'true_dx':valid['dx'],'label':y_val,'predicted_dx':[CLASS_CODES[i] for i in pred],
        'predicted_label':pred})
    for i,k in enumerate(CLASS_CODES): csv[f'prob_{k}']=p[:,i]
    csv.to_csv(run_dir/'validation_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(run_dir/'training_history.csv',index=False)
    record={'head':name,'seed':seed,'best_epoch':best_epoch,'epochs_completed':len(history),
            'train_seconds':round(time.perf_counter()-start,2),
            'trainable_params':sum(q.numel() for q in model.parameters() if q.requires_grad), **score}
    (run_dir/'validation_metrics.json').write_text(json.dumps(record,indent=2))
    # one confusion matrix per head and seed
    cm=confusion_matrix(y_val,pred,labels=np.arange(N_CLASSES))
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap='Blues')
    ax.set_xticks(range(7),CLASS_CODES);ax.set_yticks(range(7),CLASS_CODES)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
    ax.set_title(f'{name.upper()}, seed {seed} — validation')
    for i in range(7):
        for j in range(7):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.colorbar(im,ax=ax);fig.tight_layout()
    fig.savefig(run_dir/'validation_confusion_matrix.png',dpi=170);plt.close(fig)
    return record

for head in ['linear','mlp']:
    for seed in SEEDS:
        rec=train_run(head,seed)
        all_rows.append(rec)
        print('Completed:',head,seed,'macro-F1=',round(rec['macro_f1'],4),
              'balanced accuracy=',round(rec['balanced_accuracy'],4),
              'melanoma recall=',round(rec['melanoma_recall'],4))
results=pd.DataFrame(all_rows)
results.to_csv(OUT/'e3_all_seed_metrics.csv',index=False)
print('PASS: all six validation-only head training runs complete')


# --- SECTION 10 ---
from IPython.display import display
cols=['accuracy','macro_f1','balanced_accuracy','melanoma_recall',
      'melanoma_precision','macro_ovr_auroc','macro_ovr_ap','nll']
summary=results.groupby('head')[cols].agg(['mean','std'])
summary.to_csv(OUT/'e3_seed_summary.csv')
display(results[['head','seed','best_epoch','epochs_completed','macro_f1',
                 'balanced_accuracy','melanoma_recall','accuracy','train_seconds']])
display(summary)
# Per-head representative checkpoint selected by validation macro-F1 only.
selected=(results.sort_values(['head','macro_f1','seed'],ascending=[True,False,True])
          .groupby('head',sort=False).head(1))
selected.to_csv(OUT/'e3_validation_selected_heads.csv',index=False)
display(selected[['head','seed','best_epoch','macro_f1','balanced_accuracy','melanoma_recall']])
print('NOTE: best-seed scores are validation-selected and are not unbiased test estimates.')


# --- SECTION 12 ---
settings={
    'experiment':'ACIT4620_E3_Linear_and_MLP_on_Frozen_E2a50_Features',
    'notebook06_source':str(FEATURE_DIR),
    'source_checkpoint_sha256':feature_config['checkpoint_sha256'],
    'split_sha256':EXPECTED_SPLIT_SHA256,
    'class_codes':CLASS_CODES,
    'train_images':len(y_train),'validation_images':len(y_val),
    'test_images_processed':0,'test_set_locked':True,
    'features':FEATURE_DIM,'train_only_standardization':True,
    'seeds':SEEDS,'architecture':{
        'linear':'1280 -> 7',
        'mlp':'1280 -> 8 -> ReLU -> 32 -> ReLU -> Dropout(0.15) -> 7'},
    'criterion':'training-count inverse-frequency class-weighted cross entropy',
    'class_weights':dict(zip(CLASS_CODES,[float(v) for v in weights])),
    'optimizer':'AdamW','learning_rate':LEARNING_RATE,
    'weight_decay':WEIGHT_DECAY,'batch_size':BATCH_SIZE,
    'max_epochs':MAX_EPOCHS,'early_stopping_patience':PATIENCE,
    'checkpoint_selection':'validation macro-F1 only, per head and seed',
    'note':'Use identical features/seed set and evaluation protocol for E4 ANFIS; no locked-test evaluation.'
}
(OUT/'e3_experiment_config.json').write_text(json.dumps(settings,indent=2))
required=['train_feature_scaler.npz','e3_all_seed_metrics.csv','e3_seed_summary.csv',
          'e3_validation_selected_heads.csv','e3_experiment_config.json']
for name in required:
    assert (OUT/name).is_file() and (OUT/name).stat().st_size>0, name
for head in ['linear','mlp']:
    for seed in SEEDS:
        for name in ['best_head.pth','validation_predictions.npz',
                     'validation_predictions.csv','validation_metrics.json',
                     'training_history.csv','validation_confusion_matrix.png']:
            assert (OUT/f'{head}_seed{seed}'/name).is_file(),(head,seed,name)
archive=shutil.make_archive(str(PROJECT_ROOT / 'outputs' / 'ACIT4620_E3_Linear_MLP_Outputs'),
                            'zip',root_dir=OUT.parent,base_dir=OUT.name)
print('PASS: 6 trained heads, prediction files, histories, metrics, model weights and metadata')
print('Output ZIP:',archive)
print('Size MB:',round(Path(archive).stat().st_size/1048576,2))
print('Next: preserve this ZIP. Use identical saved features in E4 ANFIS.')


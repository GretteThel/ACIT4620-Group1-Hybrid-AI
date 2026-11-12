"""ACIT4620 E4: seven-output first-order Takagi-Sugeno ANFIS on frozen E2a-50 features.

VS Code / Python 3.13 compatible. Requires saved Notebook 06 TRAIN and VALIDATION
features only. No EfficientNet retraining; no test data read. The submitted
proposal defines the architecture and joint Adam/backprop optimization.
"""

# %% 1. Imports and protocol constants
import argparse
import os, json, random, time, shutil, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss
)
from anfis_core import MultiOutputSugenoANFIS, initialize_premises_train_only, rule_ablation_statistics


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--rule-counts', type=int, nargs='+', default=[4, 8, 16])
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62])
parser.add_argument('--features', type=Path, help='Original Notebook 06 feature folder or ZIP')
parser.add_argument('--output-dir', type=Path, help='New, empty output directory')
args = parser.parse_args()
if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
    parser.error('--weight-decay must be finite and nonnegative')
if any(value < 2 for value in args.rule_counts) or len(set(args.rule_counts)) != len(args.rule_counts):
    parser.error('--rule-counts must contain unique integers of at least two')
if any(value < 0 or value >= 2**32 for value in args.seeds) or len(set(args.seeds)) != len(args.seeds):
    parser.error('--seeds must contain unique integers in [0, 2**32)')
if args.features is not None and not args.features.expanduser().exists():
    parser.error('--features must point to an existing original feature folder or ZIP')
if args.output_dir is not None:
    requested_output = args.output_dir.expanduser().resolve()
    if requested_output.exists() and (not requested_output.is_dir() or any(requested_output.iterdir())):
        parser.error('--output-dir must be a new or empty directory')

# The standalone script resolves its own folder; the notebook uses its workspace.
PROJECT_ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()
if not (PROJECT_ROOT / 'data').is_dir() and (PROJECT_ROOT / 'ACIT4620_08_E4_ANFIS_VSCode' / 'data').is_dir():
    PROJECT_ROOT = PROJECT_ROOT / 'ACIT4620_08_E4_ANFIS_VSCode'
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = (args.output_dir.expanduser().resolve() if args.output_dir is not None
       else PROJECT_ROOT / 'outputs' / 'ACIT4620_E4_ANFIS')
OUT.mkdir(parents=True, exist_ok=True)

# If necessary, supply the full local path to a Notebook 06 ZIP or feature folder.
FEATURE_SOURCE_OVERRIDE = args.features
EXPECTED_ZIP = 'ACIT4620_06_Shared_Features.zip'
EXPECTED_SPLIT_SHA256 = '265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0'
EXPECTED_CHECKPOINT_SHA256 = '2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9'
CLASS_CODES = ['nv','mel','bkl','bcc','akiec','vasc','df']
EXPECTED_SPLITS = {'train': 7116, 'validation': 1441}
FEATURE_DIM, LATENT_DIM, N_CLASSES = 1280, 8, 7
RULE_COUNTS = args.rule_counts
SEEDS = args.seeds
MAX_EPOCHS, PATIENCE = 100, 15
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY = 128, 1e-3, args.weight_decay
MAX_GRAD_NORM = 5.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    torch.set_num_threads(min(4, os.cpu_count() or 2))
print('Project root:', PROJECT_ROOT)
print('Device:', DEVICE, '| PyTorch:', torch.__version__)
print('Rule counts:', RULE_COUNTS, '| seeds:', SEEDS)
print('Output:', OUT)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% 2. Locate and verify Notebook 06 features (never recreate data splits)
def locate_feature_inputs():
    if FEATURE_SOURCE_OVERRIDE is not None:
        p = Path(FEATURE_SOURCE_OVERRIDE).expanduser().resolve()
        if p.is_dir():
            roots = [p]
        elif p.is_file() and zipfile.is_zipfile(p):
            roots = []
        else:
            raise FileNotFoundError(f'FEATURE_SOURCE_OVERRIDE not found: {p}')
    else:
        p = None
        roots = [DATA_DIR]

    def candidates():
        return sorted({x.resolve() for root in roots for x in root.rglob('feature_extraction_config.json')
                       if (x.parent / 'train_features.npz').is_file()
                       and (x.parent / 'validation_features.npz').is_file()})
    found = candidates()
    if not found:
        archives = ([p] if p is not None and p.is_file() else [
            DATA_DIR / EXPECTED_ZIP,
            PROJECT_ROOT / EXPECTED_ZIP,
            Path.home() / 'Downloads' / EXPECTED_ZIP,
        ])
        archives = [x for x in archives if x.is_file()]
        if not archives:
            raise FileNotFoundError(
                f'Notebook 06 features missing. Place the extracted feature folder or {EXPECTED_ZIP} '
                f'inside {DATA_DIR}. Do not download/retrain the CNN.'
            )
        archive_path = archives[0]
        extracted = DATA_DIR / 'extracted_notebook06'
        extracted.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(archive_path) as z:
            for info in z.infolist():
                dest = (extracted / info.filename).resolve()
                if not dest.is_relative_to(extracted.resolve()):
                    raise ValueError(f'Unsafe archive path: {info.filename}')
            z.extractall(extracted)
        print('Extracted Notebook 06 ZIP:', archive_path)
        roots = [DATA_DIR]
        found = candidates()
    if len(found) != 1:
        raise RuntimeError(f'Expected one Notebook 06 feature source; found {len(found)}: {found}')
    return found[0]

CONFIG_PATH = locate_feature_inputs()
FEATURE_DIR = CONFIG_PATH.parent
feature_config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
assert feature_config['split_sha256'] == EXPECTED_SPLIT_SHA256, 'Split SHA-256 mismatch'
assert feature_config['class_codes'] == CLASS_CODES, 'Seven-class order mismatch'
assert feature_config['feature_dimension'] == FEATURE_DIM, 'Expected 1280-dimensional features'
assert feature_config['backbone_frozen'] is True, 'Backbone must be frozen'
assert feature_config['test_set_extracted'] is False, 'Held-out test must be untouched'
assert feature_config['checkpoint_sha256'] == EXPECTED_CHECKPOINT_SHA256, 'Not the E2a-50 feature checkpoint used by E3'
print('PASS: Notebook 06 source:', FEATURE_DIR)
print('Source checkpoint SHA-256:', feature_config['checkpoint_sha256'])


def read_npz(split):
    with np.load(FEATURE_DIR / f'{split}_features.npz', allow_pickle=False) as data:
        expected = {'X','y','image_id','lesion_id','dx'}
        assert expected.issubset(data.files), (split, data.files)
        return {k:data[k].copy() for k in expected}

train = read_npz('train')
valid = read_npz('validation')
for split, obj in [('train',train),('validation',valid)]:
    assert obj['X'].shape == (EXPECTED_SPLITS[split],FEATURE_DIM), (split,obj['X'].shape)
    assert obj['y'].shape == (EXPECTED_SPLITS[split],)
    assert np.isfinite(obj['X']).all()
    assert np.isin(obj['y'],np.arange(N_CLASSES)).all()
    assert len(set(obj['image_id'].tolist())) == EXPECTED_SPLITS[split]
    assert all(CLASS_CODES[int(y)] == str(dx) for y,dx in zip(obj['y'],obj['dx']))
    print(split, 'X:', obj['X'].shape, 'unique lesions:',len(set(obj['lesion_id'].tolist())))
assert set(train['image_id'].tolist()).isdisjoint(set(valid['image_id'].tolist()))
assert set(train['lesion_id'].tolist()).isdisjoint(set(valid['lesion_id'].tolist()))
print('PASS: classes, counts, unique image IDs, and zero train/validation lesion-ID overlap')


# %% 3. Fit the *same* training-only standardization as E3
mu = train['X'].astype(np.float64).mean(axis=0)
sigma = np.maximum(train['X'].astype(np.float64).std(axis=0), 1e-6)
X_train = ((train['X'].astype(np.float64)-mu)/sigma).astype(np.float32)
X_val = ((valid['X'].astype(np.float64)-mu)/sigma).astype(np.float32)
y_train = train['y'].astype(np.int64)
y_val = valid['y'].astype(np.int64)
assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
np.savez_compressed(OUT / 'train_feature_scaler.npz',mean=mu.astype(np.float32),std=sigma.astype(np.float32))
train_counts = np.bincount(y_train, minlength=N_CLASSES)
assert (train_counts > 0).all()
class_weights = len(y_train)/(N_CLASSES*train_counts)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float32,device=DEVICE))
assert train_counts.tolist() == [4796,782,772,358,230,89,89], 'Unexpected TRAIN class distribution'
assert np.bincount(y_val,minlength=7).tolist() == [974,146,168,89,34,21,9], 'Unexpected VALIDATION class distribution'
print('Training samples:',len(y_train),'validation:',len(y_val),'class counts:',train_counts.tolist())
print('Class weights:',dict(zip(CLASS_CODES,np.round(class_weights,3).tolist())))
print('PASS: train-only normalization and weighted cross-entropy; no test set accessed')


# %% 4. Evaluation functions shared with E3

def compute_metrics(y, probabilities):
    probs = np.maximum(np.asarray(probabilities,dtype=np.float64), 1e-12)
    probs /= probs.sum(axis=1,keepdims=True)
    pred = probs.argmax(axis=1)
    precision,recall,f1,_ = precision_recall_fscore_support(
        y,pred,labels=list(range(N_CLASSES)),zero_division=0)
    y_binary = np.eye(N_CLASSES,dtype=np.int64)[y]
    return {
        'accuracy': float(accuracy_score(y,pred)),
        'macro_f1': float(f1_score(y,pred,average='macro',zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
        'weighted_f1': float(f1_score(y,pred,average='weighted',zero_division=0)),
        'melanoma_precision': float(precision[1]),
        'melanoma_recall': float(recall[1]),
        'melanoma_f1': float(f1[1]),
        'melanoma_tp': int(((y==1)&(pred==1)).sum()),
        'melanoma_fn': int(((y==1)&(pred!=1)).sum()),
        'nll': float(log_loss(y,probs,labels=list(range(N_CLASSES)))),
        'multiclass_brier': float(np.mean(np.sum((probs-y_binary)**2,axis=1))),
        'macro_ovr_auroc': float(roc_auc_score(
            y,probs,labels=list(range(N_CLASSES)),multi_class='ovr',average='macro')),
        'macro_ovr_ap': float(np.mean([
            average_precision_score(y_binary[:,i],probs[:,i]) for i in range(N_CLASSES)
        ])),
    }


@torch.inference_mode()
def validate(model):
    model.eval()
    outputs=[]
    for start in range(0,len(X_val),512):
        x = torch.from_numpy(X_val[start:start+512]).to(DEVICE)
        outputs.append(model(x).cpu())
    p = torch.softmax(torch.cat(outputs,dim=0).double(),dim=1).numpy()
    return compute_metrics(y_val,p),p


# %% 5. Rule-count and trainable-parameter audit
for n_rules in RULE_COUNTS:
    model = MultiOutputSugenoANFIS(FEATURE_DIM,LATENT_DIM,n_rules,N_CLASSES)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'R={n_rules:<2}: trainable head parameters = {total:,}; projection = 1280 -> 8; outputs = 7')
print('R=8 is close to the E3 MLP head parameter count (10,767).')


# %% 6. Joint Adam training, 3 seeds x 3 rule counts, validation macro-F1 selection

def export_rule_analysis(model, run_dir, probabilities):
    model.eval()
    logits_list=[]; projected_list=[]; weights_list=[]; consequent_list=[]; contribution_list=[]; log_firing_list=[]
    with torch.no_grad():
        for start in range(0,len(X_val),512):
            batch=torch.from_numpy(X_val[start:start+512]).to(DEVICE)
            logits,z,w,f,c,log_firing = model.fuzzy_details(batch)
            logits_list.append(logits.cpu()); projected_list.append(z.cpu())
            weights_list.append(w.cpu()); consequent_list.append(f.cpu()); contribution_list.append(c.cpu()); log_firing_list.append(log_firing.cpu())
    logits=torch.cat(logits_list)
    projected=torch.cat(projected_list)
    weights=torch.cat(weights_list)
    consequents=torch.cat(consequent_list)
    contributions=torch.cat(contribution_list)
    log_firings=torch.cat(log_firing_list)
    reconstructed=contributions.sum(dim=1)
    assert torch.allclose(logits,reconstructed,atol=5e-5,rtol=1e-5)
    assert torch.allclose(weights.sum(dim=1),torch.ones(len(X_val)),atol=2e-6)
    regenerated=torch.softmax(logits.double(),dim=1).numpy()
    assert np.allclose(regenerated,probabilities,atol=1e-5)
    assert np.isfinite(contributions.numpy()).all()

    np.savez_compressed(run_dir/'validation_rule_explanations.npz',
        image_id=valid['image_id'],lesion_id=valid['lesion_id'],y_true=y_val,
        projected_features=projected.numpy(),
        normalized_rule_weights=weights.numpy(),
        rule_consequents=consequents.numpy(),
        signed_rule_contributions=contributions.numpy(),
        class_logits=logits.numpy())

    # Per-example signed contributions for the model's originally predicted class.
    predicted=logits.argmax(dim=1)
    signed_selected=contributions.gather(2,
        predicted[:,None,None].expand(-1,model.n_rules,1)).squeeze(-1).numpy()
    frame=pd.DataFrame({'image_id':valid['image_id'],'lesion_id':valid['lesion_id'],
                        'true_dx':valid['dx'],
                        'predicted_dx':[CLASS_CODES[int(i)] for i in predicted.numpy()]})
    for r in range(model.n_rules):
        frame[f'normalized_weight_r{r+1}']=weights[:,r].numpy()
        frame[f'signed_predicted_logit_contribution_r{r+1}']=signed_selected[:,r]
    frame.to_csv(run_dir/'validation_rule_contributions.csv',index=False)

    class_means=[]
    for k,name in enumerate(CLASS_CODES):
        mask=(y_val==k)
        row={'true_class':name,'validation_support':int(mask.sum())}
        for r in range(model.n_rules):
            row[f'avg_weight_rule_{r+1}']=float(weights.numpy()[mask,r].mean())
        class_means.append(row)
    pd.DataFrame(class_means).to_csv(run_dir/'rule_activation_by_class.csv',index=False)
    parameter_rows=[]
    centers=model.centers.detach().cpu().numpy()
    widths=model.widths().detach().cpu().numpy()
    for r in range(model.n_rules):
        for d in range(LATENT_DIM):
            parameter_rows.append({'rule':r+1,'latent_dim':d+1,
                                   'gaussian_center':float(centers[r,d]),
                                   'gaussian_width':float(widths[r,d])})
    pd.DataFrame(parameter_rows).to_csv(run_dir/'learned_gaussian_premises.csv',index=False)

    ablation,per_image=rule_ablation_statistics(logits,weights,consequents,log_firings)
    ablation['design']=(
        'Remove either maximum-absolute signed-contribution or minimum-absolute '
        'signed-contribution rule for the original predicted class; renormalize '
        'surviving firing weights; freeze learned parameters.'
    )
    ablation['limitation']=(
        'Latent-rule ablation is a diagnostic faithfulness proxy, not clinical '
        'causal validation or image-level occlusion.'
    )
    (run_dir/'rule_ablation_faithfulness.json').write_text(json.dumps(ablation,indent=2))
    pd.DataFrame({'image_id':valid['image_id'],**per_image}).to_csv(
        run_dir/'validation_rule_ablation.csv',index=False)

    fig,ax=plt.subplots(figsize=(max(6,model.n_rules*0.6),4.5))
    grid=np.array([[row[f'avg_weight_rule_{r+1}'] for r in range(model.n_rules)]
                   for row in class_means])
    im=ax.imshow(grid,aspect='auto')
    ax.set_yticks(range(N_CLASSES),CLASS_CODES)
    ax.set_xticks(range(model.n_rules),[f'R{r+1}' for r in range(model.n_rules)],rotation=45)
    ax.set_title('Mean normalized fuzzy-rule firing on validation by true class')
    ax.set_xlabel('Learned fuzzy rule');fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(run_dir/'rule_activation_by_class.png',dpi=170);plt.close(fig)
    return ablation


def run_one(n_rules,seed):
    seed_everything(seed)
    model=MultiOutputSugenoANFIS(FEATURE_DIM,LATENT_DIM,n_rules,N_CLASSES).to(DEVICE)
    init_meta=initialize_premises_train_only(model,X_train,DEVICE,seed)
    run_dir=OUT / f'anfis_r{n_rules}_seed{seed}'
    run_dir.mkdir(exist_ok=True)
    (run_dir/'premise_initialization.json').write_text(json.dumps(init_meta,indent=2))
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    train_loader=DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train)),
                            batch_size=BATCH_SIZE,shuffle=True,
                            generator=torch.Generator().manual_seed(seed),
                            num_workers=0,pin_memory=(DEVICE.type=='cuda'))
    checkpoint=run_dir/'best_anfis_head.pth'
    if DEVICE.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    best_f1=-1.0;best_epoch=0;bad_epochs=0;history=[]
    start=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        model.train();running_loss=0.0;seen=0
        for batch_x,batch_y in train_loader:
            batch_x=batch_x.to(DEVICE,non_blocking=True)
            batch_y=batch_y.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch_x)
            loss=criterion(logits,batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: R={n_rules} seed={seed} epoch={epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
            optimizer.step()
            running_loss+=float(loss.item())*len(batch_y);seen+=len(batch_y)
        score,_=validate(model)
        macro=score['macro_f1']
        history.append({'epoch':epoch,'train_weighted_ce':running_loss/seen,
                        'val_macro_f1':macro,'val_accuracy':score['accuracy'],
                        'val_balanced_accuracy':score['balanced_accuracy'],
                        'val_melanoma_recall':score['melanoma_recall']})
        if macro > best_f1 + 1e-8:
            best_f1=macro;best_epoch=epoch;bad_epochs=0
            torch.save({
                'state_dict':model.state_dict(),'n_rules':n_rules,'seed':seed,
                'best_epoch':best_epoch,'val_macro_f1':best_f1,
                'source_checkpoint_sha256':feature_config['checkpoint_sha256'],
                'split_sha256':EXPECTED_SPLIT_SHA256,'class_codes':CLASS_CODES,
                'architecture':'linear projection (1280->8), Gaussian premises, '
                               'product rule firing, seven first-order Sugeno consequents',
                'optimizer':'Adam end-to-end/backprop, NOT classical LSE+GD',
            },checkpoint)
        else:
            bad_epochs+=1
        if epoch==1 or epoch%10==0 or bad_epochs>=PATIENCE:
            print(f'R={n_rules:<2} seed={seed} epoch={epoch:03d} '
                  f'train_ce={running_loss/seen:.4f} val_macro_f1={macro:.4f} best={best_f1:.4f}')
        if bad_epochs>=PATIENCE:
            break
    training_seconds = time.perf_counter() - start
    peak_training_gpu_mb = (float(torch.cuda.max_memory_allocated()/1048576)
                            if DEVICE.type=='cuda' else None)

    best=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(best['state_dict'],strict=True)
    model=model.to(DEVICE)
    final,probabilities=validate(model)
    assert abs(final['macro_f1']-best_f1) < 1e-9
    prediction=probabilities.argmax(axis=1)
    np.savez_compressed(run_dir/'validation_predictions.npz',
        y_true=y_val,probs=probabilities.astype(np.float32),
        image_id=valid['image_id'],lesion_id=valid['lesion_id'],dx=valid['dx'])
    table=pd.DataFrame({
        'image_id':valid['image_id'],'lesion_id':valid['lesion_id'],
        'true_dx':valid['dx'],'label':y_val,
        'predicted_dx':[CLASS_CODES[int(k)] for k in prediction],
        'predicted_label':prediction})
    for k,name in enumerate(CLASS_CODES): table[f'prob_{name}']=probabilities[:,k]
    table.to_csv(run_dir/'validation_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(run_dir/'training_history.csv',index=False)
    per=precision_recall_fscore_support(y_val,prediction,labels=np.arange(N_CLASSES),zero_division=0)
    pd.DataFrame({'class':CLASS_CODES,'support':per[3],
                  'precision':per[0],'recall':per[1],'f1':per[2]}).to_csv(
        run_dir/'validation_per_class.csv',index=False)
    cm=confusion_matrix(y_val,prediction,labels=np.arange(N_CLASSES))
    fig,ax=plt.subplots(figsize=(7.5,6.2))
    im=ax.imshow(cm)
    ax.set_xticks(range(N_CLASSES),CLASS_CODES);ax.set_yticks(range(N_CLASSES),CLASS_CODES)
    ax.set_xlabel('Predicted diagnosis');ax.set_ylabel('True diagnosis')
    ax.set_title(f'ANFIS {n_rules} rules, seed {seed} | validation')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.colorbar(im,ax=ax);fig.tight_layout()
    fig.savefig(run_dir/'validation_confusion_matrix.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    hist=pd.DataFrame(history)
    ax.plot(hist.epoch,hist.train_weighted_ce,label='Train weighted CE')
    ax.set_xlabel('Epoch');ax.set_ylabel('Weighted CE')
    ay=ax.twinx();ay.plot(hist.epoch,hist.val_macro_f1,label='Validation macro-F1')
    ay.set_ylabel('Validation macro-F1')
    ax.set_title(f'ANFIS {n_rules} rules | seed {seed}')
    fig.tight_layout();fig.savefig(run_dir/'training_curves.png',dpi=170);plt.close(fig)

    explanation_start=time.perf_counter()
    ablation=export_rule_analysis(model,run_dir,probabilities)
    rule_analysis_and_export_seconds = time.perf_counter() - explanation_start
    # Head-only inference: input already precomputed; excludes CNN feature extraction.
    model.eval();sample=torch.from_numpy(X_val).to(DEVICE)
    with torch.inference_mode():
        _=model(sample[:32])
        if DEVICE.type=='cuda':torch.cuda.synchronize()
        times=[]
        for _ in range(5):
            t0=time.perf_counter()
            _=model(sample)
            if DEVICE.type=='cuda':torch.cuda.synchronize()
            times.append(time.perf_counter()-t0)
    latency_ms_per_image=float(np.median(times)/len(X_val)*1000)
    record={
        'n_rules':n_rules,'seed':seed,'best_epoch':best_epoch,'epochs_completed':len(history),
        'train_seconds':round(training_seconds,3),
        'rule_analysis_and_export_seconds':round(rule_analysis_and_export_seconds,3),
        'head_inference_ms_per_image':latency_ms_per_image,
        'head_validation_throughput_imgs_s':len(X_val)/float(np.median(times)),
        'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'peak_training_gpu_memory_mb':peak_training_gpu_mb,
        'rule_ablation_max_abs_logit_change':ablation['max_abs_signed']['mean_abs_original_predicted_logit_change'],
        'rule_ablation_min_abs_logit_change':ablation['min_abs_signed']['mean_abs_original_predicted_logit_change'],
        **final
    }
    (run_dir/'validation_metrics.json').write_text(json.dumps(record,indent=2))
    print('Completed:',f'R={n_rules}',f'seed={seed}',
          'macro-F1=',round(record['macro_f1'],4),
          'bal. acc.=',round(record['balanced_accuracy'],4),
          'mel. recall=',round(record['melanoma_recall'],4))
    return record

results=[]
for n_rules in RULE_COUNTS:
    for seed in SEEDS:
        results.append(run_one(n_rules,seed))
results=pd.DataFrame(results)
results.to_csv(OUT/'e4_all_seed_metrics.csv',index=False)
print(f'PASS: all {len(RULE_COUNTS)*len(SEEDS)} validation-only ANFIS runs finished')


# %% 7. Selection, repeated-seed summary and E3-compatible reporting
summary_metrics=['accuracy','macro_f1','balanced_accuracy','melanoma_recall',
                 'melanoma_precision','macro_ovr_auroc','macro_ovr_ap',
                 'nll','train_seconds','head_inference_ms_per_image','trainable_params']
summary=results.groupby('n_rules')[summary_metrics].agg(['mean','std'])
summary.to_csv(OUT/'e4_seed_summary.csv')
selected=(results.sort_values(['n_rules','macro_f1','seed'],ascending=[True,False,True])
          .groupby('n_rules',sort=False).head(1))
selected.to_csv(OUT/'e4_validation_selected_by_rule_count.csv',index=False)
chosen=results.sort_values(['macro_f1','n_rules','seed'],ascending=[False,True,True]).iloc[0]
(OUT/'e4_validation_selected_overall.json').write_text(json.dumps({
    'n_rules':int(chosen['n_rules']),'seed':int(chosen['seed']),
    'macro_f1':float(chosen['macro_f1']),'best_epoch':int(chosen['best_epoch']),
    'rule_selection_policy':'Highest validation macro-F1; tie smaller R then seed',
    'warning':'This is validation-selected; not unbiased held-out-test performance.'},indent=2))
print('\nALL RUNS')
print(results[['n_rules','seed','best_epoch','macro_f1','accuracy','balanced_accuracy',
               'melanoma_recall','train_seconds','trainable_params']].to_string(index=False))
print('\nSEED SUMMARY: macro-F1 mean and sample SD')
print(summary['macro_f1'].to_string())
print('\nBEST PER RULE COUNT')
print(selected[['n_rules','seed','best_epoch','macro_f1','balanced_accuracy','melanoma_recall']].to_string(index=False))
print('Best overall validation-selected configuration:',
      f"R={int(chosen['n_rules'])}, seed={int(chosen['seed'])}, macro-F1={chosen['macro_f1']:.4f}")


# %% 8. Provenance manifest, integrity audit, and ZIP backup
settings={
    'experiment':'ACIT4620_E4_First_Order_Takagi_Sugeno_ANFIS',
    'source_proposal':'ACIT4620 Group 1 submitted MA1 proposal, first-order multi-output Sugeno',
    'feature_source':'Notebook 06 frozen EfficientNet-B0 E2a-50, pooled 1280D features',
    'source_checkpoint_sha256':feature_config['checkpoint_sha256'],
    'split_sha256':EXPECTED_SPLIT_SHA256,
    'class_codes':CLASS_CODES,'train_images':7116,'validation_images':1441,
    'test_images_processed':0,'test_set_locked':True,
    'train_only_standardization':True,
    'projection':'trainable linear 1280 -> 8, no activation',
    'antecedents':'R trainable Gaussian centers/positive widths per projected dimension',
    'rule_combination':'product firing, computed as summed Gaussian log memberships',
    'rule_normalization':'stable Softmax of log firing strengths',
    'consequents':'each rule has 7 first-order affine output functions',
    'final_output':'weighted sum of affine consequents = 7 logits, then Softmax',
    'rule_counts':RULE_COUNTS,'seeds':SEEDS,
    'premise_initialization':'TRAIN-only KMeans on initial linearly projected features',
    'optimizer':'Adam joint gradient descent/backprop (NOT classical ANFIS LSE + GD)',
    'learning_rate':LEARNING_RATE,'weight_decay':WEIGHT_DECAY,
    'batch_size':BATCH_SIZE,'max_epochs':MAX_EPOCHS,
    'patience':PATIENCE,'gradient_clip_norm':MAX_GRAD_NORM,
    'loss':'train-frequency inverse weighted multiclass cross-entropy',
    'model_selection':'validation macro-F1; secondary metrics descriptive only',
    'explanations':'normalized rule strengths and signed per-class contributions',
    'ablation':'remove high/low absolute-contribution rule for original predicted class, '
               'renormalize remaining weights; validation-only faithfulness proxy',
    'efficiency':'head-only inference latency, parameters, training time, separate rule-analysis/export time; CNN excluded',
    'limitations':'Latent fuzzy rules lack direct diagnostic semantics; '
                  'no image-level Grad-CAM or occlusion in this feature-only notebook. '
                  'Validation used for checkpoint and model selection; test locked.',
}
(OUT/'e4_experiment_config.json').write_text(json.dumps(settings,indent=2))
required=[
    'train_feature_scaler.npz','e4_all_seed_metrics.csv','e4_seed_summary.csv',
    'e4_validation_selected_by_rule_count.csv','e4_validation_selected_overall.json',
    'e4_experiment_config.json'
]
for name in required:
    assert (OUT/name).is_file() and (OUT/name).stat().st_size>0, name
for n_rules in RULE_COUNTS:
    for seed in SEEDS:
        folder=OUT/f'anfis_r{n_rules}_seed{seed}'
        for name in [
            'best_anfis_head.pth','validation_predictions.npz','validation_predictions.csv',
            'validation_metrics.json','validation_per_class.csv',
            'training_history.csv','training_curves.png','validation_confusion_matrix.png',
            'validation_rule_explanations.npz','validation_rule_contributions.csv',
            'rule_activation_by_class.csv','learned_gaussian_premises.csv',
            'rule_ablation_faithfulness.json','validation_rule_ablation.csv',
            'rule_activation_by_class.png','premise_initialization.json'
        ]:
            assert (folder/name).is_file() and (folder/name).stat().st_size>0,(folder,name)
archive=shutil.make_archive(str(OUT.with_name(OUT.name + '_Outputs')),
                            'zip',root_dir=OUT.parent,base_dir=OUT.name)
print(f'PASS: {len(RULE_COUNTS)*len(SEEDS)} checkpoints, metrics, validation predictions, rule contributions and ablations saved')
print('ZIP backup:',archive,'size MB:',round(Path(archive).stat().st_size/1048576,2))
print('IMPORTANT: Original HAM10000 held-out test remains locked and untouched.')
print('Next: compare E3 and E4 on the same validation images, then image-level Grad-CAM and occlusion separately.')

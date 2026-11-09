"""ACIT4620 Group 1 – Notebook 11 / RQ4.

Matched COMPUTATIONAL benchmarks for frozen EfficientNet-B0 + conventional heads
and first-order Takagi–Sugeno ANFIS. Uses ORIGINAL, saved checkpoints; does not
train a model for predictive purposes, tune to test data, or open held-out test.

Benchmark modes:
* head-only: fixed synthetic shape-matched 1280-D tensors; no CNN or image I/O.
* tensor-to-logit: same preprocessed real validation images if available; otherwise
  synthetic 3x224x224 tensors, explicitly labelled in exported results.
* image-file-to-logit: optional genuine validation .jpg -> preprocessing -> CNN -> head.
* training-step: matched synthetic train-step MICROBENCHMARK, not actual training time.
* historical training: recorded duration from earlier training runs; NOT controlled.
"""
from __future__ import annotations
import os, time, json, math, hashlib, copy, platform, shutil, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anfis_core import MultiOutputSugenoANFIS

SPLIT_SHA = '265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0'
CNN_SHA = '2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9'
CLASSES = ['nv','mel','bkl','bcc','akiec','vasc','df']
SELECTED = ['cnn_e2a50','linear_seed42','mlp_seed42','anfis_r4_seed52','anfis_r8_seed42','anfis_r16_seed52']
RUNS = ['linear_seed42','linear_seed52','linear_seed62','mlp_seed42','mlp_seed52','mlp_seed62'] + [f'anfis_r{r}_seed{s}' for r in (4,8,16) for s in (42,52,62)]
TRANSFORM = transforms.Compose([
    transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class LinearHead(nn.Module):
    def __init__(self):
        super().__init__();self.classifier=nn.Linear(1280,7)
    def forward(self,x):return self.classifier(x)

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(1280,8)
        self.net=nn.Sequential(nn.ReLU(),nn.Linear(8,32),nn.ReLU(),nn.Dropout(0.15),nn.Linear(32,7))
    def forward(self,x):return self.net(self.projection(x))


def sha256(path: Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_pt(path: Path):
    return torch.load(path,map_location='cpu',weights_only=True)


def repo_candidates(base: Path):
    environment=os.environ.get('ACIT4620_REPO_ROOT','').strip()
    items=[Path(environment)] if environment else []
    items.extend([
        Path(r'C:\AI\G1-NB10'),
        Path.home()/'Projects'/'ACIT4620_Group1_Notebook09',
        Path.home()/'Projects'/'ACIT4620_Group1_E3_Full',
        Path.home()/'Projects'/'ACIT4620_Group1_E4',
        base.parent/'ACIT4620-Group1-Hybrid-AI',
        base.parent.parent/'ACIT4620-Group1-Hybrid-AI'
    ])
    return items


def find_repo(base: Path, override: str|Path|None=None):
    if override is not None:
        choices=[Path(override).expanduser()]
    else: choices=repo_candidates(base)
    for p in choices:
        target=p/'experiments'/'e3_linear_mlp'/'outputs'/'ACIT4620_E3_Linear_MLP'
        if (target/'linear_seed42'/'best_head.pth').is_file() and (target/'mlp_seed42'/'best_head.pth').is_file():
            return p.resolve()
    raise FileNotFoundError(
        'Cannot locate the SIX original E3 checkpoints. Your latest GitHub checkout must '
        'contain experiments/e3_linear_mlp/outputs/ACIT4620_E3_Linear_MLP/linear_seed42/best_head.pth '
        'and mlp_seed42/best_head.pth. Set TEAM_REPO in Notebook 11 to that checkout; '
        'do NOT fabricate or retrain weights. Checked:\n'+'\n'.join(map(str,choices)))


def check_ckpt(ckpt, split=True):
    if split: assert ckpt['split_sha256']==SPLIT_SHA, 'Split does not match proposal'
    assert ckpt['class_codes']==CLASSES, 'Class order mismatch'

@dataclass
class BenchmarkSession:
    root:Path
    repo:Path
    out:Path
    device:torch.device
    cnn:nn.Module
    heads:dict[str,nn.Module]
    scale_mean:torch.Tensor
    scale_std:torch.Tensor
    checksum:dict[str,str]


def load_models(project_root:Path|str, team_repo:Path|str|None=None, device='cpu', cpu_threads=4):
    root=Path(project_root).resolve();repo=find_repo(root,team_repo)
    assets=root/'data'/'model_assets'; out=root/'outputs'/'ACIT4620_11_Efficiency_Results';out.mkdir(parents=True,exist_ok=True)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but not available')
    if device=='cpu':torch.set_num_threads(max(1,min(int(cpu_threads),os.cpu_count() or 1)))
    device=torch.device(device)
    cnn_path=assets/'e2a50_best_efficientnet_b0.pth'
    assert cnn_path.is_file() and sha256(cnn_path)==CNN_SHA, 'Wrong E2a-50 checkpoint or checksum'
    cc=load_pt(cnn_path);check_ckpt(cc)
    cnn=efficientnet_b0(weights=None)
    assert cnn.classifier[1].in_features==1280
    cnn.classifier[1]=nn.Linear(1280,7)
    cnn.load_state_dict(cc['model_state'],strict=True)
    cnn.to(device).eval().requires_grad_(False)
    scaler=assets/'train_feature_scaler.npz'
    with np.load(scaler,allow_pickle=False) as data:
        mean=data['mean'];std=data['std']
    assert mean.shape==std.shape==(1280,) and (std>0).all() and np.isfinite(mean).all()
    mean_t=torch.tensor(mean,dtype=torch.float32,device=device)[None]
    std_t=torch.tensor(std,dtype=torch.float32,device=device)[None]
    heads={};checks={'cnn_e2a50':CNN_SHA,'train_only_scaler':sha256(scaler)}
    e3_base=repo/'experiments'/'e3_linear_mlp'/'outputs'/'ACIT4620_E3_Linear_MLP'
    for name in RUNS:
        if name.startswith(('linear','mlp')):
            path=e3_base/name/'best_head.pth'
            if not path.is_file():raise FileNotFoundError(f'Missing original E3 checkpoint: {path}')
            ck=load_pt(path);check_ckpt(ck)
            assert ck['model_name']==name.split('_')[0] and ck['seed']==int(name.split('seed')[1])
            assert ck['feature_checkpoint_sha256']==CNN_SHA
            m=LinearHead() if name.startswith('linear') else MLPHead()
        else:
            path=assets/f'{name}.pth';ck=load_pt(path);check_ckpt(ck)
            rules=int(name.split('_')[1][1:]);seed=int(name.split('seed')[1]);assert ck['n_rules']==rules and ck['seed']==seed
            assert ck['source_checkpoint_sha256']==CNN_SHA
            m=MultiOutputSugenoANFIS(1280,8,rules,7)
        m.load_state_dict(ck['state_dict'],strict=True)
        heads[name]=m.to(device).eval().requires_grad_(False)
        checks[name]=sha256(path)
    # Check the E3 scaler was generated using the same train-only feature arrays.
    e3_scaler=e3_base/'train_feature_scaler.npz'
    if not e3_scaler.is_file():raise FileNotFoundError(f'Missing E3 training-only feature scaler: {e3_scaler}')
    with np.load(e3_scaler,allow_pickle=False) as z:
        assert np.allclose(z['mean'],mean,atol=1e-5,rtol=1e-5), 'E3 scaler mean differs'
        assert np.allclose(z['std'],std,atol=1e-5,rtol=1e-5), 'E3 scaler std differs'
    print('PASS: original CNN, 6 E3 and 9 E4 checkpoints loaded; split, CNN SHA, scaler verified.')
    print('GitHub checkout:',repo,'| device:',device)
    print('Locked test images/labels were NOT accessed.')
    return BenchmarkSession(root,repo,out,device,cnn,heads,mean_t,std_t,checks)


class FullPipeline(nn.Module):
    def __init__(self, session:BenchmarkSession, name:str):
        super().__init__();self.cnn=session.cnn;self.name=name
        if name!='cnn_e2a50':
            self.head=session.heads[name]
            self.register_buffer('feature_mean',session.scale_mean)
            self.register_buffer('feature_std',session.scale_std)
    def forward(self,x):
        if self.name=='cnn_e2a50':return self.cnn(x)
        feats=torch.flatten(self.cnn.avgpool(self.cnn.features(x)),1)
        feats=(feats-self.feature_mean)/self.feature_std
        return self.head(feats)


def get_head_fn(s:BenchmarkSession,name:str):
    if name=='cnn_e2a50':return s.cnn.classifier
    return s.heads[name]


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def model_stats(s):
    backbone=sum(p.numel() for p in s.cnn.features.parameters())
    # EfficientNet also contains no trainable parameters in avgpool.
    rows=[]
    for name in ['cnn_e2a50']+RUNS:
        m=get_head_fn(s,name)
        head_count=sum(p.numel() for p in m.parameters())
        extra=0 if name=='cnn_e2a50' else head_count
        rows.append({'model':name,'backbone_parameters':backbone,'head_parameters':head_count,
                     'additional_parameters_vs_original_cnn':0 if name=='cnn_e2a50' else head_count - sum(p.numel() for p in s.cnn.classifier.parameters()),
                     'effective_deployed_parameters':backbone+head_count,
                     'head_parameter_memory_mb_f32':head_count*4/1048576,
                     'checkpoint_size_mb':None if name=='cnn_e2a50' else None})
    df=pd.DataFrame(rows)
    ckpts=[]
    assets=s.root/'data'/'model_assets'
    for name in df.model:
        if name=='cnn_e2a50':p=assets/'e2a50_best_efficientnet_b0.pth'
        elif name.startswith(('mlp','linear')):p=s.repo/'experiments'/'e3_linear_mlp'/'outputs'/'ACIT4620_E3_Linear_MLP'/name/'best_head.pth'
        else:p=assets/f'{name}.pth'
        ckpts.append(round(p.stat().st_size/1048576,5))
    df['checkpoint_size_mb']=ckpts
    df.to_csv(s.out/'parameter_and_checkpoint_cost.csv',index=False)
    return df


def timed_inference(fn,x,device,repeats=12,warmups=3,preprocessor=None,seed=42):
    assert repeats>=3 and warmups>=1
    torch.manual_seed(seed)
    durations=[]
    for i in range(warmups+repeats):
        synchronize(device)
        start=time.perf_counter()
        with torch.inference_mode():
            inp=x if preprocessor is None else preprocessor()
            out=fn(inp)
            # Materialize one scalar on GPU; avoid asynchronous timing.
            if device.type=='cuda':_=out[0,0].item()
        synchronize(device)
        duration=(time.perf_counter()-start)*1000
        assert out.ndim==2 and out.shape[1]==7 and torch.isfinite(out).all()
        if i>=warmups:durations.append(duration)
    a=np.array(durations)
    return {'repeats':repeats,'warmup_runs':warmups,'mean_batch_ms':float(a.mean()),
            'median_batch_ms':float(np.median(a)),'p95_batch_ms':float(np.percentile(a,95)),
            'sd_batch_ms':float(a.std(ddof=1)),
            'mean_ms_per_image':float(a.mean()/len(x)),
            'throughput_images_per_second':float(len(x)*1000/a.mean())}


def gpu_memory(session, fn, x):
    if session.device.type!='cuda':return float('nan')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(session.device)
    before=torch.cuda.memory_allocated(session.device)
    synchronize(session.device)
    with torch.inference_mode(): z=fn(x)
    synchronize(session.device)
    return float((torch.cuda.max_memory_allocated(session.device)-before)/1048576)


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss/1048576
    except ImportError:return float('nan')


def benchmark_head_only(s, batch_sizes=(1,32,128), repeats=16, warmups=4, all_seeds=True):
    rows=[];names=['cnn_e2a50']+ (RUNS if all_seeds else [n for n in SELECTED if n!='cnn_e2a50'])
    for batch in batch_sizes:
        gen=torch.Generator().manual_seed(20261109+batch)
        raw=torch.randn(batch,1280,generator=gen,device='cpu').to(s.device)
        for name in names:
            inp=raw if name=='cnn_e2a50' else raw # standard scores are also 1280D; shape-only microbench
            model=get_head_fn(s,name).eval()
            rss_before=_rss_mb()
            info=timed_inference(model,inp,s.device,repeats,warmups)
            rows.append({'model':name,'batch_size':batch,'input_type':'synthetic_shape_matched_1280D',
                         'device':str(s.device),'cpu_threads':torch.get_num_threads(),
                         'stage':'head_only_forward','cpu_rss_after_minus_before_mb':_rss_mb()-rss_before,
                         'peak_gpu_incremental_allocated_mb':gpu_memory(s,model,inp),**info})
    df=pd.DataFrame(rows);df.to_csv(s.out/'head_only_inference.csv',index=False)
    return df


def raw_root_candidates(s, override=None):
    if override is not None:return [Path(override).expanduser()]
    env=os.environ.get('ACIT4620_RAW_IMAGES_ROOT','').strip()
    roots=[Path(env)] if env else []
    roots.extend([s.root/'data'/'raw',
        Path.home()/'OneDrive'/'Desktop'/'OSLOMET'/'COMPUTATIONAL INTELLIGENCE'/'FOR FINAL REPORT'/'CODE'/'Notebook_10_CNN_ANFIS_XAI_VSCode_Complete'/'ACIT4620_10_CNN_ANFIS_XAI_VSCode'/'data'/'raw',
        Path(r'C:\AI\NB10')])
    return roots


def find_real_images(s, override=None, max_images=16):
    manifest=pd.read_csv(s.root/'data'/'reference'/'sample_validation_images.csv')
    assert len(manifest)>=16 and manifest.image_id.is_unique
    wanted=manifest.image_id.astype(str).tolist()[:max_images]
    for root in raw_root_candidates(s,override):
        if not root.exists():continue
        paths={}
        for p in root.rglob('*.jpg'):
            if p.stem in wanted:
                if p.stem in paths:raise RuntimeError(f'Duplicate image file: {p.stem}')
                paths[p.stem]=p
        if set(paths)==set(wanted):
            return [(im,paths[im]) for im in wanted]
    raise FileNotFoundError('Could not locate original JPEG files for validation-only IDs. '
        'Place HAM10000_images_part_1 and HAM10000_images_part_2 under data/raw/, '
        'or set RAW_IMAGES_ROOT in the notebook to Notebook 10 data/raw. '
        'Never use the locked test set as a substitute.')


def prepare_real_images(s, raw_override=None, max_images=16):
    selected=find_real_images(s,raw_override,max_images=max_images)
    xs=[]
    for _,p in selected:
        with Image.open(p) as im:xs.append(TRANSFORM(im.convert('RGB')))
    return torch.stack(xs,dim=0).to(s.device),selected


def benchmark_tensor_to_logit(s, images, mode='real_validation', batch_sizes=(1,8), repeats=8,warmups=2):
    assert images.ndim==4 and tuple(images.shape[1:])==(3,224,224)
    models={name:FullPipeline(s,name).eval() for name in SELECTED}
    rows=[]
    for b in batch_sizes:
        assert b<=len(images)
        x=images[:b]
        for name,model in models.items():
            rss_before=_rss_mb()
            info=timed_inference(model,x,s.device,repeats,warmups)
            rows.append({'model':name,'batch_size':b,'input_type':mode,'stage':'preprocessed_tensor_to_class_logits',
                         'cpu_threads':torch.get_num_threads(),'device':str(s.device),
                         'cpu_rss_after_minus_before_mb':_rss_mb()-rss_before,
                         'peak_gpu_incremental_allocated_mb':gpu_memory(s,model,x),**info})
    df=pd.DataFrame(rows);df.to_csv(s.out/'end_to_end_tensor_to_logit.csv',index=False)
    return df


def benchmark_image_file_pipeline(s, selected, batch_sizes=(1,8),repeats=6,warmups=2):
    if not selected:raise ValueError('Real validation images required')
    models={name:FullPipeline(s,name).eval() for name in SELECTED}
    rows=[]
    for b in batch_sizes:
        assert b<=len(selected)
        batch=selected[:b]
        def preprocessing():
            tensors=[]
            for _,p in batch:
                with Image.open(p) as im:tensors.append(TRANSFORM(im.convert('RGB')))
            return torch.stack(tensors).to(s.device)
        # Use a correctly shaped placeholder ONLY for batch_size metadata.
        placeholder=torch.empty((b,3,224,224),device=s.device)
        for name,model in models.items():
            info=timed_inference(model,placeholder,s.device,repeats,warmups,preprocessor=preprocessing)
            rows.append({'model':name,'batch_size':b,'input_type':'real_validation_jpg',
                         'stage':'file_decode_preprocessing_and_inference','device':str(s.device),**info})
    df=pd.DataFrame(rows);df.to_csv(s.out/'end_to_end_jpg_to_logit.csv',index=False)
    return df


def training_step_microbenchmark(s,batch_size=128,steps=8,warmups=2):
    """Matched train-step cost proxy. Does NOT update original checkpoint models."""
    gen=torch.Generator().manual_seed(1984)
    xx=torch.randn(batch_size,1280,generator=gen).to(s.device)
    yy=torch.randint(0,7,(batch_size,),generator=gen).to(s.device)
    weights=torch.tensor([.211962,1.299963,1.316802,2.839585,4.419876,11.42215,11.42215],device=s.device)
    names=['linear_seed42','mlp_seed42','anfis_r4_seed52','anfis_r8_seed42','anfis_r16_seed52']
    records=[]
    for name in names:
        m=copy.deepcopy(s.heads[name]).to(s.device).train()
        for p in m.parameters():p.requires_grad_(True)
        optimizer=torch.optim.Adam(m.parameters(),lr=1e-3) if name.startswith('anfis') else torch.optim.AdamW(m.parameters(),lr=1e-3)
        durations=[]
        for step in range(warmups+steps):
            synchronize(s.device);t0=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(m(xx),yy,weight=weights)
            loss.backward();optimizer.step()
            synchronize(s.device)
            dt=(time.perf_counter()-t0)*1000
            if step>=warmups:durations.append(dt)
        records.append({'model':name,'batch_size':batch_size,'mean_training_step_ms':float(np.mean(durations)),
                        'p95_training_step_ms':float(np.percentile(durations,95)),
                        'train_step_samples_per_second':batch_size*1000/float(np.mean(durations)),
                        'benchmark_steps':steps,'data_type':'synthetic_shape_matched_features',
                        'training_purpose':'microbenchmark_only_no_predictive_training',
                        'device':str(s.device)})
    df=pd.DataFrame(records);df.to_csv(s.out/'training_step_microbenchmark.csv',index=False)
    return df


def historical_training(s):
    base=s.root/'data'/'reference'
    e3=pd.read_csv(base/'e3_all_seed_metrics.csv')
    e4=pd.read_csv(base/'e4_all_seed_metrics.csv')
    a=e3.assign(family=lambda x:x['head'],model=lambda x:x['head']+'_seed'+x.seed.astype(str),source='E3_recorded_local')
    b=e4.assign(family=lambda x:'anfis_r'+x.n_rules.astype(str),model=lambda x:'anfis_r'+x.n_rules.astype(str)+'_seed'+x.seed.astype(str),source='E4_recorded_local')
    cols=['model','family','seed','epochs_completed','best_epoch','train_seconds','trainable_params','source']
    all_df=pd.concat([a[cols],b[cols]],ignore_index=True)
    all_df.to_csv(s.out/'historical_training_times_NOT_matched.csv',index=False)
    return all_df


def graph(s,head,full,historical,params,trainproxy):
    folder=s.out/'figures';folder.mkdir(parents=True,exist_ok=True)
    def bars(df,column,title,path):
        if df.empty:return
        fig,ax=plt.subplots(figsize=(10,5))
        ax.bar(df.model,df[column])
        ax.set_ylabel(title);ax.set_xticks(range(len(df)));ax.set_xticklabels(df.model,rotation=30,ha='right')
        ax.set_title(title);fig.tight_layout();fig.savefig(folder/path,dpi=185);plt.close(fig)
    bars(params[params.model.isin(SELECTED)],'head_parameters','Trainable head parameter count','01_head_parameters.png')
    if not head.empty:
        subset=head[(head.batch_size==1)&head.model.isin(SELECTED)]
        bars(subset,'median_batch_ms','Head-only latency, batch=1 (ms)','02_head_only_latency.png')
    if not full.empty:
        subset=full[(full.batch_size==1)&full.model.isin(SELECTED)]
        bars(subset,'median_batch_ms','CNN + head tensor-to-logit latency, batch=1 (ms)','03_full_model_latency.png')
    if not trainproxy.empty:bars(trainproxy,'mean_training_step_ms','Training-step microbenchmark, 128 features (ms)','04_train_step_cost.png')
    hist=historical.groupby('family',as_index=False).agg(train_seconds_mean=('train_seconds','mean'),train_seconds_std=('train_seconds','std'))
    fig,ax=plt.subplots(figsize=(9,5));ax.bar(hist.family,hist.train_seconds_mean,yerr=hist.train_seconds_std,capsize=4)
    ax.set_ylabel('Recorded training seconds (different original environments)')
    ax.set_title('Historical times: descriptive only, NOT matched hardware')
    fig.tight_layout();fig.savefig(folder/'05_historical_training_times_caution.png',dpi=185);plt.close(fig)


def finalize(s, mode, head, full, hist, params, micro, raw=None):
    graph(s,head,full,hist,params,micro)
    env={'cpu':platform.processor(),'logical_cpu_count':os.cpu_count(),'cpu_threads':torch.get_num_threads(),
         'device':str(s.device),'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
         'numpy':np.__version__,'python':platform.python_version(),'os':platform.platform(),
         'image_mode':mode,'validation_only':True,'test_set_accessed':False,
         'timing_note':'Warmup then repeated wall-clock measurement. Results are local-device, batch-size dependent.',
         'memory_note':'GPU incremental peak allocated memory recorded if CUDA. CPU RSS before/after is NOT peak memory.',
         'historical_training_warning':'E3/E4 original recorded seconds are not matched-hardware controlled comparisons.',
         'data_note':'Head-only and training-step inputs are synthetic 1280-D tensors, for cost only; no predictive evaluation.',
         'checkpoints_sha256':s.checksum,'split_sha256':SPLIT_SHA,'cnn_sha256':CNN_SHA,'selected':SELECTED}
    (s.out/'benchmark_provenance.json').write_text(json.dumps(env,indent=2,default=str),encoding='utf-8')
    root=s.out.parent
    dest=shutil.make_archive(str(root/'ACIT4620_11_Efficiency_Results'),'zip',root_dir=root,base_dir=s.out.name)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print('PASS: RQ4 outputs saved:',s.out)
    print('Backup ZIP:',dest,'size MB:',round(Path(dest).stat().st_size/1048576,2))
    print('TEST SPLIT REMAINS LOCKED. These are local cost measurements, not test results.')
    return Path(dest)

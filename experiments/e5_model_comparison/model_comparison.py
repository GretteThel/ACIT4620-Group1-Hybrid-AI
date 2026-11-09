"""ACIT4620 Notebook 09 — reproducible, validation-only paired model comparison.

Reads only versioned 1,441-row validation prediction exports, not CNN images
or held-out test labels. The original prediction CSVs are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
    roc_auc_score,
)

CLASSES = ['nv','mel','bkl','bcc','akiec','vasc','df']
CLASS_IDX = {x:i for i,x in enumerate(CLASSES)}
SPLIT_HASH = '265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0'
SOURCE_CKPT_HASH = '2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9'
N_VALID = 1441
N_LESIONS = 1067
MEL = CLASS_IDX['mel']


def sha256_file(path:Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def locate_root(start:Path|str|None=None)->Path:
    candidate=Path(start or Path.cwd()).resolve()
    for p in (candidate, *candidate.parents):
        if (p/'inputs'/'input_provenance.json').exists():
            return p
    raise FileNotFoundError('Open the ACIT4620_09_Model_Comparison_VSCode folder in VS Code. Missing inputs/input_provenance.json.')


def verify_sources(root:Path)->dict:
    info=json.loads((root/'inputs'/'input_provenance.json').read_text(encoding='utf-8'))
    assert info['split_sha256']==SPLIT_HASH
    assert info['class_codes']==CLASSES
    assert info['held_out_test_processed'] is False
    for row in info['files']:
        f=root/'inputs'/row['path']
        if not f.is_file() or sha256_file(f)!=row['sha256']:
            raise RuntimeError(f'Integrity check failed: {f}')
    for key in ['cnn','e3','e4']:
        p=root/'inputs'/key/'experiment_config.json'
        cfg=json.loads(p.read_text(encoding='utf-8'))
        assert cfg['split_sha256']==SPLIT_HASH, key
        assert cfg['class_codes']==CLASSES, key
        if key in ['e3','e4']:
            assert cfg['source_checkpoint_sha256']==SOURCE_CKPT_HASH, key
            assert cfg['test_images_processed']==0, key
        if key=='cnn':
            assert cfg.get('experiment')=='efficientnet_b0_classweighted_e2a_50epochs'
    return info


def list_prediction_paths(root:Path)->dict[str,Path]:
    pairs={'cnn_e2a50':root/'inputs'/'cnn'/'e2a50.csv'}
    for head in ['linear','mlp']:
        for s in [42,52,62]:
            pairs[f'{head}_seed{s}']=root/'inputs'/'e3'/f'{head}_seed{s}.csv'
    for rules in [4,8,16]:
        for s in [42,52,62]:
            pairs[f'anfis_r{rules}_seed{s}']=root/'inputs'/'e4'/f'anfis_r{rules}_seed{s}.csv'
    return pairs


def load_predictions(root:Path)->tuple[dict, pd.DataFrame,dict]:
    datasets={}
    reference=None
    info={}
    for name,path in list_prediction_paths(root).items():
        d=pd.read_csv(path)
        need=['image_id','lesion_id','true_dx','predicted_dx']+[f'prob_{x}' for x in CLASSES]
        assert set(need)<=set(d.columns), (name,set(need)-set(d.columns))
        assert len(d)==N_VALID and d['image_id'].is_unique, name
        d=d.sort_values('image_id').reset_index(drop=True)
        assert d['image_id'].notna().all() and d['lesion_id'].notna().all()
        assert d['true_dx'].isin(CLASSES).all() and d['predicted_dx'].isin(CLASSES).all()
        y=d['true_dx'].map(CLASS_IDX).to_numpy(dtype=int)
        probs=d[[f'prob_{c}' for c in CLASSES]].to_numpy(dtype=np.float64)
        assert np.isfinite(probs).all() and (probs>=-1e-10).all()
        assert (np.abs(probs.sum(axis=1)-1)<2e-3).all(), name
        assert (probs>=0).all(),name
        p=probs/probs.sum(axis=1,keepdims=True)
        pred=p.argmax(axis=1)
        assert np.array_equal(pred,d['predicted_dx'].map(CLASS_IDX).to_numpy()),(name,'predicted_dx mismatch')
        if 'label' in d.columns:assert np.array_equal(y,d['label'].to_numpy(dtype=int)), name
        if 'predicted_label' in d.columns:assert np.array_equal(pred,d['predicted_label'].to_numpy(dtype=int)), name
        ref=d[['image_id','lesion_id','true_dx']]
        if reference is None:
            reference=ref
        else:
            assert reference.equals(ref), f'Image/lesion/diagnosis alignment mismatch: {name}'
        datasets[name]={'y':y,'pred':pred,'prob':p}
        info[name]={'row_count':len(d),'prob_sum_abs_max_before_renorm':float(np.max(np.abs(probs.sum(axis=1)-1)))}
    assert reference is not None and reference['lesion_id'].nunique()==N_LESIONS
    assert (reference.groupby('lesion_id').true_dx.nunique()<=1).all()
    assert reference['true_dx'].value_counts().reindex(CLASSES).fillna(0).astype(int).tolist()==[974,146,168,89,34,21,9]
    return datasets,reference,info


def performance(y,pred,probs)->dict:
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=np.arange(7),zero_division=0)
    onehot=np.eye(7)[y]
    probs=np.clip(probs,1e-15,1)
    probs=probs/probs.sum(axis=1,keepdims=True)
    return {
        'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=np.arange(7),average='macro',zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
        'melanoma_precision':float(precision[MEL]),
        'melanoma_recall':float(recall[MEL]),
        'melanoma_f1':float(f1[MEL]),
        'melanoma_tp':int(((y==MEL)&(pred==MEL)).sum()),
        'melanoma_fn':int(((y==MEL)&(pred!=MEL)).sum()),
        'nll_raw_original_view':float(log_loss(y,probs,labels=list(range(7)))),
        'multiclass_brier':float(np.mean(np.sum((probs-onehot)**2,axis=1))),
        'macro_ovr_auroc':float(roc_auc_score(y,probs,labels=list(range(7)),multi_class='ovr',average='macro')),
        'macro_ovr_ap':float(np.mean([average_precision_score(onehot[:,c],probs[:,c]) for c in range(7)])),
    }


def metrics_and_classwise(datasets:dict)->tuple[pd.DataFrame,pd.DataFrame]:
    rows=[];per_class=[]
    for name,data in datasets.items():
        y,pred,probs=data['y'],data['pred'],data['prob']
        s=performance(y,pred,probs);rows.append({'model':name,**s})
        prec,rec,f1,support=precision_recall_fscore_support(y,pred,labels=range(7),zero_division=0)
        for j,c in enumerate(CLASSES):per_class.append({'model':name,'class':c,'support':int(support[j]),'precision':float(prec[j]),'recall':float(rec[j]),'f1':float(f1[j])})
    return pd.DataFrame(rows),pd.DataFrame(per_class)


def verify_reported_metrics(root:Path,computed:pd.DataFrame)->None:
    e3=pd.read_csv(root/'inputs'/'e3'/'e3_all_seed_metrics.csv')
    e4=pd.read_csv(root/'inputs'/'e4'/'e4_all_seed_metrics.csv')
    as_dict=computed.set_index('model')
    for _,r in e3.iterrows():
        name=f'{r["head"]}_seed{int(r["seed"])}'
        for m in ['accuracy','macro_f1','balanced_accuracy','melanoma_recall']:
            assert abs(float(r[m])-float(as_dict.loc[name,m]))<3e-6,(name,m)
    for _,r in e4.iterrows():
        name=f'anfis_r{int(r["n_rules"])}_seed{int(r["seed"])}'
        for m in ['accuracy','macro_f1','balanced_accuracy','melanoma_recall']:
            assert abs(float(r[m])-float(as_dict.loc[name,m]))<3e-6,(name,m)


def seed_table(metrics:pd.DataFrame)->pd.DataFrame:
    keep=metrics[metrics.model!='cnn_e2a50'].copy()
    keep['family']=keep.model.str.replace(r'_seed\d+$','',regex=True)
    return keep.groupby('family',sort=False).agg(
        n_seeds=('macro_f1','size'),
        macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
        accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),
        melanoma_recall_mean=('melanoma_recall','mean'),melanoma_recall_sd=('melanoma_recall','std')
    ).reset_index()


def selected_checkpoints(metrics:pd.DataFrame)->pd.DataFrame:
    values=[]
    for prefix in ['cnn_e2a50','linear_seed','mlp_seed','anfis_r4_seed','anfis_r8_seed','anfis_r16_seed']:
        subs=metrics[metrics.model.str.startswith(prefix)]
        assert len(subs) in (1,3)
        best=subs.sort_values(['macro_f1','model'],ascending=[False,True]).iloc[0].copy()
        best['family']=prefix.replace('_seed','')
        values.append(best)
    return pd.DataFrame(values).reset_index(drop=True)


def paired_error_table(datasets:dict,selected:pd.DataFrame,ref:pd.DataFrame)->pd.DataFrame:
    chosen=selected.model.to_list();y=datasets[chosen[0]]['y']; mel=(y==MEL)
    rows=[]
    for a in chosen:
        for b in chosen:
            if a>=b:continue
            pa=datasets[a]['pred'];pb=datasets[b]['pred']
            aok=pa==y; bok=pb==y
            rows.append({'model_a':a,'model_b':b,'disagreement_images':int((pa!=pb).sum()),
                         'both_correct':int((aok&bok).sum()),'both_wrong':int((~aok&~bok).sum()),
                         'a_only_correct':int((aok&~bok).sum()),'b_only_correct':int((~aok&bok).sum()),
                         'a_melanoma_tp':int((mel&(pa==MEL)).sum()),
                         'b_melanoma_tp':int((mel&(pb==MEL)).sum()),
                         'melanoma_corrected_b_vs_a':int((mel&(pa!=MEL)&(pb==MEL)).sum()),
                         'melanoma_lost_b_vs_a':int((mel&(pa==MEL)&(pb!=MEL)).sum())})
    return pd.DataFrame(rows)


def lesion_groups_by_class(ref:pd.DataFrame)->list[list[np.ndarray]]:
    by_class=[]
    for c in CLASSES:
        sub=ref[ref.true_dx==c]
        inds=[grp.index.to_numpy(dtype=int) for _,grp in sub.groupby('lesion_id',sort=True)]
        assert inds
        by_class.append(inds)
    return by_class


def paired_cluster_bootstrap(datasets:dict,ref:pd.DataFrame,pairs:list[tuple[str,str]],n_boot=1200,seed=20261109)->pd.DataFrame:
    """Resample lesions WITH replacement within true classes, carrying all images per sampled lesion.

    Conditional on trained models and reused validation data; not a test inference or seed-level CI.
    """
    rng=np.random.default_rng(seed)
    groups=lesion_groups_by_class(ref)
    rows=[]
    for a,b in pairs:
        aa=datasets[a];bb=datasets[b];y=aa['y']
        assert np.array_equal(y,bb['y'])
        observed=float(f1_score(y,bb['pred'],labels=range(7),average='macro',zero_division=0)-f1_score(y,aa['pred'],labels=range(7),average='macro',zero_division=0))
        bootstrap=np.empty(n_boot,dtype=float)
        for i in range(n_boot):
            sampled=[]
            for gl in groups:
                for idx in rng.integers(len(gl),size=len(gl)):
                    sampled.append(gl[int(idx)])
            ii=np.concatenate(sampled)
            bootstrap[i]=f1_score(y[ii],bb['pred'][ii],labels=range(7),average='macro',zero_division=0)-f1_score(y[ii],aa['pred'][ii],labels=range(7),average='macro',zero_division=0)
        lo,hi=np.percentile(bootstrap,[2.5,97.5]);rows.append({'model_a':a,'model_b':b,'delta_b_minus_a_macro_f1':observed,
                'ci95_low':float(lo),'ci95_high':float(hi),'bootstrap_resamples':n_boot,'lesions':N_LESIONS,
                'interval_includes_zero':bool(lo<=0<=hi),'interpretation':'exploratory paired class-stratified lesion bootstrap on validation-selected fitted predictions; no seed/model-selection uncertainty'})
    return pd.DataFrame(rows)


def make_figures(out:Path,selected:pd.DataFrame,seeds:pd.DataFrame,perclass:pd.DataFrame):
    out.mkdir(exist_ok=True,parents=True)
    fig,ax=plt.subplots(figsize=(10,4.5))
    labels=[s.replace('_seed','').replace('cnn_e2a50','CNN E2a-50').replace('anfis_r','ANFIS ') for s in selected.model]
    x=np.arange(len(labels)); vals=selected.macro_f1.to_numpy()
    bars=ax.bar(x,vals)
    ax.set_ylim(0,0.65);ax.set_ylabel('Validation macro-F1');ax.set_xticks(x,labels,rotation=20,ha='right')
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+0.007,f'{v:.3f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(out/'selected_macro_f1.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    x=np.arange(len(seeds));ax.bar(x,seeds.macro_f1_mean,yerr=seeds.macro_f1_sd,capsize=5)
    ax.set_xticks(x,seeds.family.str.replace('anfis_r','ANFIS ').tolist());ax.set_ylim(0,0.65);ax.set_ylabel('Validation macro-F1: mean ± sample SD (3 seeds)')
    fig.tight_layout();fig.savefig(out/'three_seed_macro_f1.png',dpi=170);plt.close(fig)
    p=perclass[perclass.model.isin(selected.model)].pivot(index='model',columns='class',values='recall').reindex(index=selected.model,columns=CLASSES)
    fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(p.values,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set_yticks(range(len(p.index)),p.index,fontsize=8);ax.set_xticks(range(len(p.columns)),p.columns)
    for i in range(len(p.index)):
        for j in range(len(p.columns)):
            ax.text(j,i,f'{p.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if p.iloc[i,j]<0.48 else 'black')
    fig.colorbar(im,ax=ax,label='Validation recall');fig.tight_layout();fig.savefig(out/'selected_per_class_recall.png',dpi=170);plt.close(fig)


def run_analysis(start=None,n_boot=1200,seed=20261109)->dict:
    root=locate_root(start)
    out=root/'outputs'/'ACIT4620_09_Validation_Comparison'
    out.mkdir(parents=True,exist_ok=True)
    provenance=verify_sources(root)
    datasets,ref,load_info=load_predictions(root)
    metrics,per_class=metrics_and_classwise(datasets)
    verify_reported_metrics(root,metrics)
    seeds=seed_table(metrics)
    selected=selected_checkpoints(metrics)
    errors=paired_error_table(datasets,selected,ref)
    pairs=[('mlp_seed42','anfis_r16_seed52'),('linear_seed42','anfis_r16_seed52'),
           ('cnn_e2a50','mlp_seed42'),('cnn_e2a50','anfis_r16_seed52'),('mlp_seed42','anfis_r8_seed42')]
    boots=paired_cluster_bootstrap(datasets,ref,pairs,n_boot=n_boot,seed=seed)
    metrics.to_csv(out/'all_models_validation_metrics.csv',index=False)
    per_class.to_csv(out/'per_class_validation_metrics.csv',index=False)
    seeds.to_csv(out/'three_seed_summary.csv',index=False)
    selected.to_csv(out/'validation_selected_checkpoints.csv',index=False)
    errors.to_csv(out/'paired_error_overlap.csv',index=False)
    boots.to_csv(out/'paired_lesion_bootstrap.csv',index=False)
    ref.to_csv(out/'matched_validation_manifest.csv',index=False)
    make_figures(out,selected,seeds,per_class)
    efficiency_e3=pd.read_csv(root/'inputs'/'e3'/'e3_all_seed_metrics.csv')
    efficiency_e4=pd.read_csv(root/'inputs'/'e4'/'e4_all_seed_metrics.csv')
    cols=['family','seed','train_seconds','trainable_params']
    eff3=efficiency_e3.assign(family=efficiency_e3['head'])[cols]
    eff4=efficiency_e4.assign(family='anfis_r'+efficiency_e4.n_rules.astype(str))[cols]
    pd.concat([eff3,eff4],ignore_index=True).to_csv(out/'reported_head_training_costs.csv',index=False)
    manifest={'experiment':'ACIT4620 Notebook 09 — validation comparison','selection':'maximum validation macro-F1 per architecture/rule count; selected seed is not independent test estimate',
              'comparison':'same 1441 validation images and 1067 lesions','class_codes':CLASSES,'split_sha256':SPLIT_HASH,'source_checkpoint_sha256':SOURCE_CKPT_HASH,
              'held_out_test_evaluated':False,'source_provenance':provenance,'probability_diagnostics':load_info,
              'bootstrap':{'n':n_boot,'seed':seed,'cluster_unit':'lesion_id','stratified_by':'true diagnosis','reused_for_selection':True,'includes_training_seed_variability':False,'includes_model_selection_uncertainty':False},
              'computational_cost_note':'Head-only recorded training seconds from E3/E4; not matched end-to-end CNN+head benchmark. Reported E4 head inference times are not compared with E3 without same measurement method.'}
    (out/'evaluation_metadata.json').write_text(json.dumps(manifest,indent=2))
    print('VERIFIED:',len(datasets),'prediction exports, same',N_VALID,'images,',N_LESIONS,'lesions, E2a-50 fingerprint')
    print('\nSELECTED CHECKPOINTS:')
    print(selected[['model','accuracy','macro_f1','balanced_accuracy','melanoma_recall']].round(4).to_string(index=False))
    print('\nTHREE-SEED SUMMARY:')
    print(seeds[['family','macro_f1_mean','macro_f1_sd','balanced_accuracy_mean','melanoma_recall_mean']].round(4).to_string(index=False))
    print('\nPAIRED BOOTSTRAP (model B minus model A):')
    print(boots[['model_a','model_b','delta_b_minus_a_macro_f1','ci95_low','ci95_high']].round(4).to_string(index=False))
    print('\nOutputs:',out)
    shutil.make_archive(str(root/'outputs'/'ACIT4620_09_Validation_Comparison_Outputs'),'zip',root_dir=out.parent,base_dir=out.name)
    print('ZIP:',root/'outputs'/'ACIT4620_09_Validation_Comparison_Outputs.zip')
    return {'root':root,'output_dir':out,'metrics':metrics,'seed_summary':seeds,'selected':selected,'bootstrap':boots,'per_class':per_class,'errors':errors}


if __name__=='__main__':
    run_analysis()

"""Corrected ACIT4620 Notebook 11: process-isolated sampled peak CPU RSS.

Runs each selected original classifier model in a FRESH Python process, using
unchanged checkpoints. Measures process-resident memory on Windows using psutil.
This is NOT exact peak tensor memory, nor an estimate of production deployment
RAM: the process also contains Python, Torch, libraries and shared CNN weights.
For the full pipeline, inputs are shape-matched 224x224 tensors. For head/training
steps, inputs are synthetic 1280-D features. No test data are opened.

Usage (from the inner Notebook 11 folder, using the working E3 Python .venv):
 python Notebook11_Peak_CPU_Memory_Addon.py --team-repo C:\\AI\\G1-NB10
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import shutil
import numpy as np
import pandas as pd
import psutil

NAMES = ['cnn_e2a50', 'linear_seed42', 'mlp_seed42',
         'anfis_r4_seed52', 'anfis_r8_seed42', 'anfis_r16_seed52']


def work(root: Path, team_repo: str|None, name: str, stage: str,
         batch: int, repeats: int, warmups: int, threads: int):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from torch import nn
    from efficiency_benchmark import load_models, FullPipeline, get_head_fn

    # Suppress loader's normal log lines so READY handshake is unambiguous.
    with contextlib.redirect_stdout(io.StringIO()):
        session = load_models(root, team_repo=team_repo, device='cpu',
                              cpu_threads=threads)
    # The loader validates all checkpoint provenance. Keep only the requested
    # head in this worker, so other classifiers do not remain resident.
    if name == 'cnn_e2a50':
        session.heads.clear()
    else:
        only = session.heads[name]
        session.heads.clear()
        session.heads[name] = only
    gc.collect()
    torch.manual_seed(20261109)

    if stage == 'head':
        model = get_head_fn(session,name).eval()
        x = torch.randn(batch,1280)
        def run():
            with torch.inference_mode():
                z=model(x)
            assert z.shape==(batch,7)
    elif stage == 'full':
        model=FullPipeline(session,name).eval()
        x=torch.randn(batch,3,224,224)
        def run():
            with torch.inference_mode():
                z=model(x)
            assert z.shape==(batch,7)
    elif stage == 'train':
        if name == 'cnn_e2a50':
            raise ValueError('CNN fine-tuning is not part of the matched head-only training-step experiment')
        # Already-loaded head checkpoint is used as the starting point, but
        # its weights are NOT saved or modified outside this throwaway worker.
        model = session.heads[name].train().requires_grad_(True)
        x = torch.randn(batch,1280)
        y = torch.randint(0,7,(batch,))
        weights = torch.tensor([.211962,1.299963,1.316802,2.839585,4.419876,11.42215,11.42215])
        opt = torch.optim.Adam(model.parameters(),lr=1e-3) if name.startswith('anfis') else torch.optim.AdamW(model.parameters(),lr=1e-3)
        def run():
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x),y,weight=weights)
            loss.backward()
            opt.step()
    else:
        raise ValueError(stage)

    # Measure from immediately before the first iteration, including warmups,
    # to catch first-use allocation (e.g. optimizer state and CNN workspaces).
    print(f'READY {os.getpid()}', flush=True)
    if sys.stdin.readline().strip() != 'GO':
        raise RuntimeError('Measurement handshake aborted')
    for _ in range(warmups+repeats):
        run()
    print('DONE', flush=True)


def measure(root, team_repo, name, stage, batch, repeats, warmups, threads, sample_ms):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker',
           '--project-root', str(root), '--model', name,
           '--stage', stage, '--batch', str(batch),
           '--repeats', str(repeats), '--warmups', str(warmups),
           '--threads', str(threads)]
    if team_repo:
        cmd.extend(['--team-repo',str(team_repo)])
    p=subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, bufsize=1)
    ready=p.stdout.readline().strip()
    parts=ready.split()
    if len(parts)!=2 or parts[0]!='READY' or not parts[1].isdigit():
        msg=p.stderr.read()
        p.wait()
        raise RuntimeError(f'{stage}/{name}: worker did not initialize ({ready!r}): {msg[-1200:]}')
    # CRITICAL ON WINDOWS: .venv\Scripts\python.exe may be a small
    # redirector/launcher which starts the actual Python worker process.
    # Sampling p.pid can therefore return implausible ~4 MiB for every model.
    # The loaded PyTorch worker explicitly reports its own OS process ID.
    worker_pid=int(parts[1])
    worker=psutil.Process(worker_pid)
    baseline=worker.memory_info().rss/1048576
    # A loaded 4-million-parameter CNN and PyTorch runtime will exceed 100 MiB.
    # Refuse to publish a suspicious reading instead of hiding the problem.
    if baseline < 100:
        p.stdin.write('STOP\n'); p.stdin.flush()
        p.communicate()
        raise RuntimeError(f'{stage}/{name}: implausible {baseline:.1f} MiB RSS for PID {worker_pid}; invalid worker/measurement')
    peak=baseline
    samples=0
    p.stdin.write('GO\n')
    p.stdin.flush()
    started=time.perf_counter()
    while p.poll() is None:
        try:
            rss=worker.memory_info().rss/1048576
            peak=max(peak,rss)
            samples+=1
        except psutil.NoSuchProcess:
            break
        time.sleep(sample_ms/1000)
    out,err=p.communicate()
    if p.returncode!=0 or 'DONE' not in out:
        raise RuntimeError(f'{stage}/{name}: worker failed: {err[-2000:]}')
    return dict(model=name,stage=stage,batch_size=batch,device='cpu',
                worker_pid=worker_pid,launcher_pid=p.pid,
                process_baseline_rss_mb=round(baseline,3),
                sampled_peak_process_rss_mb=round(peak,3),
                sampled_incremental_peak_rss_mb=round(max(0,peak-baseline),3),
                sampled_points=samples, sample_interval_ms=sample_ms,
                warmups=warmups,repeats=repeats,cpu_threads=threads,
                duration_including_setup_seconds=round(time.perf_counter()-started,2),
                image_input='synthetic_3x224x224' if stage=='full' else 'synthetic_1280D')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--team-repo',type=str,default=None)
    parser.add_argument('--stages',nargs='+',choices=['full','head','train'],default=['full','head','train'])
    parser.add_argument('--models',nargs='+',choices=NAMES,default=NAMES)
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--warmups',type=int,default=5)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--sample-ms',type=float,default=2.0)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--model',type=str)
    parser.add_argument('--stage',choices=['full','head','train'])
    parser.add_argument('--batch',type=int)
    args=parser.parse_args()
    root=args.project_root.resolve()
    assert (root/'efficiency_benchmark.py').is_file(), 'Run from the inner Notebook 11 project folder.'
    if args.worker:
        work(root,args.team_repo,args.model,args.stage,args.batch,args.repeats,args.warmups,args.threads)
        return
    if args.repeats<3 or args.warmups<1 or args.sample_ms<=0:
        raise ValueError('Use at least 3 repeats, 1 warmup and a positive sampling interval.')
    out_dir=root/'outputs'/'ACIT4620_11_Efficiency_Results'
    out_dir.mkdir(exist_ok=True,parents=True)
    rows=[]
    for stage in args.stages:
        for name in args.models:
            if stage=='train' and name=='cnn_e2a50':
                continue
            batch=128 if stage=='train' else (8 if stage=='full' else 32)
            print(f'Measuring {stage}: {name} (batch {batch})', flush=True)
            rows.append(measure(root,args.team_repo,name,stage,batch,
                                args.repeats,args.warmups,args.threads,args.sample_ms))
    df=pd.DataFrame(rows)
    csv=out_dir/'sampled_peak_cpu_rss_isolated_processes.csv'
    df.to_csv(csv,index=False)
    notes={
        'method':'Sample actual worker-PID RSS, reported by os.getpid() after checkpoint loading, not the virtualenv redirector process.',
        'windows_launcher_guard':'Worker emits READY <os.getpid()>; validate baseline above 100 MiB. Addresses old invalid ~4.125 MiB samples.',
        'memory_metric':'Sampled PEAK process-resident memory (RSS), not torch-only allocation or exact instantaneous hardware peak.',
        'scope':'Whole separate Python process, including Python/Torch libraries, loaded CNN and one selected head; not per-model parameter memory.',
        'incremental_delta_note':'Peak RSS minus post-load process RSS. Small differences may be masked by allocator reuse, process noise, and finite sampling.',
        'input_note':'Shape-matched synthetic 224x224 images for full model; synthetic 1280-dimensional features for head and training. Not a real-JPEG memory experiment.',
        'cpu_threads':args.threads,'sample_interval_ms':args.sample_ms,
        'repeats':args.repeats,'warmups':args.warmups,'source':'Notebook 11 verified CNN/E3/E4 checkpoints',
        'test_set_accessed':False,
    }
    (out_dir/'sampled_peak_cpu_rss_method.json').write_text(json.dumps(notes,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot=df[df.stage=='full']
        if not plot.empty:
            fig,ax=plt.subplots(figsize=(10,5))
            ax.bar(plot.model,plot.sampled_peak_process_rss_mb)
            ax.set_title('Whole-process sampled peak CPU RSS: full CNN + head, batch 8')
            ax.set_ylabel('Peak CPU resident memory (MiB)')
            ax.tick_params(axis='x',rotation=35)
            fig.tight_layout()
            (out_dir/'figures').mkdir(exist_ok=True)
            fig.savefig(out_dir/'figures'/'06_isolated_process_sampled_peak_cpu_rss.png',dpi=180)
            plt.close(fig)
    except ImportError:
        pass
    # Do not overwrite the earlier invalid memory archive; preserve it for audit.
    zip_name=shutil.make_archive(str(root/'outputs'/'ACIT4620_11_Efficiency_Results_With_Peak_Memory_Fixed'),
                                 'zip',root_dir=root/'outputs',base_dir='ACIT4620_11_Efficiency_Results')
    with zipfile.ZipFile(zip_name) as z:
        assert z.testzip() is None
    print('\nMemory summary:')
    print(df[['model','stage','batch_size','process_baseline_rss_mb','sampled_peak_process_rss_mb','sampled_incremental_peak_rss_mb']].to_string(index=False))
    print('Saved CSV:',csv)
    print('New ZIP (original untouched):',zip_name)
    print('CAUTION: This is sampled PROCESS RSS of actual worker PID, not model-specific peak tensor memory.')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Attach real Bilby/LAL noisy waveform features to a solved lens catalogue.

Reuses the existing project's source and ET/PSD simulator. Uses the Morse
index of each actual image (not fixed SIS parity). Separate images in geometric
optics; no wave-optics claim. All feature functions consume noisy data only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import importlib.metadata
from dataclasses import asdict
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.signal import correlate, correlation_lags


def load_base(path):
    # Preload matplotlib before gwpy, with runtime library fixed by shell runner.
    import matplotlib
    spec = importlib.util.spec_from_file_location('mnras_original_waveforms', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def observed_features(base, x, fs, delay, rng):
    a,b=x
    d=base.window_stats_pair(a,b,fs,.5)
    xc=correlate(b,a,mode='full',method='fft')
    peak=int(np.argmax(np.abs(xc)))
    lag=int(correlation_lags(len(b),len(a),mode='full')[peak])
    corr=float(xc[peak]/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
    c0,l0,m0,h0=base.spectrum_features(a,fs)
    c1,l1,m1,h1=base.spectrum_features(b,fs)
    snr=[base.observed_window_snr_proxy(v,fs,.5) for v in x]
    # Timing error model depends only on observed noisy-window SNR estimates.
    sig=np.sqrt(sum((1/fs+.03/max(s,1e-3))**2 for s in snr))
    ret={'delay_seconds':delay+rng.normal(0,sig),'delay_sigma_seconds':sig,
         'xcorr_lag_samples':lag,'xcorr_max':corr,'spectral_centroid_diff':c1-c0,
         'band_low_ratio':base.safe_ratio(l1,l0),'band_mid_ratio':base.safe_ratio(m1,m0),
         'band_high_ratio':base.safe_ratio(h1,h0)}
    ret.update(d)
    for j,v in enumerate(x):
        ret.update({f'rms_{j}':float(np.sqrt(np.mean(v*v))),f'peak_{j}':float(np.max(np.abs(v))),
                    f'noise_edge_rms_{j}':base.edge_noise_rms(v,fs),f'observed_snr_proxy_{j}':snr[j]})
    return {'obs__gw_'+k:float(v) for k,v in ret.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--base-generator',type=Path,default=Path(__file__).resolve().parents[2]/'SIS_GW_physics_baseline.py')
    p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--noise-multipliers',nargs='+',type=float,default=[.5,1.,2.,4.])
    p.add_argument('--detectors',nargs='+',default=['ET'])
    p.add_argument('--sampling-frequency',type=int,default=4096)
    p.add_argument('--duration',type=float,default=24.)
    p.add_argument('--approximant',default='IMRPhenomXPHM')
    p.add_argument('--max-events',type=int,default=0,help='Smoke test only; 0 means every event.')
    a=p.parse_args()
    if any(s<=0 for s in a.noise_multipliers): p.error('Noise multipliers must be positive')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    base=load_base(a.base_generator)
    cfg=base.Config(); cfg.detector_names=tuple(a.detectors); cfg.sampling_frequency=a.sampling_frequency
    cfg.duration=a.duration; cfg.waveform_approximant=a.approximant; cfg.save_full_strain=False
    cfg.geocent_time_includes_delay=True; base.CFG=cfg
    source=pd.read_csv(a.source_root/'source_truth.csv').set_index('event_id',verify_integrity=True)
    generator=base.build_waveform_generator(cfg)
    keys=['mass_1','mass_2','luminosity_distance','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','theta_jn','phase','psi','ra','dec','geocent_time']
    # Store input digests so resuming cannot silently reuse a different catalogue.
    input_digests={fd.name:{n:hashlib.sha256((fd/n).read_bytes()).hexdigest() for n in ['observables.csv','image_latent.csv']} for fd in a.benchmark_root.iterdir() if (fd/'observables.csv').exists()}
    versions={m:importlib.metadata.version(m) for m in ['bilby','numpy','scipy','lalsuite','gwpy','astropy']}
    detector_psd_hashes={}
    for ifo in base.bilby.gw.detector.InterferometerList(list(cfg.detector_names)):
        psd=ifo.power_spectral_density
        content=np.column_stack([psd.frequency_array,psd.psd_array]).astype('float64').tobytes()
        detector_psd_hashes[ifo.name]=hashlib.sha256(content).hexdigest()
    provenance={'base_generator_sha256':hashlib.sha256(a.base_generator.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_config':asdict(cfg),'versions':versions,'detector_psd_sha256':detector_psd_hashes}
    manifest={'args':vars(a),'input_digests':input_digests,'provenance':provenance,'source_sha256':hashlib.sha256((a.source_root/'source_truth.csv').read_bytes()).hexdigest(),
              'scope':'geometric optics, independently simulated selected images, Gaussian detector PSD noise',
              'selection':'all successful input images, no SNR cut; failed simulations abort',
              'snr_scan':'global PSD noise amplitude multipliers shared by every event, retain signal and source distance; pair SNR is diagnostic only',
              'level_multipliers':{'baseline':1.}|{f'sensitivity_{s:g}':s for s in a.noise_multipliers},
              'centering':'injected peak centered windows; controlled idealized trigger, not a detection pipeline',
              'timing':'catalogue Gaussian timing error using observed window SNR; not waveform posterior inference',
              'feature_contract':'obs__gw_ only; injection optimal SNR, true parameters and noise scale excluded'}
    manifest_path=a.out_dir/'waveform_manifest.json'
    serialized=json.dumps(manifest,indent=2,default=str)
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=json.loads(serialized):
        raise RuntimeError('Waveform output configuration or catalogue changed. Use a new out-dir; stale caches cannot be reused.')
    manifest_path.write_text(serialized)
    for fd in sorted(a.benchmark_root.iterdir()):
        if not (fd/'observables.csv').exists(): continue
        events=pd.read_csv(fd/'observables.csv'); imgs=pd.read_csv(fd/'image_latent.csv')
        if a.max_events: events=events.head(a.max_events)
        dest=a.out_dir/fd.name; dest.mkdir(exist_ok=True); cache=dest/'windows'; cache.mkdir(exist_ok=True)
        rows={'baseline':[]}|{f'sensitivity_{s:g}':[] for s in a.noise_multipliers}; diagnostics=[]
        for ev in tqdm(events.to_dict('records'),desc=f'{fd.name} real GW pairs',unit='event'):
            eid=int(ev['event_id']); group=imgs[imgs.event_id==eid]
            pair=group.sort_values('mu_abs_true',ascending=False).head(2)
            if len(pair)!=2: raise ValueError(f'{fd.name}/{eid}: requires two images')
            src={k:float(source.loc[int(ev['source_event_id']),k]) for k in keys}
            # Morse phase = -pi/2 per negative Jacobian eigenvalue.
            morse=np.array([(r['lambda_1_true']<0)+(int(r['lambda_2_true']<0)) for r in pair.to_dict('records')],float)
            mu=pair.mu_abs_true.to_numpy(float)
            delay=float(pair.iloc[1]['time_delay_true_s']-pair.iloc[0]['time_delay_true_s'])
            if not np.isfinite(delay): raise ValueError('Physical seconds delay missing')
            token=json.dumps([a.seed,fd.name,eid,src,mu.tolist(),morse.tolist(),delay,a.detectors,a.sampling_frequency,a.duration,a.approximant,provenance],sort_keys=True,default=str)
            digest=hashlib.sha256(token.encode()).hexdigest(); cachefile=cache/f'{eid}_{digest[:16]}.npz'
            seed=int(digest[:8],16); rng=np.random.default_rng(seed)
            if cachefile.exists():
                z=np.load(cachefile); signal=z['signal']; noise=z['noise']; snrs=z['snrs']
            else:
                np.random.seed(seed)
                try: base.bilby.core.utils.random.seed(seed)
                except AttributeError: pass
                def amp(frequency_array,mu_0,mu_1,t_d,which_image,cfg):
                    j=int(which_image)
                    return np.full(len(frequency_array),np.sqrt([mu_0,mu_1][j])*np.exp(-.5j*np.pi*morse[j]),complex)
                base.lens_amplification=amp
                generator=base.build_waveform_generator(cfg)
                signals=[]; noises=[]; snrs=[]
                for j in range(2):
                    pars=src|{'geocent_time':src['geocent_time']+(delay if j else 0.),'mu_0':mu[0],'mu_1':mu[1],'t_d':delay,'which_image':j}
                    result,reason=base.simulate_image(pars,j,generator,cfg)
                    if result is None: raise RuntimeError(f'{fd.name}/{eid} waveform failure: {reason}')
                    signals.append(result['reference']['signal_white']); noises.append(result['reference']['noise_white']); snrs.append(result['network_snr'])
                signal=np.array(signals); noise=np.array(noises); snrs=np.array(snrs)
                if not np.isfinite(signal).all() or not np.isfinite(noise).all(): raise ValueError('Nonfinite waveform')
                signal=signal.astype('float32');noise=noise.astype('float32')
                np.savez_compressed(cachefile,signal=signal,noise=noise,snrs=snrs)
            pair_snr=float(np.linalg.norm(snrs))
            if not np.isfinite(pair_snr) or pair_snr<=0: raise ValueError('Invalid pair SNR')
            # Same random timing variate for all SNR levels (paired study).
            for level in rows:
                scale=1. if level=='baseline' else float(level.split('_')[1])
                x=signal+noise*scale
                row={'event_id':eid,'lens_family':fd.name,**observed_features(base,x,cfg.sampling_frequency,delay,np.random.default_rng(seed+1))}
                if not np.isfinite([v for k,v in row.items() if k.startswith('obs__gw_')]).all(): raise ValueError('Nonfinite measured features')
                rows[level].append(row)
            diagnostics.append({'event_id':eid,'lens_family':fd.name,'optimal_pair_snr_true':pair_snr,'source_event_id':ev['source_event_id'],'cache_file':cachefile.name})
        for level,records in rows.items():
            leveldir=dest/level;leveldir.mkdir(exist_ok=True)
            pd.DataFrame(records).to_csv(leveldir/'waveform_features.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(dest/'waveform_diagnostics.csv',index=False)
    print(f'WAVEFORM_OUT={a.out_dir}',flush=True)

if __name__=='__main__': main()

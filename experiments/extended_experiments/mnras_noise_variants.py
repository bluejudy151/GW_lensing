#!/usr/bin/env python
"""Paired re-observation of one exact lens benchmark; preserves identities/splits.

Creates one evaluator-ready catalogue per noise scenario. No waveform is
synthesized: optional measured waveform CSVs are joined strictly by event_id.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

BASE={'astrometry_sigma':.01,'lens_center_sigma':.02,'redshift_sigma':5e-4,
      'sigma_v_fraction':.05,'theta_e_fraction':.02}
DRAW_NAMES=['image0_x','image0_y','image1_x','image1_y','center_x','center_y','z_l','z_s','sigma_v','theta_E']
SCENARIOS=['baseline','low','high','correlated','heavy_tailed'] + [
    f'{key}_{level}' for key in ['astrometry','lens_center','redshift','sigma_v','theta_e'] for level in ['low','high']]
PARAMETER={'astrometry':'astrometry_sigma','lens_center':'lens_center_sigma','redshift':'redshift_sigma',
           'sigma_v':'sigma_v_fraction','theta_e':'theta_e_fraction'}


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--waveform-root',type=Path)
    p.add_argument('--seed',type=int,default=88173)
    p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=SCENARIOS)
    p.add_argument('--student-df',type=float,default=4.)
    p.add_argument('--low-multiplier',type=float,default=.5)
    p.add_argument('--high-multiplier',type=float,default=2.)
    p.add_argument('--include-snr-variants',action='store_true')
    for name,value in BASE.items(): p.add_argument('--'+name.replace('_','-'),type=float,default=None)
    a=p.parse_args()
    if a.student_df<=2: p.error('student-df must exceed 2 for variance normalization.')
    if not 0<a.low_multiplier<1<a.high_multiplier: p.error('Require 0 < low multiplier < 1 < high multiplier.')
    if a.include_snr_variants and not a.waveform_root: p.error('SNR variants require --waveform-root.')
    return a


def base_noise(a,manifest):
    stored=manifest.get('args',{})
    return {key:float(getattr(a,key) if getattr(a,key) is not None else stored.get(key,default)) for key,default in BASE.items()}


def correlation_matrix():
    R=np.eye(len(DRAW_NAMES))
    # Correlate measurements across the two images, the two centroid axes,
    # and the two redshifts. Disjoint blocks make positive definiteness clear.
    for i,j,r in [(0,2,.6),(1,3,.6),(4,5,.3),(6,7,.5)]: R[i,j]=R[j,i]=r
    return R


def scenario_config(name,base,a):
    noise=base.copy(); correlation=np.eye(len(DRAW_NAMES)); kind='gaussian'
    if name in ['low','high']:
        factor=a.low_multiplier if name=='low' else a.high_multiplier
        noise={k:v*factor for k,v in noise.items()}
    elif name=='correlated': correlation=correlation_matrix()
    elif name=='heavy_tailed': kind='student_t'
    elif name!='baseline':
        key,level=name.rsplit('_',1)
        noise[PARAMETER[key]]*=a.low_multiplier if level=='low' else a.high_multiplier
    return noise,correlation,kind


def rng_draws(seed,family,eid,df):
    digest=hashlib.sha256(f'{seed}:{family}:{eid}'.encode()).digest()
    rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
    # Same normal components in every scenario and every invocation; these
    # draws never depend on family iteration order or selected scenarios.
    return rng.standard_normal(len(DRAW_NAMES)),float(rng.chisquare(df))


def observe(row,p,s,noise,draw):
    xo=np.array([p.x_true,s.x_true])+noise['astrometry_sigma']*draw[[0,2]]
    yo=np.array([p.y_image_true,s.y_image_true])+noise['astrometry_sigma']*draw[[1,3]]
    center=noise['lens_center_sigma']*draw[[4,5]]
    radii=np.hypot(xo-center[0],yo-center[1]); sep=float(np.hypot(xo[0]-xo[1],yo[0]-yo[1]))
    asym=float((radii[0]-radii[1])/max(float(radii.sum()),1e-15))
    zl=max(float(row.z_l_true+noise['redshift_sigma']*draw[6]),1e-5)
    zs=max(float(row.z_s_true+noise['redshift_sigma']*draw[7]),zl+1e-3)
    sig=max(float(row.sigma_v_true*(1+noise['sigma_v_fraction']*draw[8])),1e-6)
    scale=max(float(row.theta_E_true*(1+noise['theta_e_fraction']*draw[9])),1e-6)
    return {'obs__z_s_observed':zs,'obs__z_l_observed':zl,'obs__z_l_over_z_s':zl/zs,
      'obs__sigma_v_observed':sig,'obs__log_sigma_v_observed':float(np.log(sig)),
      'obs__theta_E_observed':max(sep/2,1e-8),'obs__image_separation_observed':sep,
      'obs__theta_plus_abs_observed':float(radii[0]),'obs__theta_minus_abs_observed':float(radii[1]),
      'obs__image_position_asymmetry_observed':asym,
      'obs__lens_center_x_observed':float(center[0]),'obs__lens_center_y_observed':float(center[1]),
      'obs__image_x_0_observed':float(xo[0]),'obs__image_y_0_observed':float(yo[0]),
      'obs__image_x_1_observed':float(xo[1]),'obs__image_y_1_observed':float(yo[1]),
      'obs__theta_E_observed_direct':scale,'analytic_mu0_from_image_asymmetry':1+1/max(abs(asym),1e-6)}


def wave_file(root,family,scenario):
    candidates=[root/family/scenario/'waveform_features.csv',root/scenario/family/'waveform_features.csv']
    if scenario=='baseline': candidates += [root/family/'waveform_features.csv']
    found=[f for f in candidates if f.exists()]
    if len(found)!=1: raise FileNotFoundError(f'Expected exactly one waveform file for {family}/{scenario}: {candidates}')
    return found[0]


def merge_wave(obs,path):
    wave=pd.read_csv(path)
    if 'event_id' not in wave or wave.event_id.duplicated().any(): raise ValueError(f'Invalid waveform identity table: {path}')
    if set(obs.event_id)!=set(wave.event_id): raise ValueError(f'Waveform event IDs must match exactly: {path}')
    cols=[c for c in wave if c.startswith('gw__') or c.startswith('obs__gw_')]
    if not cols: raise ValueError(f'No gw__ or obs__gw_ measurement features: {path}')
    if not np.isfinite(wave[cols].to_numpy(float)).all(): raise ValueError(f'Nonfinite waveform measurements: {path}')
    if any(t in c.lower() for c in cols for t in ['_true','latent','mu0','mu1','determinant','jacobian']):
        raise ValueError(f'Truth/latent waveform feature forbidden: {path}')
    measured=wave[['event_id']+cols].rename(columns={c:'obs__gw_'+c[4:] for c in cols if c.startswith('gw__')})
    obs=obs.drop(columns=[c for c in obs if c.startswith('gw__') or c.startswith('obs__gw_')])
    obs=obs.merge(measured,on='event_id',how='left',validate='one_to_one')
    obs['waveform_features_available']=True
    return obs


def save_family(folder,obs,lat,img,source):
    folder.mkdir(parents=True,exist_ok=False)
    # Refresh the observed contract in the primary latent view by keyed join.
    refresh=[c for c in obs if c.startswith('obs__') or c.startswith('gw__') or c in
             ['analytic_mu0_from_image_asymmetry','waveform_features_available']]
    lat=lat.drop(columns=[c for c in lat if c in refresh or c.startswith('obs__gw_') or c.startswith('gw__')])
    lat=lat.merge(obs[['event_id']+refresh],on='event_id',validate='one_to_one')
    for name,data in [('observables',obs),('event_latent',lat),('image_latent',img)]:
        data.to_csv(folder/f'{name}.csv',index=False)
        if 'split' not in data: raise ValueError(f'{name}: stored split required')
        for split,part in data.groupby('split'):
            sd=folder/str(split); sd.mkdir(exist_ok=True); part.to_csv(sd/f'{name}.csv',index=False)
    for name in ['event_manifest.csv','latent_validation_summary.csv','generation_config.json']:
        if (source/name).exists(): shutil.copy2(source/name,folder/name)


def main():
    a=parse(); source_manifest=json.loads((a.benchmark_root/'manifest.json').read_text())
    if source_manifest.get('status')!='complete': raise ValueError('Source benchmark is not marked complete.')
    base=base_noise(a,source_manifest)
    if any(v<0 for v in base.values()): raise ValueError('Noise scales must be nonnegative.')
    families=sorted(p for p in a.benchmark_root.iterdir() if (p/'observables.csv').exists())
    if a.waveform_root:
        wave_manifest=json.loads((a.waveform_root/'waveform_manifest.json').read_text())
        for folder in families:
            expected=wave_manifest.get('input_digests',{}).get(folder.name,{}).get('image_latent.csv')
            if expected!=hashlib.sha256((folder/'image_latent.csv').read_bytes()).hexdigest():
                raise ValueError(f'Waveform physical-catalogue hash mismatch: {folder.name}')
    if not families: raise ValueError('No source family tables.')
    scenarios=[(s,s,'baseline') for s in a.scenarios]
    if a.include_snr_variants:
        first=a.waveform_root/families[0].name
        snrs=sorted(p.name for p in first.iterdir() if p.is_dir() and p.name.startswith('sensitivity_'))
        if not snrs: raise FileNotFoundError(f'No sensitivity_* waveform directories under {first}')
        scenarios += [(s,'baseline',s) for s in snrs]
    a.out_root.mkdir(parents=True,exist_ok=True)
    loaded={}
    for fd in families:
        obs=pd.read_csv(fd/'observables.csv'); lat=pd.read_csv(fd/'event_latent.csv'); img=pd.read_csv(fd/'image_latent.csv')
        if obs.event_id.duplicated().any() or lat.event_id.duplicated().any() or img.duplicated(['event_id','image_id']).any():
            raise ValueError(f'Duplicate event or image identity: {fd}')
        if set(obs.event_id)!=set(lat.event_id) or set(obs.event_id)!=set(img.event_id): raise ValueError(f'Missing event/image join: {fd}')
        required=['z_l_true','z_s_true','sigma_v_true','theta_E_true','primary_image_id','secondary_image_id','split','source_event_id']
        if any(c not in obs for c in required): raise ValueError(f'{fd} needs new true-latent generator columns: {required}')
        lookup=img.set_index(['event_id','image_id'])
        draws=[rng_draws(a.seed,fd.name,int(eid),a.student_df) for eid in obs.event_id]
        loaded[fd.name]=(obs,lat,img,lookup,draws)
    for name,physical_scenario,wave_scenario in tqdm(scenarios,desc='Noise scenarios',unit='scenario'):
        output=a.out_root/name; output.mkdir(exist_ok=False)
        noise,R,kind=scenario_config(physical_scenario,base,a); chol=np.linalg.cholesky(R)
        # Coordinates mix angular, redshift and fractional variables; record
        # covariance of standardized draws and scales instead of a false
        # universal physical covariance for fractional sigma_v/theta_E.
        nm={'scenario':name,'physical_scenario':physical_scenario,'noise_scales':noise,
            'draw_order':DRAW_NAMES,'standardized_covariance':R.tolist(),'distribution':kind,
            'student_df':a.student_df if kind=='student_t' else None,
            'student_variance_normalization':'sqrt((df-2)/chi2_df)' if kind=='student_t' else None,
            'seed':a.seed,'paired_draws':'SHA256(seed:family:event_id); identical normal draws across all scenarios.',
            'clipping':'z_l>=1e-5, z_s>=z_l+1e-3, sigma_v>=1e-6, direct scale>=1e-6',
            'waveform_source':str(a.waveform_root.resolve()) if a.waveform_root else None,
            'waveform_scenario':wave_scenario if a.waveform_root else None,
            'geometry_note':'All images are true brightness-ordered; 2D Euclidean separation; direct reference scale independently perturbed.'}
        manifest=dict(source_manifest); manifest.update(status='running',noise_variant=nm,source_benchmark=str(a.benchmark_root.resolve()))
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
        try:
            for fd in families:
                original,lat,img,lookup,draws=loaded[fd.name]; obs=original.copy(); records=[]
                for i,row in enumerate(tqdm(obs.itertuples(index=False),total=len(obs),desc=f'{name}/{fd.name}',unit='event',leave=False)):
                    normal,chi2=draws[i]; draw=chol@normal
                    if kind=='student_t': draw=draw*np.sqrt((a.student_df-2)/chi2)
                    p=lookup.loc[(row.event_id,row.primary_image_id)]; s=lookup.loc[(row.event_id,row.secondary_image_id)]
                    records.append(observe(row,p,s,noise,draw))
                measured=pd.DataFrame(records,index=obs.index)
                obs[measured.columns]=measured
                if a.waveform_root:
                    obs=merge_wave(obs,wave_file(a.waveform_root,fd.name,wave_scenario))
                    diag=pd.read_csv(a.waveform_root/fd.name/'waveform_diagnostics.csv')
                    diag['diagnostic__optimal_pair_snr']=diag.optimal_pair_snr_true/wave_manifest['level_multipliers'][wave_scenario]
                    obs=obs.drop(columns=['diagnostic__optimal_pair_snr'],errors='ignore').merge(diag[['event_id','diagnostic__optimal_pair_snr']],on='event_id',validate='one_to_one')
                save_family(output/fd.name,obs,lat,img,fd)
                (output/fd.name/'noise_config.json').write_text(json.dumps(nm,indent=2))
            if (a.benchmark_root/'latent_validation_summary.csv').exists(): shutil.copy2(a.benchmark_root/'latent_validation_summary.csv',output/'latent_validation_summary.csv')
            manifest['status']='complete'; manifest['waveform_features_available']=bool(a.waveform_root) or bool(source_manifest.get('waveform_features_available',False))
        except Exception as exc:
            manifest['status']='failed'; manifest['failure']=repr(exc); raise
        finally:
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    (a.out_root/'noise_scenarios.json').write_text(json.dumps({'source':str(a.benchmark_root.resolve()),'scenarios':[s[0] for s in scenarios],'seed':a.seed},indent=2))
    print(f'NOISE_VARIANTS_ROOT={a.out_root.resolve()}',flush=True)


if __name__=='__main__': main()

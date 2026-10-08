#!/usr/bin/env python3
"""Join measured waveform features by family/event identity into a new benchmark."""
import argparse
import hashlib
import numpy as np
import json
import shutil
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root',type=Path,required=True)
    p.add_argument('--waveform-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--level',default='baseline')
    a=p.parse_args()
    provenance=json.loads((a.waveform_root/'waveform_manifest.json').read_text())
    for fd in a.benchmark_root.iterdir():
        if not (fd/'observables.csv').exists():continue
        expected=provenance.get('input_digests',{}).get(fd.name,{})
        for name in ['observables.csv','image_latent.csv']:
            if expected.get(name)!=hashlib.sha256((fd/name).read_bytes()).hexdigest():
                raise ValueError(f'Waveform catalogue provenance mismatch: {fd/name}')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    for fd in tqdm(sorted(a.benchmark_root.iterdir()),desc=f'Waveform join {a.level}'):
        if not (fd/'observables.csv').exists(): continue
        obs=pd.read_csv(fd/'observables.csv')
        feat=pd.read_csv(a.waveform_root/fd.name/a.level/'waveform_features.csv')
        if feat.event_id.duplicated().any() or set(feat.event_id)!=set(obs.event_id):
            raise ValueError(f'{fd.name}: waveform IDs must match entire event catalogue')
        if 'lens_family' not in feat or not feat.lens_family.eq(fd.name).all(): raise ValueError('Waveform family mismatch')
        cols=[c for c in feat if c.startswith('obs__gw_')]
        if not cols or not np.isfinite(feat[cols].to_numpy(float)).all(): raise ValueError('Missing waveform measurements')
        out=a.out_dir/fd.name;out.mkdir()
        obs=obs.drop(columns=[c for c in obs if c.startswith('obs__gw_')]).merge(feat[['event_id']+cols],on='event_id',validate='one_to_one')
        diag=pd.read_csv(a.waveform_root/fd.name/'waveform_diagnostics.csv')
        scale=provenance['level_multipliers'][a.level]
        diag['diagnostic__optimal_pair_snr']=diag.optimal_pair_snr_true/scale
        obs=obs.merge(diag[['event_id','diagnostic__optimal_pair_snr']],on='event_id',validate='one_to_one')
        obs['waveform_features_available']=True
        obs.to_csv(out/'observables.csv',index=False)
        for filename in ['event_manifest.csv','latent_validation_summary.csv','generation_config.json']:
            src=fd/filename
            if src.exists(): shutil.copy2(src,out/filename)
        lat=pd.read_csv(fd/'event_latent.csv');img=pd.read_csv(fd/'image_latent.csv')
        refresh=[c for c in obs if c.startswith('obs__gw_') or c in ['waveform_features_available','diagnostic__optimal_pair_snr']]
        lat=lat.drop(columns=[c for c in lat if c in refresh]).merge(obs[['event_id']+refresh],on='event_id',validate='one_to_one')
        lat.to_csv(out/'event_latent.csv',index=False);img.to_csv(out/'image_latent.csv',index=False)
        for split in ['train','validation','test']:
            dest=out/split;dest.mkdir()
            for name,table in [('observables',obs),('event_latent',lat),('image_latent',img)]:
                table[table.split==split].to_csv(dest/(name+'.csv'),index=False)
    manifest=json.loads((a.benchmark_root/'manifest.json').read_text())
    manifest.update(waveform_features_available=True,waveform_root=str(a.waveform_root.resolve()),waveform_level=a.level,
                    geometry_benchmark=str(a.benchmark_root.resolve()),feature_join='one-to-one family/event_id, exact ID sets')
    (a.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'JOINED_BENCHMARK={a.out_dir.resolve()}')

if __name__=='__main__':main()

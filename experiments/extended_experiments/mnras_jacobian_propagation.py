#!/usr/bin/env python3
"""Verify first-order magnification propagation with controlled Jacobian perturbations.
For absolute mu, delta_mu = -sign(detA)*mu^2*delta_detA. Diagnostic
perturbations to true A are not an observation model or learned estimator.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--relative-noise',nargs='+',type=float,default=[.001,.01,.1,.5])
    p.add_argument('--repeats',type=int,default=32);p.add_argument('--seed',type=int,default=8831)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False);rows=[];rng=np.random.default_rng(a.seed)
    for fd in sorted(a.benchmark_root.iterdir()):
        if not (fd/'event_latent.csv').exists():continue
        d=pd.read_csv(fd/'event_latent.csv');d=d[d.split=='test']
        for r in tqdm(d.to_dict('records'),desc=f'{fd.name} Jacobian propagation',unit='image'):
            A=np.array([[r['A11_true'],r['A12_true']],[r['A21_true'],r['A22_true']]])
            det=r['detA_true'];mu=1/abs(det);sm=r['singular_value_min_true']
            Z=rng.normal(size=(a.repeats,2,2));Z=(Z+Z.transpose(0,2,1))/2
            Z=Z/np.linalg.norm(Z,axis=(1,2))[:,None,None]
            for scale in a.relative_noise:
                perturbed=A+Z*scale*sm
                dp=np.linalg.det(perturbed);dt=dp-det
                exact=1/np.maximum(abs(dp),1e-30)-mu;linear=-np.sign(det)*mu**2*dt
                for k in range(a.repeats):
                    rows.append({'lens_family':fd.name,'event_id':r['event_id'],'group_id':r['group_id'],'repeat':k,
                                 'relative_A_noise':scale,'abs_detA':abs(det),'smin':sm,'condition':r['jacobian_condition_true'],
                                 'delta_detA':dt[k],'delta_mu_exact':exact[k],'delta_mu_linear':linear[k],
                                 'relative_linearization_error':abs(exact[k]-linear[k])/max(abs(exact[k]),1e-14),
                                 'parity_crossed':np.sign(dp[k])!=np.sign(det)})
    frame=pd.DataFrame(rows);frame.to_csv(a.out_dir/'propagation_event_level.csv',index=False)
    frame.groupby(['lens_family','relative_A_noise']).agg(n=('event_id','size'),median_linearization_error=('relative_linearization_error','median'),p90_linearization_error=('relative_linearization_error',lambda s:s.quantile(.9)),parity_crossing_fraction=('parity_crossed','mean')).reset_index().to_csv(a.out_dir/'propagation_summary.csv',index=False)
    (a.out_dir/'config.json').write_text(json.dumps(vars(a)|{'scope':'controlled true-Jacobian diagnostic; not a data-driven universal bound','formula':'d|mu| = -sign(detA)*|mu|^2*d(detA)'},indent=2,default=str))

if __name__=='__main__':main()

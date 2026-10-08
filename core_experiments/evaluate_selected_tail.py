#!/usr/bin/env python
"""Evaluate a NEW detection-selected low-y SIS catalogue using natural-population fits.
Run SIS_GW_physics_baseline.py with a fresh seed and .05 <= y <= .10 first.
The added catalogue is entirely test-only; no refitting/calibration on its labels.
"""
import os,sys,argparse,hashlib
from pathlib import Path
libdir=Path(sys.prefix)/'lib'
if (libdir/'libstdc++.so.6').exists() and os.environ.get('GW_REVIEW_LIB_READY')!='1':
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(libdir)+':'+env.get('LD_LIBRARY_PATH','');env['GW_REVIEW_LIB_READY']='1';os.execve(sys.executable,[sys.executable,*sys.argv],env)
p=argparse.ArgumentParser(description=__doc__)
ROOT=Path(__file__).resolve().parents[1]
p.add_argument('--tail-root',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
p.add_argument('--sis-root',type=Path,default=ROOT/'data_generation/data_lens_sis_gw_physics_baseline')
p.add_argument('--split-seeds',nargs='+',type=int,default=[11,23,37,42,59,71,89,101,131,173]);p.add_argument('--max-iter',type=int,default=180);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--threads',type=int,default=4)
a=p.parse_args()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]=str(a.threads)
from common import *
from threadpoolctl import threadpool_limits
original=json.loads((a.sis_root/'manifest.json').read_text());new=json.loads((a.tail_root/'manifest.json').read_text())
if new['generator']!=original['generator']:p.error('Different generator; this is not the same selected SIS population.')
conf=new['config'];base=original['config'];allowed={'seed','save_dir','target_events','max_attempts','overwrite','y_min','y_max'}
changed={k:(base.get(k),v) for k,v in conf.items() if k not in allowed and base.get(k)!=v}
if changed:p.error(f'Configuration differs beyond the low-y selection: {changed}')
if conf['seed']==base['seed']:p.error('Tail catalogue must use a new simulation seed.')
if conf['y_min']!=.05 or conf['y_max']!=.10:p.error('Expected .05 <= y <= .10 generation range.')
d,_=load_sis(a.sis_root);tail,_=load_sis(a.tail_root)
if len(tail)!=conf['target_events']:p.error('Tail generation incomplete: row count differs from requested accepted sample.')
if not tail.y_true.between(.05,.1).all():p.error('Tail catalogue includes events outside .05<=y<=.10.')
# Retained physical lens parameters identify accidentally reused realizations.
_,oldpr=load_sis(a.sis_root);_,newpr=load_sis(a.tail_root)
physical=['image_x_0','image_y_0','image_x_1','image_y_1','z_s_true','z_l_true','sigma_v_true']
if set(map(tuple,oldpr[physical].to_numpy()))&set(map(tuple,newpr[physical].to_numpy())):p.error('Tail data reuse original physical lens realizations.')
tail['identity_key']='new:'+tail.identity_key;tail['event_key']='new:'+tail.event_key
contracts=sis_contracts(d);rows=[];pieces=[]
a.out_dir.mkdir(parents=True,exist_ok=False)
with threadpool_limits(limits=a.threads):
 for seed in a.split_seeds:
  split=identity_split(d,seed);tr=split[split.split=='train'];ca=split[split.split=='calibration'];te=split[split.split=='test'];assert_disjoint(tr,ca,tail)
  for contract in ['geometry','geometry_followup']:
   fit=Fit(tr,ca,contracts[contract],6130,a.max_iter,1,True)
   for tag,test in [('natural_test',te),('new_selected_y_le_0p1',tail)]:
    pred=fit.predict(test);pred['sample']=tag;pred['split_seed']=seed;pred['contract']=contract;pieces.append(pred)
    rows.append(dict(sample=tag,split_seed=seed,contract=contract,**summarize(pred,a.bootstrap,seed)))
   print('Selected low-y evaluation:',seed,contract,flush=True)
 pd.DataFrame(rows).to_csv(a.out_dir/'metrics.csv',index=False);pd.concat(pieces).to_csv(a.out_dir/'predictions.csv',index=False)
 paths=[root/n for root in [a.sis_root,a.tail_root] for n in ['manifest.json','lens.csv','lens_params.csv','observable_features.csv']]
 save_json({'base':str(a.sis_root.resolve()),'targeted':str(a.tail_root.resolve()),'input_sha256':{str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},'tail_n':len(tail),'target':'Selected SIS population conditional on .05<=y<=.1, same selection gates; not an estimate of the natural incidence of this subset','fitting':'Only original natural-population train/calibration partitions; added low-y data test-only','uncertainty':'Cluster bootstrap over independent accepted lenses within each split; split variation separate','settings':vars(a)},a.out_dir/'design.json')

#!/usr/bin/env python
"""Reviewer follow-up experiments. Existing results and manuscript stay unchanged."""
import argparse,os,sys,json,hashlib,time,traceback
from pathlib import Path
# Select this Python environment's C++ runtime before numerical imports.
libdir=Path(sys.prefix)/'lib'
if (libdir/'libstdc++.so.6').exists() and os.environ.get('GW_REVIEW_LIB_READY')!='1':
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(libdir)+':'+env.get('LD_LIBRARY_PATH','');env['GW_REVIEW_LIB_READY']='1'
 os.execve(sys.executable,[sys.executable,*sys.argv],env)
p=argparse.ArgumentParser(description=__doc__)
ROOT=Path(__file__).resolve().parents[1]
p.add_argument('--stages',nargs='+',choices=['contracts','astrometry','matched','jacobian','tail','targeted','nfw_scale'],default=['contracts','astrometry','matched','jacobian','tail'])
p.add_argument('--out-dir',type=Path,required=True)
p.add_argument('--sis-root',type=Path,default=ROOT/'data_generation/data_lens_sis_gw_physics_baseline')
p.add_argument('--formal-root',type=Path,default=ROOT/'experiments/extended_experiments/production_MNRAS_experiments_v2')
p.add_argument('--generation-seeds',nargs='+',type=int,default=[20260922,20260923,20260924])
p.add_argument('--model-seeds',nargs='+',type=int,default=[6130,6131,6132])
p.add_argument('--split-seeds',nargs='+',type=int,default=[11,23,37,42,59,71,89,101,131,173])
p.add_argument('--bootstrap',type=int,default=1000)
p.add_argument('--max-iter',type=int,default=180)
p.add_argument('--ensemble-members',type=int,default=3)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--match-bins',type=int,default=4)
p.add_argument('--min-matched',type=int,default=30,help='Minimum events per family per partition; otherwise matching scenario is marked insufficient.')
p.add_argument('--image-sigmas',nargs='+',type=float,default=[.003,.01,.03,.1])
p.add_argument('--center-sigmas',nargs='+',type=float,default=[.005,.02,.05,.1])
p.add_argument('--center-biases',nargs='+',type=float,default=[.02,.05,.1])
p.add_argument('--target-lenses',type=int,default=240)
p.add_argument('--target-families',nargs='+',choices=['SIS','SIE','SIS_shear','NFW_like'],default=['SIS','SIE','SIS_shear','NFW_like'])
p.add_argument('--target-smin',type=float,default=.03)
p.add_argument('--max-tail-attempts',type=int,default=3000)
p.add_argument('--repeats',type=int,default=32)
p.add_argument('--resume',action='store_true')
p.add_argument('--smoke',action='store_true',help='Small functional check; scientific results must use a different output folder.')
a=p.parse_args()
if min(a.bootstrap,a.max_iter,a.ensemble_members,a.threads,a.target_lenses,a.max_tail_attempts,a.repeats)<1:p.error('Counts must be positive.')
if a.match_bins<1 or a.min_matched<10:p.error('match-bins >=1 and min-matched >=10 required.')
if any(v<0 for v in a.image_sigmas+a.center_sigmas+a.center_biases):p.error('Noise scales must be nonnegative.')
if a.smoke:
 a.generation_seeds=a.generation_seeds[:1];a.model_seeds=a.model_seeds[:1];a.split_seeds=a.split_seeds[:1];a.bootstrap=20;a.max_iter=8;a.ensemble_members=1;a.image_sigmas=[.01,.03];a.center_sigmas=[.02];a.center_biases=[.05];a.target_lenses=4;a.repeats=2
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]=str(a.threads)
# Runtime guard also covers libraries already loaded by embedded environments.
from threadpoolctl import threadpool_limits
from common import save_json
import controlled_inputs,astrometry,matched_transfer,diagnostics,targeted_tail,nfw_scale_control
runners={'contracts':controlled_inputs.run,'astrometry':astrometry.run,'matched':matched_transfer.run,'jacobian':diagnostics.jacobian,'tail':diagnostics.tail,'targeted':targeted_tail.run,'nfw_scale':nfw_scale_control.run}
a.out_dir=a.out_dir.resolve()
if a.out_dir.exists() and not a.resume:p.error('Output directory exists. Use a new folder or --resume with identical settings.')
a.out_dir.mkdir(parents=True,exist_ok=True)
config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
inputs={}
for root,names in [(a.sis_root,['lens.csv','lens_params.csv','observable_features.csv']),(a.formal_root,['run_config.json'])]:
 for n in names:
  f=root/n
  if not f.is_file():p.error(f'Missing input {f}')
  inputs[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
# Hash actual saved numerical inputs as well as their controller manifest.
for gen in a.generation_seeds:
 folder=a.formal_root/f'seed_{gen}'
 files=[]
 if set(a.stages)&{'matched','targeted','nfw_scale'}:
  for family in ['SIS','SIE','SIS_shear','NFW_like']:
   files += [folder/'benchmark'/family/n for n in ['observables.csv','event_latent.csv','image_latent.csv']]
  files.append(folder/'benchmark/manifest.json')
 if set(a.stages)&{'jacobian','tail'}:files.append(folder/'core_evaluation/family_matrix_predictions.csv')
 for f in files:
  if not f.is_file():p.error(f'Missing input {f}')
  inputs[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
if 'tail' in a.stages:
 for rel in ['outputs/near_critical_threshold_curves/near_critical_event_errors.csv','experiments/extended_experiments/production_repeated_observation_conditional_risk/conditional_risk_by_event.csv']:
  f=ROOT/rel;inputs[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
manifest={'settings':config,'code_sha256':code,'inputs_sha256':inputs}
path=a.out_dir/'run_manifest.json'
if path.exists():
 if json.loads(path.read_text())!=manifest:p.error('Resume configuration/code/input hashes differ. Use a new output directory.')
else:save_json(manifest,path)
with threadpool_limits(limits=a.threads):
 for stage in a.stages:
  out=a.out_dir/stage;out.mkdir(exist_ok=True);state=out/'status.json'
  if state.exists() and json.loads(state.read_text()).get('status')=='complete':print('Resume: completed',stage,flush=True);continue
  start=time.time();save_json({'status':'running','smoke':a.smoke},state)
  try:
   runners[stage](a,out)
   save_json({'status':'complete','smoke':a.smoke,'seconds':time.time()-start},state)
  except Exception as exc:
   save_json({'status':'failed','error':str(exc),'seconds':time.time()-start},state);raise
print(f'Completed requested stages. Results: {a.out_dir}',flush=True)

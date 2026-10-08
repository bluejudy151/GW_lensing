"""New held-out lens realizations concentrated on the critical tail.
Uses existing exact lens solver + independent finite-difference checks. No GW
waveforms or detector selection: this is a four-family geometry stress test.
"""
from collections import Counter
from types import SimpleNamespace
import numpy as np
import pandas as pd
from common import *
import generate_universal_latent_benchmark as generator
from mnras_noise_variants import observe, BASE

def prior_table(a):
 from astropy.cosmology import Planck18
 from astropy.constants import c
 import astropy.units as u
 prior,_=generator.load_sis_prior(a.sis_root)
 split=generator.group_partition(prior,7619);prior['partition']=prior.event_id.map(split)
 prior['source_luminosity_distance_true_Mpc']=Planck18.luminosity_distance(prior.z_s_true.to_numpy()).value
 dl=Planck18.angular_diameter_distance(prior.z_l_true.to_numpy());ds=Planck18.angular_diameter_distance(prior.z_s_true.to_numpy());dls=Planck18.angular_diameter_distance_z1z2(prior.z_l_true.to_numpy(),prior.z_s_true.to_numpy())
 prior['time_delay_factor_s_per_arcsec2']=((1+prior.z_l_true.to_numpy())*dl*ds/dls/c).to_value(u.s)*(np.pi/(180*3600))**2
 return prior[prior.partition=='test'].copy()

def generate(a,gen,fam,prior,destination):
 dest=destination/fam;dest.mkdir(exist_ok=True)
 if (dest/'complete.json').exists():
  return pd.read_csv(dest/'repeated_observations.csv'),pd.read_csv(dest/'event_latent.csv')
 if a.target_lenses>len(prior):raise ValueError(f'--target-lenses cannot exceed {len(prior)} reserved test prior identities; use more independent prior rows, not duplicated noise copies.')
 config=json.loads((a.formal_root/f'seed_{gen}/benchmark/manifest.json').read_text())['args'];settings=SimpleNamespace(**config)
 physical_rng=np.random.default_rng(stable_seed(gen,'targeted',fam));obs_rng=np.random.default_rng(stable_seed(gen,'repeat',fam))
 chosen=prior.sample(n=a.target_lenses,random_state=stable_seed(gen,fam,'prior'));events=[];images=[];repeated=[];reject=Counter()
 for i,(_,pr) in enumerate(chosen.iterrows()):
  for attempt in range(a.max_tail_attempts):
   theta=float(pr.theta_E_true);y=float(physical_rng.uniform(.05,.1 if fam=='SIS' else .30));phi=float(physical_rng.uniform(0,2*np.pi))
   model,kw,meta=generator.model_for(fam,theta,physical_rng,settings)
   eid=900000000+1000000*FAMILIES.index(fam)+i
   rows,reason=generator.rows_for(model,kw,fam,eid,theta,y,phi,meta,settings)
   if reason:reject[reason]+=1;continue
   primary,secondary=sorted(rows,key=lambda r:-r['mu_abs_true'])[:2]
   if fam!='SIS' and primary['singular_value_min_true']>=a.target_smin:reject['outside_target_smin']+=1;continue
   minimum=min(r['fermat_potential_true'] for r in rows)
   for r in rows:r['time_delay_true_s']=(r['fermat_potential_true']-minimum)*float(pr.time_delay_factor_s_per_arcsec2)
   event=generator.observed_event(rows,pr,eid,obs_rng,settings)
   event.update(detA_true=primary['detA_true'],singular_value_min_true=primary['singular_value_min_true'],jacobian_condition_true=primary['jacobian_condition_true'],identity_key=f'prior:{int(pr.event_id)}',event_key=f'targeted:{gen}:{fam}:{i}',split='test',generation_seed=gen,attempts_for_lens=attempt+1)
   events.append(event)
   for r in rows:images.append(dict(r,identity_key=event['identity_key'],event_key=event['event_key'],is_primary=r['image_id']==primary['image_id'],is_secondary=r['image_id']==secondary['image_id']))
   for repeat in range(a.repeats):
    measurements=observe(SimpleNamespace(**event),SimpleNamespace(**primary),SimpleNamespace(**secondary),BASE,obs_rng.standard_normal(10))
    repeated.append({**event,**measurements,'repeat_id':repeat})
   break
  else:
   pd.DataFrame(events).to_csv(dest/'partial_event_latent.csv',index=False)
   save_json({'status':'failed_quota','prior_id':int(pr.event_id),'rejections':dict(reject),'completed':len(events),'max_attempts_per_prior':a.max_tail_attempts},dest/'failure.json')
   raise RuntimeError(f'{fam}: failed to obtain a critical lens for reserved prior {pr.event_id}; partial records saved, no silent replacement.')
  if i%20==0 or i==a.target_lenses-1:print(f'targeted {gen} {fam}: {i+1}/{a.target_lenses} independent prior identities',flush=True)
 e=pd.DataFrame(events);r=pd.DataFrame(repeated);im=pd.DataFrame(images)
 e.to_csv(dest/'event_latent.csv',index=False);r.to_csv(dest/'repeated_observations.csv',index=False);im.to_csv(dest/'image_latent.csv',index=False)
 save_json({'status':'complete','lenses':len(e),'unique_prior_identities':e.identity_key.nunique(),'repeats_per_lens':a.repeats,'rejections':dict(reject),'selection':'SIS .05<=y<.1; other families .05<=y<.30 and primary smin below threshold','target_smin':a.target_smin,'population':'Geometry stress test; no SNR/delay selection; not a new selected SIS catalogue','targeting_uses_truth':True,'finite_difference_validation':True},dest/'complete.json')
 return r,e

def run(a,out):
 prior=prior_table(a);results=[];predictions=[]
 for gen in a.generation_seeds:
  folder=out/f'seed_{gen}';folder.mkdir(exist_ok=True);d=load_benchmark(a.formal_root/f'seed_{gen}/benchmark')
  tr=d[d.split=='train'];cal=d[d.split=='calibration']
  tests=[]
  for fam in a.target_families:
   r,e=generate(a,gen,fam,prior,folder);assert_disjoint(tr,cal,r);tests.append(r)
  test=pd.concat(tests,ignore_index=True)
  for seed in a.model_seeds:
   fit=Fit(tr,cal,GEOMETRY,seed,a.max_iter,a.ensemble_members);p=fit.predict(test);p['repeat_id']=test.repeat_id.to_numpy();p['model_seed']=seed;p['generation_seed']=gen;predictions.append(p)
  pooled=pd.concat([p for p in predictions if p.generation_seed.iloc[0]==gen])
  for fam,part in pooled.groupby('lens_family'):
   results.append(dict(generation_seed=gen,family=fam,subset='targeted',**summarize(part,a.bootstrap,stable_seed(gen,fam))))
  pd.DataFrame(results).to_csv(out/'tail_metrics.csv',index=False)
  pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
 save_json({'training':'Original benchmark training identities only; targeted lenses are test-only','calibration':'Original benchmark held-out calibration identities, never targeted test labels','independence':'One new lens per reserved test prior identity per family/generation; repeated observations and model seeds averaged before identity bootstrap; generations reported separately','selection':'Deliberate critical-tail proposal, not sampling from original selected population','not_done':'No new waveforms, no reapplication of original detection gates, no estimate of population tail incidence','default_objective':'240 distinct held-out prior identities per family per generation, each with 32 observations'},out/'interpretation.json')

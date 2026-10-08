"""Offline common-support controls: magnification and observed image separation.
Truth is allowed only to define diagnostic population samples; never a predictor.
"""
import numpy as np
import pandas as pd
from common import *

def matched_sample(d,mode,seed,nbins=4):
 if mode=='original':return d.copy(),pd.DataFrame(),{'status':'original','uses_test_truth_for_diagnostic_sampling':False}
 columns=['mu0_true']+(['obs__image_separation_observed'] if mode=='mu_separation' else [])
 tr=d[d.split=='train'];edges=[]
 for c in columns:
  # Bounds and bin edges are determined from TRAINING data only.
  lo=max(tr.loc[tr.lens_family==f,c].quantile(.01) for f in FAMILIES)
  hi=min(tr.loc[tr.lens_family==f,c].quantile(.99) for f in FAMILIES)
  if not 0<lo<hi:return d.iloc[:0],pd.DataFrame(),{'status':'no_training_support','column':c}
  edges.append(np.linspace(np.log(lo),np.log(hi),nbins+1))
 bins=[];valid=np.ones(len(d),bool)
 for c,e in zip(columns,edges):
  v=np.log(np.maximum(d[c].to_numpy(),1e-15));valid&=(v>=e[0])&(v<=e[-1]);bins.append(np.minimum(np.searchsorted(e,v,side='right')-1,len(e)-2))
 cells=np.stack(bins,axis=1);temp=d.copy();temp['match_cell']=[':'.join(map(str,row)) for row in cells];temp=temp.loc[valid].copy()
 counts=temp[temp.split=='train'].groupby(['match_cell','lens_family']).size().unstack(fill_value=0).reindex(columns=FAMILIES,fill_value=0)
 support=set(counts.index[(counts>=5).all(axis=1)]);temp=temp[temp.match_cell.isin(support)]
 selected=[];auditrows=[]
 for (split,cell),g in temp.groupby(['split','match_cell']):
  n=min((g.lens_family==f).sum() for f in FAMILIES)
  for f in FAMILIES:
   part=g[g.lens_family==f];auditrows.append(dict(split=split,match_cell=cell,lens_family=f,available=len(part),kept=n))
   if n:selected.append(part.sample(n=n,random_state=stable_seed(seed,split,cell,f)))
 result=pd.concat(selected).sort_index() if selected else d.iloc[:0].copy()
 info={'status':'matched','columns':columns,'log_edges':[v.tolist() for v in edges],'training_common_cells':sorted(support),'sampling':'Equal counts per family within each bin and split, no replacement','uses_test_truth_for_diagnostic_sampling':True,'limitation':'Only finite-bin distribution matching on common support; not full causal isolation or deployment selection'}
 return result,pd.DataFrame(auditrows),info

def train_subset(tr,omit,seed):
 groups=tr[tr.lens_family!='NFW_like'] if omit else tr
 # Same training budget in paired mixed/leave-out fits, independent of test data.
 n=min(len(tr[tr.lens_family!='NFW_like']),len(tr))
 if 'match_cell' in tr and tr.match_cell.notna().all():
  pieces=[]
  for cell,g in groups.groupby('match_cell'):
   target=len(tr[(tr.match_cell==cell)&(tr.lens_family!='NFW_like')]);pieces.append(g.sample(n=min(target,len(g)),random_state=stable_seed(seed,cell)))
  return pd.concat(pieces)
 return groups.sample(n=n,random_state=seed)

def run(a,out,loader=None):
 metrics=[];preds=[];designs={};memberships=[];support=[];balance=[]
 for gen in a.generation_seeds:
  d=loader(a,gen,out) if loader else load_benchmark(a.formal_root/f'seed_{gen}'/'benchmark')
  d,normcols=normalized_geometry(d)
  for mode in ['original','mu','mu_separation']:
   matched,count,info=matched_sample(d,mode,stable_seed(gen,mode),a.match_bins);designs[f'{gen}:{mode}']=info
   if len(count):support.append(count.assign(generation_seed=gen,matching=mode))
   if len(matched):
    memberships.append(matched[['event_key','identity_key','lens_family','split','mu0_true','obs__image_separation_observed']].assign(generation_seed=gen,matching=mode))
    for (sp,fam),part in matched.groupby(['split','lens_family']):
     balance.append(dict(generation_seed=gen,matching=mode,split=sp,family=fam,n=len(part),n_identities=part.identity_key.nunique(),mu_median=part.mu0_true.median(),separation_median=part.obs__image_separation_observed.median()))
   sufficient=len(matched)>0 and all(len(matched[(matched.split==s)&(matched.lens_family==f)])>=a.min_matched for s in ['train','calibration','test'] for f in FAMILIES)
   if not sufficient:
    designs[f'{gen}:{mode}']['evaluation_status']='insufficient_common_support';print(f'{gen} {mode}: insufficient common support; recorded, not replaced',flush=True);continue
   for contract,cols in [('angular_geometry',GEOMETRY),('normalized_geometry',normcols)]:
    for modelseed in a.model_seeds:
     for omitted in [False,True]:
      tr=train_subset(matched[matched.split=='train'],omitted,modelseed)
      cal=train_subset(matched[matched.split=='calibration'],omitted,modelseed+10000)
      te=matched[(matched.split=='test')&(matched.lens_family=='NFW_like')];assert_disjoint(tr,cal,te)
      protocol='leave_NFW_out' if omitted else 'mixed';print(f'transfer {gen} {mode} {contract} {modelseed} {protocol}',flush=True)
      fit=Fit(tr,cal,cols,modelseed,a.max_iter,a.ensemble_members);p=fit.predict(te)
      meta=dict(generation_seed=gen,matching=mode,contract=contract,model_seed=modelseed,protocol=protocol)
      preds.append(p.assign(**meta));metrics.append(dict(**meta,train_n=len(tr),calibration_n=len(cal),calibration_identities=fit.calibration_identities,conformal_q=fit.q,**summarize(p,a.bootstrap,modelseed)))
      pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
 save_json(designs,out/'matching_design.json')
 if support:pd.concat(support).to_csv(out/'support_counts.csv',index=False)
 if memberships:pd.concat(memberships).to_csv(out/'matching_membership.csv',index=False)
 pd.DataFrame(balance).to_csv(out/'balance.csv',index=False)
 if preds:
  p=pd.concat(preds);p.to_csv(out/'predictions.csv',index=False);paired=[]
  for keys,g in p.groupby(['generation_seed','matching','contract','model_seed']):
   left=g[g.protocol=='leave_NFW_out'];right=g[g.protocol=='mixed'];z=left.merge(right,on=['event_key','identity_key'],suffixes=('_loo','_mixed'),validate='one_to_one')
   for metric in ['abs_error','covered','relative_error','log_error','width']:z[metric]=z[metric+'_loo']-z[metric+'_mixed']
   paired.append(dict(zip(['generation_seed','matching','contract','model_seed'],keys),**summarize(z[['event_key','identity_key','abs_error','covered','relative_error','log_error','width']],a.bootstrap,stable_seed(*keys),coverage_bounds=False)))
  pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
 save_json({'features':{'angular_geometry':GEOMETRY,'normalized_geometry':normcols},'no_NFW_reference_angle_or_sigma_v_inputs':True,'training_budget':'Same total number of training events per mixed/leave-out pair; after matching also same per-cell budget','calibration':'Equal total calibration event budget per mixed/leave-out pair; held-out identities in available training families only; NFW calibration never used in leave-out','coverage':'Diagnostic empirical coverage, no guarantee under family shift or truth-based matched selection','difference_sign':'leave-out minus mixed','population_control':'Matching does not remove all profile, selection or measurement-noise differences'},out/'interpretation.json')

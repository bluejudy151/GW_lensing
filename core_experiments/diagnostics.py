"""Compare true determinant, singular value and condition; cluster-aware tail intervals."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from common import *

def relation(x,y,mu,clusters,bootstrap,seed):
 x=np.log(np.maximum(np.asarray(x,float),1e-14));y=np.log(np.maximum(np.asarray(y,float),1e-14));mu=np.log(np.asarray(mu,float))
 def stat(ix):
  xx=x[ix];yy=y[ix]
  slope=float(np.polyfit(xx,yy,1)[0]) if np.ptp(xx)>1e-10 else np.nan
  rho=float(spearmanr(xx,yy).statistic) if np.ptp(xx)>1e-10 and np.ptp(yy)>1e-10 else np.nan
  return slope,rho
 whole=np.arange(len(x));slope,rho=stat(whole)
 # Partial Spearman controlling ranked true magnification. Determinant is
 # algebraically redundant with magnification: label singular cases explicitly.
 design=np.column_stack([np.ones(len(mu)),rankdata(mu)]);rx=rankdata(x);ry=rankdata(y)
 ux=rx-design@np.linalg.lstsq(design,rx,rcond=None)[0];uy=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
 eligible=np.std(ux)>1e-6*max(np.std(rx),1) and np.std(uy)>1e-6*max(np.std(ry),1)
 partial=float(np.corrcoef(ux,uy)[0,1]) if eligible else np.nan
 groups=np.asarray(clusters);unique=np.unique(groups);ixmap={v:np.flatnonzero(groups==v) for v in unique};rng=np.random.default_rng(seed);vals=[]
 for _ in range(bootstrap):vals.append(stat(np.concatenate([ixmap[v] for v in rng.choice(unique,len(unique),replace=True)])))
 vals=np.array(vals);lo=np.nanquantile(vals,.025,axis=0);hi=np.nanquantile(vals,.975,axis=0)
 return dict(n_events=len(x),n_identities=len(unique),slope=slope,slope_lo95=lo[0],slope_hi95=hi[0],spearman=rho,spearman_lo95=lo[1],spearman_hi95=hi[1],partial_spearman_given_logmu=partial,partial_status='estimated' if eligible else 'not_identifiable_collinear_or_constant')

def saved_core(a,gen):
 path=a.formal_root/f'seed_{gen}/core_evaluation/family_matrix_predictions.csv'
 d=pd.read_csv(path);d=d[(d.contract=='geometry')&(d.protocol=='mixed_family')&(d.domain=='benchmark')]
 d['bias']=d.mu0_pred-d.mu0_true;d['relative_error']=d.abs_error/d.mu0_true;d['log_error']=abs(np.log(d.mu0_pred/d.mu0_true));d['covered']=d.conformal_covered;d['width']=d.conformal_width
 return d

def jacobian(a,out):
 rows=[];bins=[];controlled=[];evs=[]
 for gen in a.generation_seeds:
  p=saved_core(a,gen)
  cols=['mu0_true','abs_detA','smin','condition','abs_error','relative_error','log_error','covered','width']
  e=p.groupby(['test_family','identity_key','event_key'],as_index=False)[cols].mean();evs.append(e.assign(generation_seed=gen))
  for fam,g in e.groupby('test_family'):
   print(f'Jacobian {gen} {fam}',flush=True)
   for axis in ['abs_detA','smin','condition']:
    for response in ['abs_error','relative_error','log_error']:
     rows.append(dict(generation_seed=gen,family=fam,axis=axis,response=response,**relation(g[axis],g[response],g.mu0_true,g.identity_key,a.bootstrap,stable_seed(gen,fam,axis,response))))
    labels=pd.qcut(g[axis],min(12,g[axis].nunique()),duplicates='drop')
    for interval,part in g.groupby(labels,observed=True):bins.append(dict(generation_seed=gen,family=fam,axis=axis,bin=str(interval),n=len(part),n_identities=part.identity_key.nunique(),x_median=part[axis].median(),mae=part.abs_error.mean(),median_error=part.abs_error.median(),mean_relative_error=part.relative_error.mean()))
   # Narrow true-magnification strata are diagnostic; no fit, threshold or
   # uncertainty calibration uses these test labels.
   for label,part in g.groupby(pd.qcut(g.mu0_true,5,duplicates='drop'),observed=True):
    for axis in ['smin','condition']:
     controlled.append(dict(generation_seed=gen,family=fam,mu_bin=str(label),axis=axis,**relation(part[axis],part.relative_error,part.mu0_true,part.identity_key,min(a.bootstrap,300),stable_seed(gen,fam,label,axis))))
 pd.DataFrame(rows).to_csv(out/'axis_comparison.csv',index=False);pd.DataFrame(bins).to_csv(out/'axis_bins.csv',index=False);pd.DataFrame(controlled).to_csv(out/'within_magnification_bins.csv',index=False);pd.concat(evs).to_csv(out/'model_seed_averaged_events.csv',index=False)
 save_json({'prediction_source':'Saved mixed-family geometry predictions; losses averaged over model seeds per event, generation catalogues kept separate','variables':['abs_detA','smin','condition'],'responses':['absolute error','relative absolute error','absolute log error'],'partial_correlation':'Partial rank correlation controlling rank(log true mu); collinear variables reported as not identifiable','scope':'Associations in this simulated population, not independent causal effects or a universal error exponent'},out/'interpretation.json')
 # Small directly reviewable overview.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(10,3.2));allbins=pd.DataFrame(bins)
 for ax,axis in zip(axes,['abs_detA','smin','condition']):
  for fam in FAMILIES:
   g=allbins[(allbins.axis==axis)&(allbins.family==fam)]
   ax.scatter(g.x_median,g.mean_relative_error,s=12,label=fam,alpha=.55)
  ax.set(xscale='log',yscale='log',xlabel=axis,ylabel='Mean relative error')
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'conditioning_comparison.pdf');plt.close(fig)

def tail(a,out):
 rows=[]
 for gen in a.generation_seeds:
  p=saved_core(a,gen)
  for fam,g in p.groupby('test_family'):
   for label,mask in [('all',np.ones(len(g),bool)),('smin_lt_0p03',g.smin<.03),('smin_lt_0p1',g.smin<.1)]:
    rows.append(dict(experiment='formal_four_family',generation_seed=gen,family=fam,subset=label,**summarize(g[mask],a.bootstrap,stable_seed(gen,fam,label))))
 e=pd.read_csv(ROOT/'outputs/near_critical_threshold_curves/near_critical_event_errors.csv')
 e['event_key']=e.event_id.astype(str) if 'event_id' in e else e.index.astype(str);e['identity_key']=e.event_key;e['bias']=e.mu0_pred-e.mu0_true;e['abs_error']=abs(e.bias);e['relative_error']=e.abs_error/e.mu0_true
 for label,mask in [('all',np.ones(len(e),bool)),('y_le_0p2',e.y_true<=.2),('y_le_0p1',e.y_true<=.1)]:rows.append(dict(experiment='selected_SIS_fixed',family='SIS',subset=label,**summarize(e[mask],a.bootstrap,stable_seed(label))))
 e=pd.read_csv(ROOT/'experiments/extended_experiments/production_repeated_observation_conditional_risk/conditional_risk_by_event.csv')
 e['event_key']=e.latent_event_id.astype(str);e['identity_key']=e.event_key;e['abs_error']=e.mc_mae;e['covered']=e.interval_90_coverage
 for label,mask in [('all',np.ones(len(e),bool)),('y_le_0p2',e.y_true<=.2),('y_le_0p1',e.y_true<=.1)]:rows.append(dict(experiment='repeated_SIS_latent_lens',family='SIS',subset=label,**summarize(e[mask],a.bootstrap,stable_seed('repeat',label))))
 pd.DataFrame(rows).to_csv(out/'tail_cluster_intervals.csv',index=False)
 save_json({'resampling':'Independent latent lenses / reused-prior identity clusters; never noise-realization rows or model seeds','sparse_rule':'Fewer than 10 identities labelled sparse_diagnostic; bootstrap intervals with >=2 clusters describe only resampling of observed clusters; singleton intervals missing','coverage':'Percentile bootstrap can collapse to [0,0] for all failures; additionally report a conservative Hoeffding interval for equally weighted independent-cluster mean coverage, valid only under independent bounded clusters','no_population_extrapolation':'Three latent lenses with 32 noise copies each are three clusters, not 96 independent lenses'},out/'interpretation.json')

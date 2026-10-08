"""Shared, explicit contracts and identity-aware evaluation for the review follow-up."""
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, sys, importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/extended_experiments'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from evaluate_universal_identifiability import GEOMETRY, FOLLOWUP, FAMILIES, load_benchmark, conformal_quantile
from experiments.mu0_extension_tools import WAVE_PROXY_COLS


def save_json(obj,path):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False))


def stable_seed(*parts):return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:4],'little')

def audit(cols,frame):
 bad=[c for c in cols if not c.startswith(('obs__','norm__')) or any(s in c.lower() for s in ['true','latent','source_luminosity_distance','snr_pair','snr_1','snr_2','mu0','mu1','detA','jacobian'])]
 if bad:raise ValueError(f'Forbidden model input: {bad}')
 missing=set(cols)-set(frame)
 if missing:raise ValueError(f'Missing required features: {sorted(missing)}')
 if frame[cols].isna().all().any():raise ValueError('Entirely missing input column')
 return list(dict.fromkeys(cols))


def load_sis(root):
 root=Path(root);lens=pd.read_csv(root/'lens.csv');obs=pd.read_csv(root/'observable_features.csv');params=pd.read_csv(root/'lens_params.csv')
 for d in [lens,obs,params]:
  if d.event_id.duplicated().any():raise ValueError('Nonunique event identity')
 if set(lens.event_id)!=set(obs.event_id) or set(lens.event_id)!=set(params.event_id):raise ValueError('SIS event ID sets differ')
 frame=lens[['event_id','mu_0','y']].rename(columns={'mu_0':'mu0_true','y':'y_true'}).merge(obs.rename(columns={c:'obs__'+c for c in obs if c!='event_id'}),on='event_id',validate='one_to_one')
 frame['identity_key']=frame.event_id.astype(str);frame['event_key']='SIS:'+frame.identity_key;frame['lens_family']='SIS'
 params=params.set_index('event_id').loc[frame.event_id].reset_index()
 return frame,params


def sis_contracts(d):
 follow=FOLLOWUP+['obs__log1p_z_s_observed','obs__log1p_z_l_observed']
 g=GEOMETRY;scale=['obs__theta_E_observed']
 return {name:audit(c,d) for name,c in {'gw_proxies':WAVE_PROXY_COLS,'gw_followup':WAVE_PROXY_COLS+follow,'geometry':g,'geometry_followup':g+follow+scale,'full_observed':WAVE_PROXY_COLS+follow+g+scale}.items()}


def identity_split(d,seed):
 ids=np.random.default_rng(seed).permutation(sorted(d.identity_key.unique()))
 a=int(.7*len(ids));b=int(.85*len(ids))
 mapping={v:('train' if i<a else 'calibration' if i<b else 'test') for i,v in enumerate(ids)}
 out=d.copy();out['split']=out.identity_key.map(mapping);return out


def assert_disjoint(train,cal,test):
 sets=[set(x.identity_key) for x in [train,cal,test]]
 if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Identity leakage across partitions')


class Fit:
 """HistGB log-target ensemble; preprocessing and calibration use separate partitions."""
 def __init__(self,train,cal,cols,seed,max_iter=180,members=3,sis=False):
  self.cols=audit(cols,train);self.sis=sis;self.models=[]
  if set(train.identity_key)&set(cal.identity_key):raise ValueError('Train/calibration identities overlap')
  tau=np.log(train.mu0_true.to_numpy()-1) if sis else np.log(train.mu0_true.to_numpy())
  if not np.isfinite(tau).all():raise ValueError('Nonfinite target transform')
  groups=train.identity_key.to_numpy();unique=np.unique(groups);index={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(seed)
  for j in range(members):
   ix=np.arange(len(train)) if members==1 else np.concatenate([index[g] for g in rng.choice(unique,len(unique),replace=True)])
   model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingRegressor(max_iter=max_iter,max_leaf_nodes=15,min_samples_leaf=10,learning_rate=.06,l2_regularization=.1,early_stopping=False,random_state=seed+j))
   model.fit(train.iloc[ix][cols],tau[ix]);self.models.append(model)
  raw=self.raw(cal);scores=np.maximum.reduce([raw.lo-cal.mu0_true.to_numpy(),cal.mu0_true.to_numpy()-raw.hi,np.zeros(len(cal))])
  cluster=pd.DataFrame({'id':cal.identity_key.to_numpy(),'score':scores}).groupby('id').score.max()
  self.q=conformal_quantile(cluster);self.calibration_identities=len(cluster)
  if not np.isfinite(self.q):raise ValueError('Too few calibration identities for finite 90% conformal interval')
 def raw(self,d):
  v=np.exp(np.clip(np.column_stack([m.predict(d[self.cols]) for m in self.models]),-30,30))+(1 if self.sis else 0)
  return SimpleNamespace(pred=np.median(v,axis=1),lo=np.quantile(v,.05,axis=1),hi=np.quantile(v,.95,axis=1))
 def predict(self,d):
  r=self.raw(d);cols=[c for c in ['event_id','event_key','identity_key','lens_family','mu0_true','y_true','detA_true','singular_value_min_true','jacobian_condition_true'] if c in d]
  out=d[cols].reset_index(drop=True).copy();out['mu0_pred']=r.pred;out['lo']=np.maximum(0,r.lo-self.q);out['hi']=r.hi+self.q
  out['abs_error']=abs(out.mu0_pred-out.mu0_true);out['relative_error']=out.abs_error/out.mu0_true
  out['log_error']=abs(np.log(out.mu0_pred/out.mu0_true));out['bias']=out.mu0_pred-out.mu0_true
  out['covered']=((out.mu0_true>=out.lo)&(out.mu0_true<=out.hi)).astype(float);out['width']=out.hi-out.lo
  return out


def summarize(d,bootstrap=1000,seed=6130,cluster='identity_key',coverage_bounds=True):
 """Event-weighted means with resampling of independent identity clusters."""
 if len(d)==0:return {'n_rows':0,'n_events':0,'n_identities':0,'status':'empty'}
 measures=[c for c in ['abs_error','bias','relative_error','log_error','covered','width'] if c in d]
 # Average losses over repeated observations/model fits, never count fits as new lenses.
 e=d.groupby([cluster,'event_key'],as_index=False)[measures].mean()
 groups=e.groupby(cluster)[measures];s=groups.sum().to_numpy();n=groups.count().to_numpy();rng=np.random.default_rng(seed)
 result={'n_rows':len(d),'n_events':len(e),'n_identities':len(s),'status':'ok' if len(s)>=10 else 'sparse_diagnostic'}
 boots=[]
 if len(s)>=2:
  for _ in range(bootstrap):
   ix=rng.integers(0,len(s),len(s));boots.append(s[ix].sum(axis=0)/n[ix].sum(axis=0))
 for j,c in enumerate(measures):
  name={'abs_error':'mae','covered':'coverage90','width':'mean_width'}.get(c,c)
  result[name]=float(e[c].mean())
  result[name+'_lo95']=float(np.quantile(np.array(boots)[:,j],.025)) if boots else np.nan
  result[name+'_hi95']=float(np.quantile(np.array(boots)[:,j],.975)) if boots else np.nan
 if 'covered' in measures and coverage_bounds:
  # Bounded per-cluster coverage, not binomial trials at repeated-row level.
  p=float(groups.mean().covered.mean());delta=np.sqrt(np.log(40)/(2*len(s)))
  result.update(equal_cluster_coverage90=p,coverage_hoeffding_lo95=max(0,p-delta),coverage_hoeffding_hi95=min(1,p+delta))
 return result


def normalized_geometry(d):
 out=d.copy();sep=np.maximum(d['obs__image_separation_observed'].to_numpy(),1e-6);cols=[]
 for ax in ['x','y']:
  for i in [0,1]:
   name=f'norm__image_{ax}_{i}';out[name]=(d[f'obs__image_{ax}_{i}_observed']-d[f'obs__lens_center_{ax}_observed'])/sep;cols.append(name)
 for s in ['theta_plus_abs','theta_minus_abs']:
  name='norm__'+s;out[name]=d[f'obs__{s}_observed']/sep;cols.append(name)
 out['norm__asymmetry']=d['obs__image_position_asymmetry_observed'];cols.append('norm__asymmetry')
 return out,cols

"""Same estimator, features audited, same latent split for every SIS contract."""
import numpy as np
import pandas as pd
from common import *

def run(a,out):
 d,_=load_sis(a.sis_root);contracts=sis_contracts(d)
 save_json({'features':contracts,'target':'log(mu0-1)','splits':'70/15/15 by event identity','estimator':'identical HistGB settings; no raw waveform inference','forbidden':'simulation distance, injected SNR and all true/latent fields','new_geometry_followup_features':len(contracts['geometry_followup'])},out/'contracts.json')
 rows=[];preds=[];pairs=[];splits=[]
 for splitseed in a.split_seeds:
  split=identity_split(d,splitseed);tr=split[split.split=='train'];cal=split[split.split=='calibration'];te=split[split.split=='test'];assert_disjoint(tr,cal,te)
  splits.append(split[['event_id','identity_key','split']].assign(split_seed=splitseed))
  for name,cols in contracts.items():
   print(f'contracts split={splitseed} {name}',flush=True)
   fit=Fit(tr,cal,cols,a.model_seeds[0],a.max_iter,1,True);p=fit.predict(te);p['contract']=name;p['split_seed']=splitseed;preds.append(p)
   for subset,m in [('all',np.ones(len(p),bool)),('y_le_0p1',p.y_true<=.1),('y_le_0p2',p.y_true<=.2)]:
    rows.append(dict(split_seed=splitseed,contract=name,subset=subset,n_features=len(cols),**summarize(p[m],a.bootstrap,splitseed)))
  pp=pd.concat([x for x in preds if x.split_seed.iloc[0]==splitseed]);base=pp[pp.contract=='geometry'][['event_key','abs_error']].rename(columns={'abs_error':'reference_error'})
  for name,g in pp.groupby('contract'):
   z=g.merge(base,on='event_key',validate='one_to_one');z['abs_error']=z.abs_error-z.reference_error
   # Negative delta means the named contract has lower error than geometry.
   pairs.append(dict(split_seed=splitseed,contract=name,reference='geometry',**summarize(z[['event_key','identity_key','abs_error']],a.bootstrap,splitseed)))
 pd.concat(preds).to_csv(out/'predictions.csv',index=False);pd.concat(splits).to_csv(out/'split_manifest.csv',index=False)
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(pairs).rename(columns={'mae':'mae_difference','mae_lo95':'difference_lo95','mae_hi95':'difference_hi95'}).to_csv(out/'paired_differences.csv',index=False)
 pd.DataFrame(rows).query("subset=='all'").groupby('contract').agg(mae_mean=('mae','mean'),mae_split_sd=('mae','std'),n_splits=('mae','count')).to_csv(out/'split_summary.csv')

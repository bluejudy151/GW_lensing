"""Astrometry/centroid sensitivity on fixed selected SIS identities.
The noise grid is a scenario analysis, not a claim about telescope performance.
"""
import numpy as np
import pandas as pd
from common import *

def observe_sis(base,ref,z,sigma_image,sigma_center,bias):
 out=base.copy();n=len(base)
 x0=ref.image_x_0.to_numpy()+sigma_image*z[:,0];y0=ref.image_y_0.to_numpy()+sigma_image*z[:,1]
 x1=ref.image_x_1.to_numpy()+sigma_image*z[:,2];y1=ref.image_y_1.to_numpy()+sigma_image*z[:,3]
 angle=np.arctan2(ref.image_y_0-ref.lens_center_y,ref.image_x_0-ref.lens_center_x)
 cx=ref.lens_center_x.to_numpy()+sigma_center*z[:,4]+bias*np.cos(angle)
 cy=ref.lens_center_y.to_numpy()+sigma_center*z[:,5]+bias*np.sin(angle)
 plus=np.hypot(x0-cx,y0-cy);minus=np.hypot(x1-cx,y1-cy);sep=plus+minus
 # Keep the existing selected SIS definition: sum of radii, clipped asymmetry.
 fields={'image_x_0_observed':x0,'image_y_0_observed':y0,'image_x_1_observed':x1,'image_y_1_observed':y1,'lens_center_x_observed':cx,'lens_center_y_observed':cy,'theta_plus_abs_observed':plus,'theta_minus_abs_observed':minus,'image_separation_observed':sep,'theta_E_observed':sep/2,'image_position_asymmetry_observed':np.clip((plus-minus)/np.maximum(sep,1e-8),1e-4,.999)}
 for c,v in fields.items():out['obs__'+c]=v
 return out

def run(a,out):
 d,ref=load_sis(a.sis_root);contracts=sis_contracts(d)
 settings=[(i,c,0.) for i in a.image_sigmas for c in a.center_sigmas]
 settings+= [(0.01,0.02,b) for b in a.center_biases if b>0]
 z=np.random.default_rng(81006).standard_normal((len(d),6));rows=[];pieces=[]
 for sp in a.split_seeds:
  original=identity_split(d,sp);baseline=observe_sis(original,ref,z,.01,.02,0)
  train=baseline[baseline.split=='train'];cal=baseline[baseline.split=='calibration'];test=baseline[baseline.split=='test'];assert_disjoint(train,cal,test)
  fits={name:Fit(train,cal,contracts[name],a.model_seeds[0],a.max_iter,1,True) for name in ['geometry','geometry_followup','gw_followup']}
  for im,ctr,bias in settings:
   frame=observe_sis(original,ref,z,im,ctr,bias);tr=frame[frame.split=='train'];ca=frame[frame.split=='calibration'];te=frame[frame.split=='test']
   for name in ['geometry','geometry_followup','gw_followup']:
    print(f'astrometry split={sp} image={im} center={ctr} bias={bias} {name}',flush=True)
    for mode in ['frozen_nominal','matched_retrain']:
     fit=fits[name] if mode=='frozen_nominal' or name=='gw_followup' else Fit(tr,ca,contracts[name],a.model_seeds[0],a.max_iter,1,True)
     p=fit.predict(te);meta=dict(split_seed=sp,sigma_image=im,sigma_center=ctr,center_bias=bias,contract=name,mode=mode)
     pieces.append(p.assign(**meta))
     rows.append(dict(**meta,**summarize(p,a.bootstrap,sp)))
   pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
 pd.concat(pieces).to_csv(out/'predictions.csv',index=False)
 save_json({'settings':settings,'features':contracts,'catalogue':'Fixed accepted SIS sample; no new SNR/delay selection','paired_standard_normal_draws':True,'centroid_bias':'Shared offset along primary image direction; controlled misspecification, not a complete lens reconstruction error model','matched_retrain':'Train and calibration observations at test noise level','frozen_nominal':'Train/calibration fixed at .01/.02 arcsec, evaluate shifted noise','target':'log(mu0-1)'},out/'design.json')

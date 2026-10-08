"""NFW angular normalization control, preserving the exact dimensionless map.

Dilate every NFW angular coordinate/scale by c = theta_reference/r_tangential.
The Hessian and magnification are invariant under this transformation; the
critical-radius distribution then equals the shared empirical reference scale.
Only geometry is used. No copied GW amplitudes/time series or velocity model.
"""
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from functools import lru_cache
from common import *
from mnras_noise_variants import observe,BASE
import matched_transfer

@lru_cache(maxsize=128)
def dimensionless_tangential_radius(rs_ratio,alpha_ratio):
 from lenstronomy.LensModel.lens_model import LensModel
 model=LensModel(['NFW']);kw=[{'Rs':rs_ratio,'alpha_Rs':alpha_ratio,'center_x':0.,'center_y':0.}]
 def f(r):return float(model.alpha(r,0.,kw)[0]/r-1)
 grid=np.geomspace(1e-5,1e3,500);roots=[]
 for x,y in zip(grid[:-1],grid[1:]):
  if f(x)*f(y)<0:roots.append(brentq(f,x,y,xtol=1e-12))
 if not roots:raise ValueError('No finite tangential critical radius found for NFW')
 return max(roots)


def scale_rows(row,primary,secondary):
 from lenstronomy.LensModel.lens_model import LensModel
 model=LensModel(['NFW']);kw=json.loads(primary.lens_kwargs_json);theta=float(row.theta_E_true)
 rt=theta*dimensionless_tangential_radius(round(kw[0]['Rs']/theta,12),round(kw[0]['alpha_Rs']/theta,12));factor=theta/rt
 scaled=[dict(kw[0])]
 for name in ['Rs','alpha_Rs','center_x','center_y']:scaled[0][name]*=factor
 pp=primary.copy();ss=secondary.copy()
 for im in [pp,ss]:
  oldx,oldy=float(im.x_true),float(im.y_image_true)
  im.x_true*=factor;im.y_image_true*=factor
  h0=np.asarray(model.hessian(oldx,oldy,kw));h1=np.asarray(model.hessian(im.x_true,im.y_image_true,scaled))
  np.testing.assert_allclose(h1,h0,rtol=2e-10,atol=2e-10)
  beta0=np.array(model.ray_shooting(oldx,oldy,kw));beta1=np.array(model.ray_shooting(im.x_true,im.y_image_true,scaled))
  np.testing.assert_allclose(beta1,factor*beta0,rtol=1e-9,atol=1e-10)
 return pp,ss,{'angular_factor':factor,'old_tangential_radius':rt,'new_tangential_radius':theta,'reference_angle':theta,'scaled_kwargs_json':json.dumps(scaled)}

def control_loader(a,gen,out,normalize=True):
 folder=a.formal_root/f'seed_{gen}/benchmark';data=load_benchmark(folder);pieces=[];records=[]
 for fam,g in data.groupby('lens_family'):
  img=pd.read_csv(folder/fam/'image_latent.csv');prim=img[img.is_primary].set_index('event_id');sec=img[img.is_secondary].set_index('event_id');obs=[]
  for row in g.itertuples(index=False):
   primary=prim.loc[row.event_id].copy();secondary=sec.loc[row.event_id].copy();config={}
   if fam=='NFW_like' and normalize:
    # Cache the root by dimensionless profile parameters;
    # scale_rows still verifies the physical Hessian and ray mapping each time.
    primary,secondary,config=scale_rows(row,primary,secondary)
    records.append(dict(generation_seed=gen,event_id=row.event_id,event_key=row.event_key,identity_key=row.identity_key,**config))
   # Common random numbers across physical scales. Existing source identities
   # and train/calibration/test roles are retained exactly.
   z=np.random.default_rng(stable_seed('angle_control',gen,fam,row.event_id)).standard_normal(10)
   obs.append(observe(row,primary,secondary,BASE,z))
  current=g.copy();m=pd.DataFrame(obs,index=current.index)
  for c in GEOMETRY:current[c]=m[c]
  pieces.append(current)
 pd.DataFrame(records).to_csv(out/f'scale_audit_{gen}.csv',index=False)
 answer=pd.concat(pieces).sort_index();answer.to_csv(out/f'controlled_catalogue_{gen}.csv',index=False)
 return answer

def run(a,out):
 for tag,normalize in [('nominal_reobserved',False),('critical_radius_normalized',True)]:
  dest=out/tag;dest.mkdir(exist_ok=True)
  matched_transfer.run(a,dest,loader=lambda aa,gg,oo:control_loader(aa,gg,oo,normalize))
 save_json({'transformation':'NFW theta -> theta/r_t,0, equivalently multiply all angular scales and source/image coordinates by reference_theta / original_tangential_radius','invariants':'Source/lens redshifts, dimensionless Hessian, determinant, singular values, magnifications and empirical prior identity','changed':'NFW angular image geometry, then newly measured positions with 0.01 arcsec astrometry and 0.02 arcsec centroid noise','measurement':'Nominal and scaled catalogues both reobserved from exact latent positions with the same standard-normal draws; controlled catalogue only refreshes geometry inputs for prediction','predictors':'Angular geometry and dimensionless geometry; no direct reference-angle or sigma_v input','validation':'Hessian invariance and scaled ray-shooting checked on both primary images for every NFW event','limits':'Controls the NFW angular normalization, not the radial profile or magnification distribution; offline matched-mu and matched-mu+separation analyses additionally reported; not a waveform/detection simulation'},out/'scale_control_design.json')

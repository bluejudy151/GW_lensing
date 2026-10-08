#!/usr/bin/env python
# coding: utf-8
"""Leakage-audited four-family lensing evaluation with held-out calibration.

This is an empirical conditional-on-simulator experiment, not an identifiability
proof. True Jacobians enter supervised auxiliary targets and diagnostic plots;
no true test latent is used by a deployable selection or uncertainty rule.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import os
# Avoid severe OpenMP oversubscription on shared CPU servers.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm.auto import tqdm

GEOMETRY = ['obs__image_separation_observed', 'obs__theta_plus_abs_observed',
            'obs__theta_minus_abs_observed', 'obs__image_position_asymmetry_observed',
            'obs__lens_center_x_observed', 'obs__lens_center_y_observed',
            'obs__image_x_0_observed', 'obs__image_y_0_observed',
            'obs__image_x_1_observed', 'obs__image_y_1_observed']
FOLLOWUP = ['obs__z_s_observed', 'obs__z_l_observed', 'obs__z_l_over_z_s',
            'obs__sigma_v_observed', 'obs__log_sigma_v_observed']
DIRECT_SCALE = ['obs__theta_E_observed_direct']
LATENT = ['detA_true', 'singular_value_min_true', 'jacobian_condition_true']
FAMILIES = ['SIS', 'SIE', 'SIS_shear', 'NFW_like']


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--parameter-ood-root', type=Path)
    p.add_argument('--composite-root', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[6130, 6131, 6132])
    p.add_argument('--bootstrap', type=int, default=300)
    p.add_argument('--neighbors-per-family', type=int, default=1000)
    p.add_argument('--max-iter', type=int, default=180)
    p.add_argument('--ensemble-members', type=int, default=3)
    p.add_argument('--contracts', nargs='+', default=None)
    p.add_argument('--critical-smin', type=float, default=.03)
    p.add_argument('--near-critical-smin', type=float, default=.1)
    p.add_argument('--ambiguity-ratio', type=float, default=2.)
    p.add_argument('--skip-ambiguity', action='store_true')
    p.add_argument('--protocols', nargs='+', choices=['single_family','mixed_family','leave_one_family_out'], default=['single_family','mixed_family','leave_one_family_out'])
    return p.parse_args()


def finite_quantile(x, level):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return np.nan
    return float(np.quantile(x, level))


def conformal_quantile(scores, alpha=.1):
    """Finite-sample split-conformal order statistic (no test labels)."""
    scores = np.asarray(scores, float); scores = scores[np.isfinite(scores)]
    if not len(scores): return np.inf
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else float(np.partition(scores, k-1)[k-1])


def metric(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    e = pred-y; re = np.abs(e)/np.maximum(y, 1e-12)
    le = np.abs(np.log(np.maximum(pred, 1e-12))-np.log(np.maximum(y, 1e-12)))
    return dict(n=len(y), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e*e))),
                bias=float(np.mean(e)), median_relative_error=float(np.median(re)),
                mean_relative_error=float(np.mean(re)), p90_relative_error=finite_quantile(re,.9),
                median_log_error=float(np.median(le)), p90_log_error=finite_quantile(le,.9))


def load_benchmark(root, external=False):
    frames=[]
    for folder in sorted(root.iterdir()):
        if not (folder/'observables.csv').exists(): continue
        obs = pd.read_csv(folder/'observables.csv'); lat = pd.read_csv(folder/'event_latent.csv')
        if 'event_id' not in obs or obs.event_id.duplicated().any():
            raise ValueError(f'Unique event_id required: {folder}')
        if lat.event_id.duplicated().any(): raise ValueError(f'Duplicate latent event: {folder}')
        obs['lens_family'] = folder.name
        missing = [c for c in LATENT if c not in lat]
        if missing: raise ValueError(f'{folder}: missing exact primary Jacobian columns {missing}')
        if set(obs.event_id)!=set(lat.event_id): raise ValueError(f'{folder}: observation/latent event IDs differ')
        latcols = ['event_id'] + LATENT
        for c in ['kappa_true','gamma1_true','gamma2_true','source_group_id', 'identity_group_id', 'split']:
            if c in lat and c not in obs: latcols.append(c)
        obs = obs.drop(columns=[c for c in LATENT if c in obs]).merge(lat[latcols],on='event_id',validate='one_to_one')
        if 'mu0_true' not in obs: raise ValueError(f'{folder}: missing mu0_true')
        if not (np.isfinite(obs.mu0_true)&(obs.mu0_true>0)).all():
            raise ValueError('Positive finite absolute primary-image magnification required')
        if 'split' not in obs:
            raise ValueError(f'{folder}: stored identity-safe split is required; regenerate with the new generator')
        obs['split'] = obs['split'].replace({'val':'calibration','validation':'calibration','cal':'calibration'})
        if not set(obs.split).issubset({'train','calibration','test'}):
            raise ValueError(f'{folder}: unknown split names {obs.split.unique()}')
        identity = next((c for c in ['identity_group_id','source_group_id','group_id','source_event_id'] if c in obs), 'event_id')
        # Shared source identities across families are never split across roles.
        obs['identity_key'] = obs[identity].astype(str)
        if identity=='event_id': obs['identity_key'] = folder.name+':'+obs.identity_key
        obs['event_key'] = folder.name+':'+obs.event_id.astype(str)
        obs['catalogue'] = str(root.resolve())
        frames.append(obs)
    if not frames: raise ValueError(f'No family tables found under {root}')
    data=pd.concat(frames,ignore_index=True)
    if not external and set(data.lens_family) != set(FAMILIES):
        raise ValueError(f'Four families required; got {sorted(data.lens_family.unique())}')
    leaking = data.groupby('identity_key').split.nunique()
    if (leaking>1).any(): raise ValueError('Source identity reused across train/calibration/test')
    for name,g in data.groupby('lens_family'):
        if not external and any((g.split==s).sum()<2 for s in ['train','calibration','test']):
            raise ValueError(f'{name}: at least two events per stored split required')
    return data


def feature_contracts(data):
    numeric = set(data.select_dtypes(include=[np.number]).columns)
    g = [c for c in GEOMETRY if c in numeric]
    f = [c for c in FOLLOWUP if c in numeric]
    w = sorted(c for c in numeric if c.startswith('obs__gw_') and all(not part[c].isna().all() for _,part in data.groupby('lens_family')))
    scale=[c for c in DIRECT_SCALE if c in numeric]
    for c in g+f+w+scale:
        if any(t in c.lower() for t in ['true','latent','determinant','jacobian','kappa','gamma','mu0','mu1','family']):
            raise ValueError(f'Forbidden feature name: {c}')
    contracts={'geometry':g,'geometry_followup':g+f+scale}
    if w: contracts.update(gw_only=w,gw_followup=w+f,full_observed=w+f+g+scale)
    return {k:list(dict.fromkeys(v)) for k,v in contracts.items() if v}


class FittedModel:
    def __init__(self, train, cal, cols, seed, args):
        self.cols=cols; self.known=set(train.lens_family)
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        X=self.imputer.fit_transform(train[cols]); C=self.imputer.transform(cal[cols])
        if not np.isfinite(X).all(): raise ValueError('Nonfinite imputed training features')
        self.scaler=RobustScaler().fit(X); Xs=self.scaler.transform(X); Cs=self.scaler.transform(C)
        self.models=[]; rng=np.random.default_rng(seed)
        target=np.log(train.mu0_true.to_numpy(float))
        train_groups=train.identity_key.to_numpy()
        unique_groups=np.unique(train_groups)
        for j in range(args.ensemble_members):
            # Cluster bootstrap preserves identities if a source has several family realizations.
            sampled=rng.choice(unique_groups,len(unique_groups),replace=True)
            ids=np.concatenate([np.flatnonzero(train_groups==z) for z in sampled])
            model=self.regressor(seed+j,args)
            model.fit(X[ids],target[ids]); self.models.append(model)
        self.condition_model=self.regressor(seed+779,args)
        self.condition_model.fit(X,np.log(np.maximum(train.jacobian_condition_true.to_numpy(float),1.)))
        self.det_model=self.regressor(seed+991,args)
        self.det_model.fit(X,np.log(np.maximum(np.abs(train.detA_true.to_numpy(float)),1e-14)))
        # Robust standardized distances; reference subset chosen only from training.
        refids=rng.choice(len(Xs),min(5000,len(Xs)),replace=False)
        self.nn=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(Xs[refids])
        self.ood_threshold=finite_quantile(self.nn.kneighbors(Cs)[0][:,0],.95)
        self.family_nn={}; self.family_ood_threshold={}
        for fam in sorted(self.known):
            ref=Xs[train.lens_family.to_numpy()==fam]
            if len(ref)>5000: ref=ref[rng.choice(len(ref),5000,replace=False)]
            self.family_nn[fam]=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(ref)
            cmask=cal.lens_family.to_numpy()==fam
            self.family_ood_threshold[fam]=finite_quantile(self.family_nn[fam].kneighbors(Cs[cmask])[0][:,0],.95) if cmask.any() else np.nan
        cr=self.raw(C); ycal=cal.mu0_true.to_numpy(float)
        # Nonnegative CQR correction never contracts the raw ensemble interval.
        scores=np.maximum.reduce([cr['raw_lo']-ycal,ycal-cr['raw_hi'],np.zeros(len(cal))])
        # Calibration unit is an identity, not an image or repeated source record.
        calibration=pd.DataFrame({'identity':cal.identity_key.to_numpy(),'score':scores,'family':cal.lens_family.to_numpy()})
        self.q=conformal_quantile(calibration.groupby('identity').score.max())
        self.family_q={fam:conformal_quantile(g.groupby('identity').score.max()) for fam,g in calibration.groupby('family')}
        condition=cr['predicted_condition']
        self.condition_edges=np.unique(np.quantile(condition,[.25,.5,.75]))
        labels=np.digitize(condition,self.condition_edges); self.condition_q={}
        for label in np.unique(labels):
            g=calibration[labels==label]
            self.condition_q[int(label)]=conformal_quantile(g.groupby('identity').score.max()) if g.identity.nunique()>=20 else self.q
        self.calibration_n=calibration.identity.nunique()

    @staticmethod
    def regressor(seed,args):
        return HistGradientBoostingRegressor(max_iter=args.max_iter,max_leaf_nodes=15,
                min_samples_leaf=10,learning_rate=.06,l2_regularization=.1,
                early_stopping=False,random_state=seed)

    def raw(self,X):
        ensemble=np.exp(np.clip(np.column_stack([m.predict(X) for m in self.models]),-30,30))
        return dict(mu0_pred=np.median(ensemble,axis=1),raw_lo=np.quantile(ensemble,.05,axis=1),
                    raw_hi=np.quantile(ensemble,.95,axis=1),ensemble_std=np.std(ensemble,axis=1),
                    predicted_condition=np.exp(np.clip(self.condition_model.predict(X),0,30)),
                    predicted_abs_detA=np.exp(np.clip(self.det_model.predict(X),-30,30)))

    def predict(self,test):
        X=self.imputer.transform(test[self.cols]); Xs=self.scaler.transform(X); out=self.raw(X)
        n=len(test); out['conformal_lo']=np.maximum(0,out['raw_lo']-self.q); out['conformal_hi']=out['raw_hi']+self.q
        fq=np.array([self.family_q.get(f,np.nan) for f in test.lens_family])
        out['family_conditional_lo']=np.maximum(0,out['raw_lo']-fq); out['family_conditional_hi']=out['raw_hi']+fq
        labels=np.digitize(out['predicted_condition'],self.condition_edges)
        cq=np.array([self.condition_q.get(int(j),self.q) for j in labels])
        out['predicted_condition_lo']=np.maximum(0,out['raw_lo']-cq); out['predicted_condition_hi']=out['raw_hi']+cq
        out['ood_score']=self.nn.kneighbors(Xs)[0][:,0]; out['ood_threshold']=np.full(n,self.ood_threshold)
        out['family_aware_ood_score']=np.full(n,np.nan); out['family_aware_ood_threshold']=np.full(n,np.nan)
        for fam in self.known:
            mask=test.lens_family.to_numpy()==fam
            if mask.any():
                out['family_aware_ood_score'][mask]=self.family_nn[fam].kneighbors(Xs[mask])[0][:,0]
                out['family_aware_ood_threshold'][mask]=self.family_ood_threshold[fam]
        out['known_family']=test.lens_family.isin(self.known).to_numpy()
        out['ood_flag']=out['ood_score']>out['ood_threshold']
        out['family_aware_ood_flag']=np.where(out['known_family'],out['family_aware_ood_score']>out['family_aware_ood_threshold'],True)
        out['calibration_identity_count']=np.full(n,self.calibration_n)
        return pd.DataFrame(out,index=test.index)


def matrix(data,externals,contracts,args):
    pieces=[]; summaries=[]; calibration_audit=[]
    protocols=[('single_family',f,[f]) for f in FAMILIES]
    protocols += [('mixed_family','ALL',FAMILIES)]
    protocols += [('leave_one_family_out','ALL_EXCEPT_'+f,[x for x in FAMILIES if x!=f]) for f in FAMILIES]
    protocols=[p for p in protocols if p[0] in args.protocols]
    total=len(args.seeds)*len(contracts)*len(protocols)
    with tqdm(total=total,desc='Model/calibration fits',unit='fit') as progress:
        for seed, (contract,cols), (protocol,training_label,train_families) in itertools.product(args.seeds,contracts.items(),protocols):
            train=data[(data.split=='train')&data.lens_family.isin(train_families)]
            cal=data[(data.split=='calibration')&data.lens_family.isin(train_families)]
            fit=FittedModel(train,cal,cols,seed,args)
            calibration_audit.append(dict(seed=seed,contract=contract,protocol=protocol,train_family=training_label,
                train_events=len(train),calibration_events=len(cal),calibration_identities=fit.calibration_n,
                conformal_correction=fit.q,family_conditional=json.dumps(fit.family_q),
                ood_threshold=fit.ood_threshold,exchangeability_required=True))
            domains=[('benchmark',data[data.split=='test'])]+[(name,d[d.split=='test']) for name,d in externals.items()]
            for domain,test in domains:
                if not len(test): continue
                if any(c not in test or test[c].isna().all() for c in cols): continue
                if protocol=='leave_one_family_out' and domain=='benchmark':
                    test=test[~test.lens_family.isin(train_families)]
                if not len(test): continue
                result=fit.predict(test)
                basecols=['event_key','event_id','identity_key','lens_family','mu0_true']+LATENT+[c for c in ['diagnostic__optimal_pair_snr'] if c in test]
                frame=test[basecols].join(result).reset_index(drop=True)
                frame['seed']=seed; frame['contract']=contract; frame['protocol']=protocol
                frame['train_family']=training_label; frame['domain']=domain
                frame['test_family']=frame.lens_family
                frame['abs_error']=abs(frame.mu0_pred-frame.mu0_true)
                frame['abs_detA']=abs(frame.detA_true)
                frame['smin']=frame.singular_value_min_true
                frame['condition']=frame.jacobian_condition_true
                frame['critical_region']=np.select([frame.smin<args.critical_smin,frame.smin<args.near_critical_smin],['critical_neighborhood','near_critical'],default='normal')
                for method in ['raw','conformal','family_conditional','predicted_condition']:
                    available=frame[method+'_lo'].notna()&frame[method+'_hi'].notna()
                    frame[method+'_covered']=np.where(available,((frame.mu0_true>=frame[method+'_lo'])&(frame.mu0_true<=frame[method+'_hi'])).astype(float),np.nan)
                    frame[method+'_width']=frame[method+'_hi']-frame[method+'_lo']
                pieces.append(frame)
                for family,g in frame.groupby('test_family'):
                    row=dict(seed=seed,contract=contract,protocol=protocol,train_family=training_label,test_family=family,domain=domain,
                             shift=('in_distribution' if family in train_families else 'cross_family_ood') if domain=='benchmark' else ('composite_family_ood' if domain=='composite_ood' else ('same_family_parameter_ood' if family in train_families else 'parameter_and_family_ood')),**metric(g.mu0_true,g.mu0_pred))
                    for method in ['raw','conformal','family_conditional','predicted_condition']:
                        row[method+'_coverage90']=g[method+'_covered'].mean(); row[method+'_mean_width']=g[method+'_width'].mean()
                    row['ood_rejection_fraction']=float((g.ood_score>g.ood_threshold).mean()); summaries.append(row)
            progress.update(); progress.set_postfix(seed=seed,contract=contract,protocol=protocol)
    if not pieces: raise ValueError('No predictions were produced')
    pred=pd.concat(pieces,ignore_index=True); metrics=pd.DataFrame(summaries)
    pred.to_csv(args.out_dir/'family_matrix_predictions.csv',index=False)
    metrics.to_csv(args.out_dir/'family_matrix_metrics.csv',index=False)
    groups=['contract','protocol','train_family','test_family','domain','shift']
    numeric=[c for c in metrics if c not in groups+['seed','n']]
    summary=metrics.groupby(groups)[numeric].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in summary.columns]
    summary.to_csv(args.out_dir/'family_matrix_seed_summary.csv',index=False)
    pd.DataFrame(calibration_audit).to_csv(args.out_dir/'calibration_audit.csv',index=False)
    return pred


def slope(x,y):
    mask=np.isfinite(x)&np.isfinite(y); x=np.asarray(x)[mask]; y=np.asarray(y)[mask]
    return float(np.polyfit(x,y,1)[0]) if len(x)>2 and np.ptp(x)>1e-10 else np.nan


def boot_ci(values):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    return (float(np.quantile(values,.025)),float(np.quantile(values,.975))) if len(values) else (np.nan,np.nan)


def diagnostic_summaries(pred,args):
    key=['contract','protocol','train_family','test_family','domain']
    rows=[]; bins=[]; selective=[]; metric_bootstrap=[]; rng=np.random.default_rng(882)
    for group,g in tqdm(pred.groupby(key),desc='Jacobian/coverage/selection',unit='cell'):
        meta=dict(zip(key,group))
        # Average repeated model-seed predictions within event before bootstrapping;
        # resample identity clusters so replicated source realizations are not independent.
        ev=g.groupby(['identity_key','event_key'],as_index=False).agg(
            abs_error=('abs_error','mean'),abs_detA=('abs_detA','first'),smin=('smin','first'),
            condition=('condition','first'),mu=('mu0_true','first'),interval_width=('conformal_width','mean'))
        units=ev.identity_key.unique(); unit_rows={u:np.flatnonzero(ev.identity_key.to_numpy()==u) for u in units}
        for axis in ['abs_detA','smin','condition']:
            x=np.log(np.maximum(ev[axis].to_numpy(),1e-14))
            for response in ['abs_error','interval_width']:
                y=np.log(np.maximum(ev[response].to_numpy(),1e-14))
                boots=[]
                for _ in range(args.bootstrap):
                    ids=np.concatenate([unit_rows[u] for u in rng.choice(units,len(units),replace=True)])
                    boots.append(slope(x[ids],y[ids]))
                lo,hi=boot_ci(boots); mask=np.isfinite(x)&np.isfinite(y)
                corr=float(np.corrcoef(x[mask],y[mask])[0,1]) if mask.sum()>2 and np.ptp(x[mask])>1e-10 and np.ptp(y[mask])>1e-10 else np.nan
                sr=float(spearmanr(x[mask],y[mask]).statistic) if mask.sum()>2 and np.ptp(x[mask])>1e-10 and np.ptp(y[mask])>1e-10 else np.nan
                rows.append(dict(**meta,axis=axis,response=response,n_events=len(ev),n_identity_clusters=len(units),slope_loglog=slope(x,y),slope_ci_lo=lo,slope_ci_hi=hi,pearson_loglog=corr,spearman=sr))
        # Report cluster-bootstrap mean risks and coverage without treating model
        # seeds or repeated realizations as independent observations.
        mg=g.copy()
        mg['bias']=mg.mu0_pred-mg.mu0_true
        mg['mean_relative_error']=mg.abs_error/mg.mu0_true
        mg['mean_log_error']=abs(np.log(mg.mu0_pred)-np.log(mg.mu0_true))
        value_cols=['abs_error','bias','mean_relative_error','mean_log_error','conformal_covered','raw_covered','predicted_condition_covered']
        mg=mg.groupby(['identity_key','event_key'],as_index=False)[value_cols].mean()
        sums=mg.groupby('identity_key')[value_cols].sum(); counts=mg.groupby('identity_key')[value_cols].count()
        ids=np.arange(len(sums)); samples=[]
        sv=sums.to_numpy(); cv=counts.to_numpy()
        for _ in range(args.bootstrap):
            chosen=rng.choice(ids,len(ids),replace=True)
            samples.append(sv[chosen].sum(axis=0)/np.maximum(cv[chosen].sum(axis=0),1))
        samples=np.asarray(samples)
        for j,col in enumerate(value_cols):
            lo,hi=boot_ci(samples[:,j])
            metric_bootstrap.append(dict(**meta,metric='mae' if col=='abs_error' else col,
                estimate=mg[col].mean(),ci_lo=lo,ci_hi=hi,n_events=len(mg),
                n_identity_clusters=len(sums),resampling_unit='identity_cluster_after_model_seed_average'))
        # Quantile bins are descriptive test-data display bins, never calibration rules.
        for seed,sg in g.groupby('seed'):
            for axis in ['mu0_true','abs_detA','smin','critical_region']+(['diagnostic__optimal_pair_snr'] if 'diagnostic__optimal_pair_snr' in sg else []):
                labels=sg.critical_region if axis=='critical_region' else pd.qcut(sg[axis],q=min(5,sg[axis].nunique()),duplicates='drop').astype(str)
                if axis=='diagnostic__optimal_pair_snr': labels=pd.cut(sg[axis],bins=[0,8,16,32,64,np.inf],include_lowest=True).astype(str)
                for label,bg in sg.groupby(labels,observed=True):
                    row=dict(**meta,seed=seed,axis=axis,bin=str(label),n=len(bg),mae=bg.abs_error.mean())
                    for m in ['raw','conformal','family_conditional','predicted_condition']:
                        row[m+'_coverage90']=bg[m+'_covered'].mean(); row[m+'_mean_width']=bg[m+'_width'].mean()
                    bins.append(row)
            scores={'interval_width':'conformal_width','family_agnostic_ood':'ood_score',
                    'family_aware_ood':'family_aware_ood_score','predicted_jacobian_condition':'predicted_condition',
                    'oracle_true_condition_DIAGNOSTIC_ONLY':'condition'}
            for name,col in scores.items():
                eligible=sg[sg[col].notna()].sort_values(col)
                if not len(eligible): continue
                for fraction in np.linspace(.1,1,10):
                    keep=eligible.iloc[:max(1,int(np.ceil(fraction*len(eligible))))]
                    selective.append(dict(**meta,seed=seed,selection_score=name,deployment_uses_true_latent=name.startswith('oracle'),
                        requested_retained_fraction=fraction,retained_fraction=len(keep)/len(sg),n=len(keep),
                        mae=keep.abs_error.mean(),coverage90=keep.conformal_covered.mean(),mean_interval_width=keep.conformal_width.mean()))
            for label,score,threshold in [('family_agnostic_ood','ood_score','ood_threshold'),('family_aware_ood','family_aware_ood_score','family_aware_ood_threshold')]:
                keep=sg[sg[score]<=sg[threshold]]
                selective.append(dict(**meta,seed=seed,selection_score=label+'_calibration_threshold',deployment_uses_true_latent=False,
                    requested_retained_fraction=np.nan,retained_fraction=len(keep)/len(sg),n=len(keep),mae=keep.abs_error.mean(),
                    coverage90=keep.conformal_covered.mean(),mean_interval_width=keep.conformal_width.mean()))
    pd.DataFrame(rows).to_csv(args.out_dir/'jacobian_boundary_summary.csv',index=False)
    pd.DataFrame(bins).to_csv(args.out_dir/'coverage_by_physical_region.csv',index=False)
    pd.DataFrame(selective).to_csv(args.out_dir/'selective_risk.csv',index=False)
    pd.DataFrame(metric_bootstrap).to_csv(args.out_dir/'metrics_identity_bootstrap.csv',index=False)
    # Predictions already contain the complete event-level boundary and uncertainty data.


class DistanceTransform:
    def __init__(self,train,kind):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        X=self.imputer.fit_transform(train)
        self.scale=RobustScaler() if kind=='robust' else StandardScaler()
        X=self.scale.fit_transform(X); self.extra=None; self.whitener=None
        if kind=='pca_whiten':
            candidate=PCA(svd_solver='full').fit(X)
            good=candidate.explained_variance_>max(candidate.explained_variance_.max()*1e-10,1e-12)
            self.whitener=(candidate.components_[good].T/np.sqrt(candidate.explained_variance_[good]))
        elif kind=='mahalanobis':
            covariance=LedoitWolf().fit(X).covariance_
            vals,vec=np.linalg.eigh(covariance)
            self.whitener=vec@np.diag(1/np.sqrt(np.maximum(vals,1e-10)))
    def transform(self,X):
        X=self.scale.transform(self.imputer.transform(X))
        return X if self.whitener is None else X@self.whitener


def ambiguity_maps(data,contracts,args):
    rows=[]; rng=np.random.default_rng(333)
    map_contracts={k:v for k,v in contracts.items() if k in ['geometry','geometry_followup','full_observed']}
    tasks=list(itertools.product(map_contracts.items(),['zscore','robust','mahalanobis','pca_whiten']))
    for (contract,cols),scaling in tqdm(tasks,desc='Observable ambiguity maps',unit='map'):
        transformer=DistanceTransform(data.loc[data.split=='train',cols],scaling)
        for fa,fb in itertools.combinations(FAMILIES,2):
            a=data[(data.split=='test')&(data.lens_family==fa)].sample(frac=1,random_state=51).head(args.neighbors_per_family)
            b=data[(data.split=='test')&(data.lens_family==fb)].sample(frac=1,random_state=52).head(args.neighbors_per_family)
            if len(a)<1 or len(b)<1: continue
            # Nearest neighbours are distinct source identities: shared source priors
            # must not manufacture a zero-distance cross-family match.
            za=transformer.transform(a[cols]); zb=transformer.transform(b[cols])
            nbr=NearestNeighbors(n_neighbors=min(len(b),20),n_jobs=1).fit(zb)
            distances,indices=nbr.kneighbors(za); chosen=[]
            for ia in range(len(a)):
                candidates=[j for j in range(indices.shape[1]) if b.iloc[indices[ia,j]].identity_key!=a.iloc[ia].identity_key]
                if candidates: chosen.append((ia,int(indices[ia,candidates[0]]),float(distances[ia,candidates[0]])))
            if not chosen: continue
            ia=np.array([x[0] for x in chosen]); ib=np.array([x[1] for x in chosen]); dist=np.array([x[2] for x in chosen])
            aa=a.iloc[ia].reset_index(drop=True); bb=b.iloc[ib].reset_index(drop=True)
            ma=aa.mu0_true.to_numpy(); mb=bb.mu0_true.to_numpy()
            pair=pd.DataFrame(dict(contract=contract,scaler=scaling,family_a=fa,family_b=fb,
                event_a=aa.event_key,event_b=bb.event_key,identity_a=aa.identity_key,identity_b=bb.identity_key,
                observable_distance=dist,mu_a=ma,mu_b=mb,abs_delta_mu=abs(ma-mb),
                mu_ratio=np.maximum(ma,mb)/np.minimum(ma,mb),abs_detA_a=abs(aa.detA_true),abs_detA_b=abs(bb.detA_true)))
            pair['ambiguity_ratio_exceeded']=pair.mu_ratio>args.ambiguity_ratio
            pair['distance_bin']=pd.qcut(pair.observable_distance,q=min(5,pair.observable_distance.nunique()),labels=False,duplicates='drop') if pair.observable_distance.nunique()>1 else 0
            pair.to_csv(args.out_dir/f'ambiguity_pairs_{contract}_{scaling}_{fa}_{fb}.csv',index=False)
            for label,g in pair.groupby('distance_bin'):
                # Two-way identity-cluster Poisson bootstrap accounts for reused
                # neighbours; matching itself is held fixed in this uncertainty band.
                ua=g.identity_a.unique(); ub=g.identity_b.unique(); boot_m=[]; boot_f=[]
                av=g.identity_a.to_numpy(); bv=g.identity_b.to_numpy(); dv=g.abs_delta_mu.to_numpy(); flag=g.ambiguity_ratio_exceeded.to_numpy(float)
                for _ in range(args.bootstrap):
                    wa=dict(zip(ua,rng.poisson(1,len(ua)))); wb=dict(zip(ub,rng.poisson(1,len(ub))))
                    weights=np.array([wa[x]*wb[y] for x,y in zip(av,bv)])
                    if weights.sum():
                        expanded=np.repeat(dv,weights); boot_m.append(float(np.median(expanded))); boot_f.append(float(np.average(flag,weights=weights)))
                mlo,mhi=boot_ci(boot_m); flo,fhi=boot_ci(boot_f)
                rows.append(dict(contract=contract,scaler=scaling,family_a=fa,family_b=fb,distance_bin=label,
                    distance_lo=g.observable_distance.min(),distance_hi=g.observable_distance.max(),n_pairs=len(g),
                    n_query_identities=len(ua),n_reference_identities=len(ub),median_abs_delta_mu=g.abs_delta_mu.median(),
                    median_ci_lo=mlo,median_ci_hi=mhi,ambiguity_ratio_threshold=args.ambiguity_ratio,
                    ambiguity_fraction=g.ambiguity_ratio_exceeded.mean(),ambiguity_fraction_ci_lo=flo,
                    ambiguity_fraction_ci_hi=fhi,bootstrap='two_way_identity_poisson_fixed_matching'))
    pd.DataFrame(rows).to_csv(args.out_dir/'ambiguity_map_summary.csv',index=False)


def main():
    args=parse()
    if args.ensemble_members<1 or args.bootstrap<1: raise ValueError('Positive ensemble and bootstrap counts required')
    if not 0<args.critical_smin<args.near_critical_smin: raise ValueError('Invalid smin region thresholds')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    data=load_benchmark(args.benchmark_root); contracts=feature_contracts(data)
    if args.contracts:
        missing=set(args.contracts)-set(contracts)
        if missing: raise ValueError(f'Unavailable observable contracts {missing}')
        contracts={k:contracts[k] for k in args.contracts}
    for contract,cols in contracts.items():
        for family,part in data.groupby('lens_family'):
            empty=[c for c in cols if part[c].isna().all()]
            if empty: raise ValueError(f'{contract}/{family}: unavailable feature columns {empty}')
    externals={}
    for name,path in [('parameter_ood',args.parameter_ood_root),('composite_ood',args.composite_root)]:
        if path:
            if path.resolve()==args.benchmark_root.resolve():
                raise ValueError(f'{name}: external catalogue must differ from the training benchmark')
            external=load_benchmark(path,external=True)
            # Source IDs can be shared only if their held-out split is the same.
            joined=data[['identity_key','split']].drop_duplicates().merge(external[['identity_key','split']].drop_duplicates(),on='identity_key')
            if len(joined) and (joined.split_x!=joined.split_y).any():
                raise ValueError(f'{name}: cross-catalogue identity split leakage')
            externals[name]=external
    data[['lens_family','event_id','event_key','identity_key','split']].to_csv(args.out_dir/'split_manifest.csv',index=False)
    audit=[]
    for contract,cols in contracts.items():
        for col in cols: audit.append(dict(contract=contract,feature=col,allowed_observed_feature=True,uses_true_latent=False))
    pd.DataFrame(audit).to_csv(args.out_dir/'feature_contract_audit.csv',index=False)
    info=[]
    for contract in ['geometry','geometry_followup','gw_only','gw_followup','full_observed']:
        for fam,g in data.groupby('lens_family'):
            cols=contracts.get(contract,[]); available=bool(cols) and all(c in g and not g[c].isna().all() for c in cols)
            info.append(dict(lens_family=fam,contract=contract,available=available,n_features=len(cols),
                             status='evaluated' if available else 'unavailable_or_not_requested'))
    pd.DataFrame(info).to_csv(args.out_dir/'information_hierarchy_contracts.csv',index=False)
    config=dict(vars(args)); config.update(versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_contracts=contracts,
        followup_contract='GW plus follow-up uses redshifts and velocity dispersion; directly observed lens normalization is reserved for geometry/full-observed contracts',
        target='log absolute magnification of brightness-ranked primary image; no mu>1 assumption',
        split='Generator-stored identity-safe train/calibration/test; model seeds never resplit events',
        uncertainty='Empirical bootstrap ensemble intervals; separate split-conformal correction; source-identity maximum calibration score',
        caveats=['Cross-family and parameter OOD do not inherit in-distribution conformal coverage guarantees.',
            'Family-conditional intervals require a known family with calibration examples; unknown families are NA.',
            'Predicted-condition bins have calibration-driven fallback; they are not exact pointwise conditional coverage.',
            'Critical-neighborhood is defined by dimensionless s_min thresholds; it is not a traced source-plane caustic.',
            'Conformal calibration with fewer than nine independent identities returns an infinite correction; tiny smoke-run intervals are intentionally uninformative.',
            'Nearest-neighbour maps are finite-sample ambiguity diagnostics, not a proof of non-identifiability.',
            'Map confidence intervals condition on fixed nearest-neighbour matches and use two-way identity clusters.',
            'Conditioning slopes depend on features, selection and simulator; universality must be tested, not assumed.',
            'Model-seed variation uses one fixed benchmark; it does not measure population-generation seed uncertainty.'])
    (args.out_dir/'suite_config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
    pred=matrix(data,externals,contracts,args)
    diagnostic_summaries(pred,args)
    if not args.skip_ambiguity: ambiguity_maps(data,contracts,args)
    print(f'UNIVERSAL_EVALUATION_OUT={args.out_dir.resolve()}')


if __name__=='__main__': main()

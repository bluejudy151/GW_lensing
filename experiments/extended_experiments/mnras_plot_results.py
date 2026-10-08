#!/usr/bin/env python3
"""Reproduce scientific diagnostic panels from saved evaluation CSVs (English)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path)
    a=p.parse_args(); out=a.out_dir or a.evaluation_dir/'figures';out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.evaluation_dir/'family_matrix_predictions.csv')
    # One explicit protocol and seed; no repeated seed points treated as new data.
    contract='full_observed' if 'full_observed' in set(d.contract) else 'geometry_followup'
    s=d[(d.contract==contract)&(d.protocol=='mixed_family')&(d.domain=='benchmark')&(d.seed==d.seed.min())]
    colors={'SIS':'#355f9d','SIE':'#319070','SIS_shear':'#bc7847','NFW_like':'#a34854'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    tasks=['boundary','uncertainty','ambiguity','selective','hierarchy']
    for task in tqdm(tasks,desc='Result panels',unit='figure'):
        if task=='boundary':
            fig,axes=plt.subplots(1,3,figsize=(11,3.3))
            for ax,x,label in zip(axes,['abs_detA','smin','condition'],[r'$|\det A|$',r'$s_{\min}(A)$',r'$\mathrm{cond}(A)$']):
                for fam,g in s.groupby('test_family'):
                    ax.scatter(g[x],np.maximum(g.abs_error,1e-10),s=5,alpha=.22,color=colors[fam],rasterized=True,label=fam)
                ax.set(xscale='log',yscale='log',xlabel=label,ylabel=r'$|\hat\mu-\mu|$')
            axes[-1].legend(fontsize=7);save(fig,'jacobian_error_boundary')
        elif task=='uncertainty':
            fig,ax=plt.subplots(figsize=(4.6,3.5))
            for fam,g in s.groupby('test_family'):
                valid=g[np.isfinite(g.conformal_width)]
                ax.scatter(valid.abs_detA,valid.conformal_width,s=5,alpha=.25,label=fam,color=colors[fam],rasterized=True)
            ax.set(xscale='log',yscale='log',xlabel=r'$|\det A|$',ylabel='Calibrated interval width');ax.legend(fontsize=7);save(fig,'jacobian_uncertainty')
        elif task=='ambiguity':
            pairfiles=list(a.evaluation_dir.glob('ambiguity_pairs_*.csv'))
            if not pairfiles:continue
            pairs=pd.concat([pd.read_csv(f) for f in pairfiles],ignore_index=True)
            pairs=pairs[(pairs.contract==contract)&(pairs.scaler=='zscore')]
            summary=pd.read_csv(a.evaluation_dir/'ambiguity_map_summary.csv')
            summary=summary[(summary.contract==contract)&(summary.scaler=='zscore')]
            fig,axes=plt.subplots(1,3,figsize=(11,3.4))
            for ax,(fa,fb) in zip(axes,[('SIE','SIS_shear'),('NFW_like','SIE'),('NFW_like','SIS_shear')]):
                g=pairs[((pairs.family_a==fa)&(pairs.family_b==fb))|((pairs.family_a==fb)&(pairs.family_b==fa))]
                if not len(g):continue
                sc=ax.scatter(g.observable_distance,g.abs_delta_mu,c=np.log10(np.maximum(np.minimum(g.abs_detA_a,g.abs_detA_b),1e-12)),s=8,alpha=.5,cmap='viridis',rasterized=True)
                ax.set(xlabel='Standardized observable distance',ylabel=r'$|\Delta\mu|$',title=fa+' / '+fb)
                band=summary[((summary.family_a==fa)&(summary.family_b==fb))|((summary.family_a==fb)&(summary.family_b==fa))].sort_values('distance_lo')
                centers=(band.distance_lo+band.distance_hi)/2
                ax.plot(centers,band.median_abs_delta_mu,color='black',lw=1)
                ax.fill_between(centers,band.median_ci_lo,band.median_ci_hi,color='black',alpha=.15)
                ax.set_yscale('symlog',linthresh=.1)
            save(fig,'cross_family_ambiguity')
            fig,axes=plt.subplots(1,3,figsize=(11,3.4))
            for ax,(fa,fb) in zip(axes,[('SIE','SIS_shear'),('NFW_like','SIE'),('NFW_like','SIS_shear')]):
                g=pairs[((pairs.family_a==fa)&(pairs.family_b==fb))|((pairs.family_a==fb)&(pairs.family_b==fa))]
                ax.scatter(g.observable_distance,g.mu_ratio,s=7,alpha=.4,rasterized=True)
                ax.axhline(2.,ls='--',color='#a34854',label='Magnification ratio = 2')
                ax.set(xlabel='Standardized observable distance',ylabel='Magnification ratio',yscale='log',title=fa+' / '+fb)
            axes[0].legend(fontsize=7);save(fig,'ambiguity_ratio_threshold')
        elif task=='selective':
            sel=pd.read_csv(a.evaluation_dir/'selective_risk.csv');sel=sel[(sel.contract==contract)&(sel.protocol=='mixed_family')&(sel.domain=='benchmark')&sel.requested_retained_fraction.notna()&~sel.deployment_uses_true_latent]
            fig,axes=plt.subplots(1,2,figsize=(8,3.4))
            for score,g in sel.groupby('selection_score'):
                z=g.groupby('requested_retained_fraction')[['mae','coverage90']].mean()
                axes[0].plot(z.index,z.mae,label=score);axes[1].plot(z.index,z.coverage90,label=score)
            axes[0].set(xlabel='Retained fraction',ylabel='MAE (family mean)');axes[1].set(xlabel='Retained fraction',ylabel='Interval coverage');axes[1].axhline(.9,color='.5',ls='--');axes[0].legend(fontsize=6);save(fig,'selective_prediction')
        elif task=='hierarchy':
            met=pd.read_csv(a.evaluation_dir/'family_matrix_metrics.csv');met=met[(met.protocol=='single_family')&(met.train_family==met.test_family)&(met.domain=='benchmark')]
            met=met[met.contract.isin(['gw_only','gw_followup','full_observed'])]
            if met.empty:continue
            tab=met.pivot_table(index='test_family',columns='contract',values='median_relative_error',aggfunc='mean');fig,ax=plt.subplots(figsize=(6,3.5));tab.plot.bar(ax=ax);ax.set(ylabel='Median relative error',xlabel='Lens family');ax.tick_params(axis='x',rotation=0);save(fig,'information_hierarchy')
    print(f'FIGURES={out.resolve()}')

if __name__=='__main__':main()

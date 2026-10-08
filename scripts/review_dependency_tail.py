"""Identity-calibration and controlled SIS tail diagnostics; user-run experiments."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "experiments/extended_experiments"
sys.path.insert(0, str(LEGACY))
from evaluate_universal_identifiability import GEOMETRY, FAMILIES, load_benchmark, FittedModel, conformal_quantile
SPLITS = [11,23,37,42,59,71,89,101,131,173]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str)+"\n")


def summarize(p, bootstrap, seed):
    """Event-weighted metrics, identity-resampled uncertainty and cluster coverage."""
    e = p.pred.to_numpy()-p.truth.to_numpy()
    cover = (p.truth >= p.lo) & (p.truth <= p.hi)
    z = pd.DataFrame(dict(id=p.identity_key.to_numpy(), abs=np.abs(e), squared=e**2,
                          bias=e, covered=cover.to_numpy(float), width=(p.hi-p.lo).to_numpy()))
    grouped = z.groupby("id")
    sums = grouped[["abs","squared","bias","covered","width"]].sum().to_numpy()
    counts = grouped.size().to_numpy()
    def values(ix):
        v = sums[ix].sum(axis=0)/counts[ix].sum()
        v[1] = np.sqrt(v[1])
        return v
    fields = ["mae","rmse","bias","coverage90","mean_width"]
    row = dict(n_rows=len(p), n_identities=len(sums), status="sparse_diagnostic" if len(sums)<10 else "ok")
    row.update(zip(fields, values(np.arange(len(sums)))))
    row.update(abs_error_p50=np.quantile(abs(e),.5), abs_error_p90=np.quantile(abs(e),.9),
               abs_error_p95=np.quantile(abs(e),.95), signed_error_p10=np.quantile(e,.1),
               signed_error_p90=np.quantile(e,.9), equal_identity_coverage=grouped.covered.mean().mean(),
               simultaneous_identity_coverage=grouped.covered.min().mean())
    if len(sums)>1 and bootstrap:
        rng = np.random.default_rng(seed)
        b = np.array([values(rng.integers(0,len(sums),len(sums))) for _ in range(bootstrap)])
        for name, limits in zip(fields,np.quantile(b,[.025,.975],axis=0).T):
            row[name+"_lo95"],row[name+"_hi95"] = limits
    return row


def representative(frame, seed):
    # Selection uses identity and event keys only, never labels or scores.
    out = frame.copy()
    out["selection_hash"] = [hashlib.sha256(f"{seed}:{i}:{e}".encode()).hexdigest()
                             for i,e in zip(out.identity_key,out.event_key)]
    return out.sort_values("selection_hash").drop_duplicates("identity_key").drop(columns="selection_hash")


def cluster_run(a):
    rows, audits, predictions = [], [], []
    for generation in a.generations:
        root = a.benchmark_parent / f"seed_{generation}/benchmark"
        data = load_benchmark(root)
        for omitted in [None,"NFW_like"]:
            pool = data if omitted is None else data[data.lens_family != omitted]
            tr,cal = [pool[pool.split == s].copy() for s in ["train","calibration"]]
            te = data[data.split == "test"].copy()
            for seed in a.fit_seeds:
                print(f"cluster generation={generation} omitted={omitted} fit={seed}",flush=True)
                fit = FittedModel(tr,cal,GEOMETRY,seed,SimpleNamespace(max_iter=180,ensemble_members=3))
                cr = fit.raw(fit.imputer.transform(cal[GEOMETRY]))
                raw = fit.raw(fit.imputer.transform(te[GEOMETRY]))
                scores = np.maximum.reduce([cr["raw_lo"]-cal.mu0_true,cal.mu0_true-cr["raw_hi"],np.zeros(len(cal))])
                c = cal[["identity_key","event_key"]].copy();c["score"] = scores
                qvalues = {"row":conformal_quantile(scores),
                           "identity_max":conformal_quantile(c.groupby("identity_key").score.max()),
                           "one_per_identity":conformal_quantile(representative(c,8127).score)}
                if not np.isclose(qvalues["identity_max"],fit.q):
                    raise ValueError("Existing cluster calibration was not reproduced")
                for mode,q in qvalues.items():
                    if not np.isfinite(q):
                        raise ValueError("Insufficient calibration units")
                    meta = dict(generation=generation, omitted=omitted or "none", fit_seed=seed, calibration=mode)
                    audits.append(dict(**meta,q=q,calibration_rows=len(cal),calibration_identities=cal.identity_key.nunique(),
                                       min_cluster_size=int(cal.groupby("identity_key").size().min()),
                                       max_cluster_size=int(cal.groupby("identity_key").size().max())))
                    p = te[["identity_key","event_key","lens_family"]].copy()
                    p["truth"],p["pred"] = te.mu0_true.to_numpy(),raw["mu0_pred"]
                    p["lo"],p["hi"] = np.maximum(0,raw["raw_lo"]-q),raw["raw_hi"]+q
                    predictions.append(p.assign(**meta))
                    for scope,g in [("all_families",p)]+list(p.groupby("lens_family")):
                        # Full test clusters and the fixed representative test sample have distinct targets.
                        for population,h in [("all_realizations",g),("one_per_identity",representative(g,9127))]:
                            rows.append(dict(**meta,test_family=scope,test_population=population,
                                             **summarize(h,a.bootstrap,seed)))
                pd.DataFrame(rows).to_csv(a.output/"metrics.csv",index=False)
    pd.concat(predictions).to_csv(a.output/"predictions.csv",index=False)
    pd.DataFrame(audits).to_csv(a.output/"calibration_audit.csv",index=False)
    write_json(a.output/"design.json",dict(arguments=vars(a),
        existing_method="identity maximum, confirmed against existing FittedModel.q",
        guarantee="identity-max addresses simultaneous coverage for a new exchangeable cluster, conditional on fitted model; cluster size/composition generation must also be exchangeable. One representative targets its prespecified sampling rule. Row calibration has no general dependent-row guarantee. Unseen families have no guarantee.",
        bootstrap="prior identity resampling, does not establish exchangeability"))


def sis_load(root, prefix):
    obs,lens,params = [pd.read_csv(root/n) for n in ["observable_features.csv","lens.csv","lens_params.csv"]]
    for d in [obs,lens,params]:
        if d.event_id.duplicated().any(): raise ValueError("Duplicate SIS identities")
    if not set(obs.event_id)==set(lens.event_id)==set(params.event_id): raise ValueError("Catalogue ID mismatch")
    d = obs.rename(columns={c:"obs__"+c for c in obs if c!="event_id"}).merge(lens[["event_id","mu_0","y"]],on="event_id",validate="one_to_one")
    d["identity_key"] = prefix+d.event_id.astype(str);d["event_key"] = d.identity_key
    return d,params.set_index("event_id").loc[d.event_id].reset_index()


def reduced_noise(d, ref, factor):
    out = d.copy()
    for c in ["image_x_0","image_y_0","image_x_1","image_y_1","lens_center_x","lens_center_y"]:
        obs = "obs__"+c+"_observed"
        out[obs] = ref[c].to_numpy()+factor*(d[obs].to_numpy()-ref[c].to_numpy())
    r = []
    for i in [0,1]:
        r.append(np.hypot(out[f"obs__image_x_{i}_observed"]-out.obs__lens_center_x_observed,
                          out[f"obs__image_y_{i}_observed"]-out.obs__lens_center_y_observed))
    sep = r[0]+r[1]
    for c,v in dict(theta_plus_abs_observed=r[0],theta_minus_abs_observed=r[1],image_separation_observed=sep,
                    image_position_asymmetry_observed=np.clip((r[0]-r[1])/np.maximum(sep,1e-8),1e-4,.999)).items():
        out["obs__"+c] = v
    return out


def sis_fit(tr,cal,cols,target,weight):
    model = make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingRegressor(
        max_iter=180,max_leaf_nodes=15,min_samples_leaf=10,learning_rate=.06,l2_regularization=.1,
        early_stopping=False,random_state=6130))
    y = np.log(tr.mu_0-1) if target=="log" else tr.mu_0
    w = np.where(tr.y<=.2,weight,1.)
    model.fit(tr[cols],y,histgradientboostingregressor__sample_weight=w)
    def predict(d):
        p = model.predict(d[cols])
        return 1+np.exp(np.clip(p,-30,30)) if target=="log" else np.maximum(1.,p)
    q = conformal_quantile(abs(predict(cal)-cal.mu_0.to_numpy()))
    return predict,q


def tail_run(a):
    d,ref = sis_load(a.data_root,"original:")
    tail,tref = sis_load(a.tail_root,"external:")
    if not tail.y.between(.05,.1).all(): raise ValueError("Expected independent .05<=y<=.1 tail")
    original_config = json.loads((a.data_root/"manifest.json").read_text())
    tail_config = json.loads((a.tail_root/"manifest.json").read_text())
    allowed = {"seed","save_dir","target_events","max_attempts","overwrite","y_min","y_max"}
    if original_config["generator"] != tail_config["generator"] or original_config["config"]["seed"] == tail_config["config"]["seed"]:
        raise ValueError("Independent tail must use same generator and a distinct seed")
    changed = {k for k,v in tail_config["config"].items() if k not in allowed and v != original_config["config"].get(k)}
    if changed: raise ValueError(f"Tail selection/generation settings differ: {changed}")
    if len(tail)!=tail_config["config"]["target_events"]: raise ValueError("Incomplete independent tail")
    signature = ["image_x_0","image_y_0","image_x_1","image_y_1","z_s_true","z_l_true"]
    if set(map(tuple,ref[signature].to_numpy())) & set(map(tuple,tref[signature].to_numpy())):
        raise ValueError("External lenses overlap original catalogue")
    manifest = pd.read_csv(a.protocol_root/"split_manifest.csv")
    reference = pd.read_csv(a.protocol_root/"predictions.csv").query("contract == 'geometry'")
    reserve_order = np.random.default_rng(71007).permutation(len(tail))
    if not 0 < a.enrichment_count < len(tail): raise ValueError("Enrichment reserve must leave independent test lenses")
    reserve_ids = set(tail.iloc[reserve_order[:a.enrichment_count]].event_id)
    scenarios = [("saved_nominal",None,1.,"log"),("paired_nominal",1.,1.,"log"),
                 ("noise_half",.5,1.,"log"),("truth_geometry",0.,1.,"log"),
                 ("tail_weight_5",None,5.,"log"),("tail_weight_20",None,20.,"log"),
                 ("direct_target",None,1.,"direct"),("tail_add_independent",None,1.,"log")]
    rows,predictions,counts = [],[],[]
    for seed in a.split_seeds:
        m = manifest[manifest.split_seed==seed]
        if set(m.event_id)!=set(d.event_id) or m.event_id.duplicated().any(): raise ValueError("Invalid primary manifest")
        for name,factor,weight,target in scenarios:
            print(f"tail split={seed} scenario={name}",flush=True)
            base = d.copy() if factor is None else reduced_noise(d,ref,factor)
            ext = tail.copy() if factor is None else reduced_noise(tail,tref,factor)
            base = base.merge(m[["event_id","split"]],on="event_id",validate="one_to_one")
            tr,cal,te = [base[base.split==s] for s in ["train","calibration","test"]]
            if (len(tr),len(cal),len(te))!=(1750,375,375): raise ValueError("Incorrect primary partition sizes")
            reserved = ext[ext.event_id.isin(reserve_ids)]
            common_tail_test = ext[~ext.event_id.isin(reserve_ids)]
            if name=="tail_add_independent":
                tr = pd.concat([tr,reserved],ignore_index=True)
            fn,q = sis_fit(tr,cal,GEOMETRY,target,weight)
            counts.append(dict(split_seed=seed,scenario=name,n_train=len(tr),n_train_y_le_0p1=int((tr.y<=.1).sum()),
                               n_train_y_le_0p2=int((tr.y<=.2).sum()),tail_sample_weight=weight,
                               weighted_tail_fraction=float(np.where(tr.y<=.2,weight,0).sum()/np.where(tr.y<=.2,weight,1).sum()),q=q))
            tests = [("natural_test",te),("independent_low_y_common_holdout",common_tail_test)]
            if name != "tail_add_independent": tests.append(("independent_low_y_full",ext))
            for sample,test in tests:
                p = test[["identity_key","event_key","event_id","y"]].copy()
                p["truth"],p["pred"] = test.mu_0.to_numpy(),fn(test)
                p["lo"],p["hi"] = np.maximum(1,p.pred-q),p.pred+q
                if name=="saved_nominal" and sample=="natural_test":
                    saved = reference[reference.split_seed==seed].set_index("event_id").loc[test.event_id]
                    if not np.allclose(p.pred,saved.mu0_pred,atol=1e-8): raise ValueError("Primary HistGB reproduction failed")
                predictions.append(p.assign(split_seed=seed,scenario=name,sample=sample))
                for subset,g in [("all",p),("y_le_0p2",p[p.y<=.2]),("y_le_0p1",p[p.y<=.1])]:
                    if len(g): rows.append(dict(split_seed=seed,scenario=name,sample=sample,subset=subset,
                                               **summarize(g,a.bootstrap,seed)))
            pd.DataFrame(rows).to_csv(a.output/"metrics.csv",index=False)
            pd.DataFrame(counts).to_csv(a.output/"training_tail_counts.csv",index=False)
    pd.concat(predictions).to_csv(a.output/"predictions.csv",index=False)
    pd.DataFrame(rows).groupby(["scenario","sample","subset"])[["mae","rmse","bias","coverage90","mean_width","abs_error_p90","abs_error_p95"]].agg(["mean","std"]).to_csv(a.output/"summary.csv")
    write_json(a.output/"design.json",dict(arguments=vars(a),
        observation="noise scaled using identical saved residuals; all geometry derived again; oracle truth is diagnostic only",
        training="tail weights retain original identities; independent enrichment adds a prespecified low-y reserve, evaluated only on its disjoint common holdout. Calibration remains original natural population.",
        enrichment_train_ids=sorted(reserve_ids),
        limitation="not a causal variance decomposition; true-geometry tree still has approximation error. Full 1000-lens evaluation excludes enriched model. Compare enrichment on common holdout only. Direct-target changes training loss as well as parameterization.",
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in [a.data_root,a.tail_root] for p in [root/"lens.csv",root/"lens_params.csv",root/"observable_features.csv"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment",choices=["cluster","tail"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--enrichment-count",type=int,default=500)
    p.add_argument("--generations",type=int,nargs="+",default=[20260922,20260923,20260924])
    p.add_argument("--fit-seeds",type=int,nargs="+",default=[6130,6131,6132])
    p.add_argument("--split-seeds",type=int,nargs="+",default=SPLITS)
    p.add_argument("--benchmark-parent",type=Path,default=LEGACY/"production_MNRAS_experiments_v2")
    p.add_argument("--data-root",type=Path,default=ROOT/"data_generation/data_lens_sis_gw_physics_baseline")
    p.add_argument("--tail-root",type=Path,default=ROOT.parent/"review_followup_20261006/data/sis_tail_selected_seed20261006")
    p.add_argument("--protocol-root",type=Path,default=ROOT.parent/"review_followup_20261006/results_main/contracts")
    a = p.parse_args()
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        p.error(f"Output path contains existing data: {a.output}. Use a new --output directory; existing results are preserved.")
    a.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=a.threads):
        (cluster_run if a.experiment=="cluster" else tail_run)(a)


if __name__=="__main__": main()

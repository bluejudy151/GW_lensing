"""Coordinate-likelihood SIS baseline on the manuscript's primary identity splits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.special import i0e, logsumexp

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 23, 37, 42, 59, 71, 89, 101, 131, 173]


def posterior(obs, y, theta, logprior, image_sigma, centre_sigma):
    # d and m have independent Gaussian noise; integrate uniform direction exactly.
    p0 = obs[["image_x_0_observed", "image_y_0_observed"]].to_numpy(float)
    p1 = obs[["image_x_1_observed", "image_y_1_observed"]].to_numpy(float)
    centre = obs[["lens_center_x_observed", "lens_center_y_observed"]].to_numpy(float)
    d = (p0 - p1) / 2
    m = (p0 + p1) / 2 - centre
    vd = image_sigma**2 / 2
    vm = vd + centre_sigma**2
    yy = y[:, None]
    prior_interp = RegularGridInterpolator((y, np.log(theta)), np.exp(logprior), bounds_error=False, fill_value=0.)
    result = []
    for k, (di, mi) in enumerate(zip(d, m)):
        # Local linear theta quadrature resolves astrometric likelihood at large scales.
        radius = np.linalg.norm(di)
        lower = max(theta[0], radius-10*np.sqrt(vd))
        upper = min(theta[-1], radius+10*np.sqrt(vd))
        if upper <= lower:
            raise ValueError("Observed separation outside training prior support; widen prior support")
        local_theta = np.linspace(lower, upper, len(theta))
        tt = local_theta[None, :]
        grid_y, grid_logt = np.meshgrid(y, np.log(local_theta), indexing="ij")
        density = prior_interp(np.stack([grid_y, grid_logt], axis=-1)) / tt
        with np.errstate(divide="ignore"):
            local_prior = np.log(density)
        # Trapezoid weights in theta; uniform y cell weights cancel.
        local_prior[:, [0, -1]] -= np.log(2.)
        base = local_prior - tt**2 / (2 * vd) - (tt * yy)**2 / (2 * vm)
        vec = di[None, :] / vd + y[:, None] * mi[None, :] / vm
        z = tt * np.linalg.norm(vec, axis=1)[:, None]
        logw = base + np.log(i0e(z)) + z
        norm = logsumexp(logw)
        if not np.isfinite(norm):
            raise ValueError("Posterior normalization failed")
        py = np.exp(logsumexp(logw, axis=1) - norm)
        cdf = np.cumsum(py)
        dy = y[1]-y[0]
        quant = np.interp([.05, .5, .95], np.r_[0., cdf], np.r_[y[0]-dy/2, y+dy/2])
        result.append([1 + 1 / quant[1], 1 + 1 / quant[2], 1 + 1 / quant[0]])
        if (k + 1) % 250 == 0:
            print(f"  posterior {k+1}/{len(obs)}", flush=True)
    return np.asarray(result)


def selected_prior(train, ny, nt, bandwidth):
    # KDE on accepted training truth only, never calibration/test truth.
    yedges = np.linspace(.05, .98, ny + 1)
    logt = np.log(train["theta_E(arcsec)"].to_numpy())
    hy = bandwidth * train.y.std(ddof=1) * len(train)**(-1/6)
    ht = bandwidth * logt.std(ddof=1) * len(train)**(-1/6)
    tedges = np.linspace(logt.min() - 5 * ht, logt.max() + 5 * ht, nt + 1)
    hist, _, _ = np.histogram2d(train.y, logt, bins=[yedges, tedges])
    prior = gaussian_filter(hist, [hy / np.diff(yedges)[0], ht / np.diff(tedges)[0]], mode="constant")
    prior /= prior.sum()
    with np.errstate(divide="ignore"):
        logprior = np.log(prior)
    return (yedges[:-1] + yedges[1:]) / 2, np.exp((tedges[:-1] + tedges[1:]) / 2), logprior, dict(
        y_bandwidth=hy, logtheta_bandwidth=ht, theta_min=float(np.exp(tedges[0])), theta_max=float(np.exp(tedges[-1])))


def metric_values(truth, pred, lo, hi):
    e = pred - truth
    return np.array([np.abs(e).mean(), np.sqrt((e**2).mean()), e.mean(),
                     (np.abs(e)/truth).mean(), ((truth >= lo) & (truth <= hi)).mean(), (hi-lo).mean()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_generation/data_lens_sis_gw_physics_baseline")
    parser.add_argument("--protocol-root", type=Path, default=ROOT.parent / "review_followup_20261006/results_main/contracts")
    parser.add_argument("--output", type=Path, default=ROOT / "results/sis_bayesian_primary")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--y-grid", type=int, default=512)
    parser.add_argument("--theta-grid", type=int, default=512)
    parser.add_argument("--bandwidth-scale", type=float, default=1.)
    parser.add_argument("--image-sigma", type=float, default=.01)
    parser.add_argument("--centre-sigma", type=float, default=.02)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.image_sigma, args.centre_sigma, args.bandwidth_scale) <= 0 or min(args.y_grid, args.theta_grid) < 32:
        parser.error("Noise/bandwidth must be positive; each grid must have at least 32 cells")
    paths = [args.data_root / x for x in ["observable_features.csv", "lens.csv", "lens_params.csv"]]
    paths += [args.protocol_root / "split_manifest.csv", args.protocol_root / "predictions.csv"]
    obs, lens, params, manifest, reference = [pd.read_csv(p) for p in paths]
    for frame in [obs, lens, params]:
        if frame.event_id.duplicated().any():
            raise ValueError("Duplicate event IDs")
    if not set(obs.event_id) == set(lens.event_id) == set(params.event_id):
        raise ValueError("Catalogue ID sets differ")
    if manifest.duplicated(["split_seed", "event_id"]).any():
        raise ValueError("Duplicate manifest identities")
    frame = obs.merge(lens[["event_id", "mu_0", "y"]], on="event_id", validate="one_to_one")
    frame = frame.merge(params[["event_id", "theta_E(arcsec)"]], on="event_id", validate="one_to_one")
    reference = reference.query("contract == 'geometry'")
    for seed in args.split_seeds:
        split = manifest[manifest.split_seed == seed]
        if set(split.event_id) != set(frame.event_id) or split.split.value_counts().to_dict() != dict(train=1750, calibration=375, test=375):
            raise ValueError(f"Invalid primary split {seed}")
        ref = reference[reference.split_seed == seed]
        if set(ref.event_id) != set(split.loc[split.split == "test", "event_id"]):
            raise ValueError("Reference test identities differ")
        merged = frame.merge(ref[["event_id", "mu0_true"]], on="event_id", validate="one_to_one")
        if not np.allclose(merged.mu_0, merged.mu0_true, rtol=1e-12):
            raise ValueError("Reference truth differs from input catalogue")
    print("Validated catalogue and primary splits; inference uses six observed coordinates only.", flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    predictions, metrics, paired, prior_records = [], [], [], []
    names = ["mae", "rmse", "bias", "mean_relative_error", "coverage90", "mean_width"]
    for seed in args.split_seeds:
        print(f"Split {seed}", flush=True)
        split = frame.merge(manifest.loc[manifest.split_seed == seed, ["event_id", "split"]], on="event_id", validate="one_to_one")
        train = split[split.split == "train"]
        evaluation = split[split.split != "train"].reset_index(drop=True)
        y, theta, lp, prior_info = selected_prior(train, args.y_grid, args.theta_grid, args.bandwidth_scale)
        prior_records.append(dict(split_seed=seed, **prior_info))
        values = posterior(evaluation, y, theta, lp, args.image_sigma, args.centre_sigma)
        evaluation[["pred", "raw_lo", "raw_hi"]] = values
        cal = evaluation[evaluation.split == "calibration"]
        scores = np.maximum.reduce([cal.raw_lo-cal.mu_0, cal.mu_0-cal.raw_hi, np.zeros(len(cal))])
        rank = int(np.ceil((len(cal)+1)*.9))
        q = float(np.sort(scores)[rank-1])
        test = evaluation[evaluation.split == "test"].copy()
        ref = reference[reference.split_seed == seed].set_index("event_id").loc[test.event_id]
        for method in ["sis_bayes_raw", "sis_bayes_conformal", "histgb_geometry"]:
            p = test[["event_id", "mu_0", "y"]].rename(columns={"mu_0":"mu0_true", "y":"y_true"}).copy()
            if method == "histgb_geometry":
                p["mu0_pred"], p["lo"], p["hi"] = ref.mu0_pred.to_numpy(), ref.lo.to_numpy(), ref.hi.to_numpy()
            else:
                expand = q if method.endswith("conformal") else 0.
                p["mu0_pred"] = test.pred.to_numpy()
                p["lo"] = np.maximum(1., test.raw_lo.to_numpy()-expand)
                p["hi"] = test.raw_hi.to_numpy()+expand
            p["split_seed"], p["method"] = seed, method
            predictions.append(p)
            for label, mask in [("all", np.ones(len(p), bool)), ("y_le_0p2", p.y_true <= .2), ("y_le_0p1", p.y_true <= .1)]:
                g = p[mask]
                if not len(g):
                    continue
                arrays = [g[c].to_numpy() for c in ["mu0_true", "mu0_pred", "lo", "hi"]]
                vals = metric_values(*arrays)
                row = dict(split_seed=seed, method=method, subset=label, n=len(g),
                           status="sparse_diagnostic" if len(g)<10 else "ok", conformal_q=q if method == "sis_bayes_conformal" else None)
                row.update(zip(names, vals))
                rng = np.random.default_rng(seed)
                if len(g) >= 2 and args.bootstrap > 0:
                    boot = np.array([metric_values(*[a[ix] for a in arrays]) for ix in rng.integers(0, len(g), (args.bootstrap, len(g)))])
                    for name, limits in zip(names, np.quantile(boot, [.025, .975], axis=0).T):
                        row[name+"_lo95"], row[name+"_hi95"] = limits
                metrics.append(row)
        delta = np.abs(test.pred.to_numpy()-test.mu_0.to_numpy()) - np.abs(ref.mu0_pred.to_numpy()-test.mu_0.to_numpy())
        for label, mask in [("all", np.ones(len(test), bool)), ("y_le_0p2", test.y <= .2), ("y_le_0p1", test.y <= .1)]:
            v = delta[np.asarray(mask)]
            if not len(v):
                continue
            row = dict(split_seed=seed, subset=label, n=len(v), bayes_minus_histgb_mae=v.mean())
            if len(v) >= 2 and args.bootstrap > 0:
                rng = np.random.default_rng(seed)
                b = v[rng.integers(0, len(v), (args.bootstrap, len(v)))].mean(axis=1)
                row["lo95"], row["hi95"] = np.quantile(b, [.025, .975])
            paired.append(row)
        pd.concat(predictions).to_csv(args.output / "predictions.csv", index=False)
        pd.DataFrame(metrics).to_csv(args.output / "metrics_by_split.csv", index=False)
        pd.DataFrame(paired).to_csv(args.output / "paired_mae.csv", index=False)
    pd.DataFrame(metrics).groupby(["method", "subset"])[names].agg(["mean", "std"]).to_csv(args.output / "summary.csv")
    manifest[manifest.split_seed.isin(args.split_seeds)].to_csv(args.output / "split_manifest.csv", index=False)
    (args.output / "provenance.json").write_text(json.dumps(dict(
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        prior="training-only joint KDE of y and log Einstein angle in accepted selected population",
        point="posterior median magnification (absolute-loss Bayes estimate)",
        intervals="central 90% posterior; separate split-conformal expansion, calibration only",
        coordinates="translation invariant; integrate direction uniformly, do not use true centre=0",
        scope="pure SIS, correct image identities, known Gaussian noise; no mass sheet",
        prior_bandwidths=prior_records), indent=2)+"\n")
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()

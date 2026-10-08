#!/usr/bin/env python
# coding: utf-8
"""Balanced exact local-Jacobian lens catalogue (geometry/follow-up only).

The old SIS catalogue supplies *true* lens-scale/redshift priors only. Every
lens/source configuration is newly ray-traced. No old waveform is copied.
Angles are arcsec, Fermat potentials arcsec^2, time delays seconds.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from tqdm.auto import tqdm

FAMILIES = ['SIS', 'SIE', 'SIS_shear', 'NFW_like', 'SIE_shear']


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sis-root', type=Path, required=True)
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument('--out-root', type=Path, help='Create a timestamped subdirectory.')
    out.add_argument('--out-dir', type=Path, help='Create this exact directory; must not exist.')
    p.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES[:4])
    p.add_argument('--n-events', type=int, default=5000)
    p.add_argument('--max-attempts', type=int, default=100000)
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--split-seed', type=int, default=7619)
    p.add_argument('--q-min', type=float, default=.70)
    p.add_argument('--q-max', type=float, default=.98)
    p.add_argument('--shear-min', type=float, default=0.)
    p.add_argument('--shear-max', type=float, default=.12)
    p.add_argument('--source-y-min', type=float, default=.05)
    p.add_argument('--source-y-max', type=float, default=.95)
    p.add_argument('--nfw-rs-ratio', type=float, default=5.)
    p.add_argument('--nfw-rs-min', type=float, default=None)
    p.add_argument('--nfw-rs-max', type=float, default=None)
    p.add_argument('--nfw-alpha-ratio', type=float, default=8.)
    p.add_argument('--astrometry-sigma', type=float, default=.01)
    p.add_argument('--lens-center-sigma', type=float, default=.02)
    p.add_argument('--redshift-sigma', type=float, default=5e-4)
    p.add_argument('--sigma-v-fraction', type=float, default=.05)
    p.add_argument('--theta-e-fraction', type=float, default=.02)
    p.add_argument('--critical-angles', type=int, default=96,
                   help='Coarse angular mesh, refined to twice this for critical-curve distance.')
    p.add_argument('--validation-rtol', type=float, default=.01,
                   help='Fail if independent finite-difference magnification check exceeds this.')
    a = p.parse_args()
    if a.n_events < 3 or a.max_attempts < a.n_events:
        p.error('n-events must be >=3; max-attempts must be >=n-events')
    if not 0 < a.source_y_min < a.source_y_max < 1:
        p.error('Require 0 < source-y-min < source-y-max < 1.')
    if not 0 < a.q_min <= a.q_max <= 1 or not 0 <= a.shear_min <= a.shear_max < .5:
        p.error('Invalid q/shear ranges.')
    if (a.nfw_rs_min is None) != (a.nfw_rs_max is None):
        p.error('nfw-rs-min and nfw-rs-max must be supplied together.')
    if a.nfw_rs_min is None:
        a.nfw_rs_min = a.nfw_rs_max = a.nfw_rs_ratio
    if not 0 < a.nfw_rs_min <= a.nfw_rs_max or a.nfw_alpha_ratio <= 0:
        p.error('Invalid NFW radial/deflection scale.')
    if a.critical_angles < 32 or a.validation_rtol <= 0:
        p.error('critical-angles >=32 and validation-rtol >0 required.')
    return a


def load_sis_prior(root):
    d = pd.read_csv(root / 'lens_params.csv')
    aliases = {'z_l_true': ['z_l_true', 'z_l'], 'z_s_true': ['z_s_true', 'z_s'],
               'sigma_v_true': ['sigma_v_true', 'sigma_v'],
               'theta_E_true': ['theta_E_true', 'theta_E(arcsec)']}
    selected = {}
    for target, choices in aliases.items():
        found = next((c for c in choices if c in d), None)
        if found is None:
            raise ValueError(f'Truth prior missing {choices}; observed columns are never substituted.')
        selected[target] = found
        d[target] = pd.to_numeric(d[found], errors='raise')
    if 'event_id' not in d or d.event_id.duplicated().any():
        raise ValueError('Prior event_id missing or nonunique.')
    if len(d) < 3:
        raise ValueError('Need at least three independent prior rows.')
    valid = ((d.z_l_true > 0) & (d.z_s_true > d.z_l_true) &
             (d.theta_E_true > 0) & (d.sigma_v_true > 0))
    if not valid.all() or not np.isfinite(d[list(aliases)].to_numpy()).all():
        raise ValueError('Invalid physical prior row; fix source catalogue instead of silently discarding.')
    return d, selected


def group_partition(prior, split_seed):
    """All realizations sharing a prior row retain the same fixed split."""
    ids = sorted(prior.event_id.tolist(), key=lambda x: hashlib.sha256(
        f'{split_seed}:{x}'.encode()).digest())
    n = len(ids)
    ntr = max(1, int(.70*n)); nv = max(1, int(.15*n))
    ntr = min(ntr, n-2); nv = min(nv, n-ntr-1)
    return {v: ('train' if i < ntr else 'validation' if i < ntr+nv else 'test')
            for i, v in enumerate(ids)}


def split_counts(n):
    nt = max(1, int(.70*n)); nv = max(1, int(.15*n))
    nt = min(nt, n-2); nv = min(nv, n-nt-1)
    return {'train': nt, 'validation': nv, 'test': n-nt-nv}


def model_for(family, theta, rng, a):
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.Util.param_util import phi_q2_ellipticity
    meta = {'q_true': 1., 'external_shear_true': 0., 'gamma_true': 0.,
            'lens_phi_true': 0., 'shear_phi_true': 0.,
            'nfw_rs_over_theta_e': np.nan, 'nfw_alpha_rs_over_theta_e': np.nan}
    if family == 'NFW_like':
        rr = float(rng.uniform(a.nfw_rs_min, a.nfw_rs_max))
        kw = [{'Rs': theta*rr, 'alpha_Rs': theta*a.nfw_alpha_ratio,
               'center_x': 0., 'center_y': 0.}]
        meta.update(nfw_rs_over_theta_e=rr, nfw_alpha_rs_over_theta_e=a.nfw_alpha_ratio)
        return LensModel(['NFW']), kw, meta
    if family in ['SIE', 'SIE_shear']:
        q, angle = float(rng.uniform(a.q_min, a.q_max)), float(rng.uniform(0, np.pi))
        e1, e2 = phi_q2_ellipticity(angle, q)
        names, kw = ['SIE'], [{'theta_E': theta, 'e1': float(e1), 'e2': float(e2), 'center_x': 0., 'center_y': 0.}]
        meta.update(q_true=q, lens_phi_true=angle)
    else:
        names, kw = ['SIS'], [{'theta_E': theta, 'center_x': 0., 'center_y': 0.}]
    if family in ['SIS_shear', 'SIE_shear']:
        g, angle = float(rng.uniform(a.shear_min, a.shear_max)), float(rng.uniform(0, np.pi))
        names += ['SHEAR']
        kw += [{'gamma1': float(g*np.cos(2*angle)), 'gamma2': float(g*np.sin(2*angle))}]
        meta.update(external_shear_true=g, gamma_true=g, shear_phi_true=angle)
    return LensModel(names), kw, meta


def bracket_roots(f, grid):
    vals = np.asarray(f(grid), float)
    roots = []
    for i in np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1]*vals[1:] <= 0)):
        r = float(brentq(lambda t: float(f(t)), grid[i], grid[i+1], xtol=1e-12, rtol=1e-12))
        if not roots or abs(r-roots[-1]) > 1e-7 * max(grid[-1], 1):
            roots.append(r)
    return roots


def nfw_images_and_critical(model, kw, theta, beta, phi):
    rs, alpha = kw[0]['Rs'], kw[0]['alpha_Rs']
    grid = np.geomspace(max(theta*1e-5, 1e-7), max(100*theta, 20*rs, 20*alpha), 1500)
    def radial_alpha(r):
        return np.asarray(model.alpha(np.asarray(r), np.zeros_like(np.asarray(r)), kw)[0])
    rt = bracket_roots(lambda r: 1-radial_alpha(r)/r, grid)
    def radial_eigen(r):
        return 1-np.asarray(model.hessian(np.asarray(r), np.zeros_like(np.asarray(r)), kw)[0])
    rr = bracket_roots(radial_eigen, grid)
    # Include turning radii to avoid missing close roots near a radial caustic.
    grid = np.unique(np.concatenate([grid, rr]))
    pairs = []
    for side in [1., -1.]:
        roots = bracket_roots(lambda r: side*(r-radial_alpha(r))-beta, grid)
        pairs.extend([(r, side) for r in roots])
    xy = np.asarray([[side*r*np.cos(phi), side*r*np.sin(phi)] for r, side in pairs])
    return (xy[:, 0], xy[:, 1], np.asarray(rt+rr)) if len(xy) else (np.array([]), np.array([]), np.asarray(rt+rr))


def critical_isothermal(model, kw, theta, n):
    """Polar critical locus for centered degree-one potential + constant shear.

    H_iso scales as 1/r. Its determinant vanishes. Thus det(B-H_iso/r)=0
    has r=tr(adj(B) H_iso)/det(B). No eigenvalue is used as a distance proxy.
    """
    angle = np.arange(2*n)*np.pi/n
    x, y = theta*np.cos(angle), theta*np.sin(angle)
    xx, xy, yx, yy = [np.asarray(v) for v in model.hessian(x, y, kw)]
    g1 = kw[-1].get('gamma1', 0.); g2 = kw[-1].get('gamma2', 0.)
    B = np.array([[1-g1, -g2], [-g2, 1+g1]])
    hxx, hxy, hyx, hyy = (xx-g1)*theta, (xy-g2)*theta, (yx-g2)*theta, (yy+g1)*theta
    radial = (B[1,1]*hxx - B[0,1]*hyx - B[1,0]*hxy + B[0,0]*hyy)/np.linalg.det(B)
    if not np.isfinite(radial).all() or (radial <= 0).any():
        raise RuntimeError('Invalid isothermal critical curve.')
    return np.column_stack([radial*np.cos(angle), radial*np.sin(angle)])


def polygon_distance(point, polygon):
    delta = np.roll(polygon, -1, axis=0)-polygon
    t = np.clip(np.sum((point-polygon)*delta, axis=1)/np.sum(delta*delta, axis=1), 0, 1)
    return float(np.min(np.linalg.norm(polygon+t[:,None]*delta-point, axis=1)))


def finite_difference_A(model, kw, x, y, theta):
    """Fourth-order derivative of ray shooting, separate from lens hessian."""
    radius = float(np.hypot(x, y))
    h = max(min(1e-4*theta, 1e-3*radius), 1e-8*theta)
    def at(step):
        pts = [(x+step,y),(x-step,y),(x+2*step,y),(x-2*step,y),
               (x,y+step),(x,y-step),(x,y+2*step),(x,y-2*step)]
        bx, by = model.ray_shooting(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), kw)
        v = np.column_stack([bx, by])
        return np.column_stack([(8*(v[0]-v[1])-(v[2]-v[3]))/(12*step),
                                (8*(v[4]-v[5])-(v[6]-v[7]))/(12*step)])
    A1, A2 = at(h), at(h/2)
    return A2, float(np.max(np.abs(A2-A1))), h/2


def rows_for(model, kw, family, eid, theta, ys, phi, meta, a):
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    beta = ys*theta; bx, by = beta*np.cos(phi), beta*np.sin(phi)
    critical_r = None
    if family == 'SIS':
        r = np.array([theta+beta, theta-beta]); side = np.array([1., -1.])
        xs, yy = side*r*np.cos(phi), side*r*np.sin(phi)
    elif family == 'NFW_like':
        xs, yy, critical_r = nfw_images_and_critical(model, kw, theta, beta, phi)
    else:
        xs, yy = LensEquationSolver(model).image_position_from_source(
            bx, by, kw, solver='analytical', arrival_time_sort=False, magnification_limit=None)
        xs, yy = np.asarray(xs), np.asarray(yy)
    if len(xs) < 2:
        return None, 'fewer_than_two_images'
    rayx, rayy = model.ray_shooting(xs, yy, kw)
    residual = np.hypot(np.asarray(rayx)-bx, np.asarray(rayy)-by)
    if not np.isfinite(residual).all() or (residual > max(1e-7*theta, 1e-8)).any():
        raise RuntimeError(f'{family} lens solver residual exceeded tolerance: {residual}')
    fxx, fxy, fyx, fyy = [np.asarray(v) for v in model.hessian(xs, yy, kw)]
    ray_mu = np.asarray(model.magnification(xs, yy, kw))
    fermat = np.asarray(model.fermat_potential(xs, yy, kw, x_source=bx, y_source=by))
    critical_poly = critical_isothermal(model, kw, theta, a.critical_angles) if family not in ['SIS', 'NFW_like'] else None
    rows = []
    for i, (x, y) in enumerate(zip(xs, yy)):
        H = np.array([[fxx[i], fxy[i]], [fyx[i], fyy[i]]], float); A = np.eye(2)-H
        det = float(np.linalg.det(A)); sv = np.linalg.svd(A, compute_uv=False); ev = np.linalg.eigvalsh(A)
        if det == 0 or not np.isfinite(H).all():
            return None, 'exact_singularity_or_nonfinite_hessian'
        mu = 1/det
        fdA, fd_refinement, fdstep = finite_difference_A(model, kw, x, y, theta)
        detfd = float(np.linalg.det(fdA)); mufd = 1/detfd
        # Analytic SIS is independent of library hessian; others use library ray
        # magnification plus a separately differentiated lens mapping check.
        mu_reference = float((1+1/ys) if i == 0 else (1-1/ys)) if family == 'SIS' else float(ray_mu[i])
        if family == 'SIS':
            distance, distance_err = abs(np.hypot(x, y)-theta), 0.
            method = 'analytic_circular_critical_curve'
        elif family == 'NFW_like':
            if not len(critical_r):
                raise RuntimeError('Multiple-image NFW lens has no numerical critical radius.')
            distance = float(np.min(np.abs(np.hypot(x, y)-critical_r))); distance_err = 1e-12
            method = 'radial_eigenvalue_brent_roots'
        else:
            point = np.array([x,y]); distance = polygon_distance(point, critical_poly)
            distance_err = abs(distance-polygon_distance(point, critical_poly[::2]))
            method = 'analytic_polar_locus_piecewise_linear_angular_mesh'
        relfd = abs(mufd-mu)/max(abs(mu),1e-15)
        if not np.isfinite(mufd) or relfd > a.validation_rtol:
            raise RuntimeError(f'{family} independent mu check failed: relative error={relfd:.3g}; image={(x,y)}')
        kap = float((H[0,0]+H[1,1])/2); g1 = float((H[0,0]-H[1,1])/2); g2 = float((H[0,1]+H[1,0])/2)
        rows.append({'event_id':eid,'lens_family':family,'image_id':i,
            'x_true':float(x),'y_image_true':float(y),'r_true':float(np.hypot(x,y)),
            'source_x_true':float(bx),'source_y_true':float(by),'source_offset_y_true':ys,
            'mu_signed_true':mu,'mu_abs_true':abs(mu),'mu_ray_signed_reference':mu_reference,
            'mu_ray_abs_reference':abs(mu_reference),'mu_fd_signed_reference':mufd,
            'mu_fd_relative_error':relfd,'mu_ray_abs_error':abs(abs(mu_reference)-abs(mu)),
            'mu_ray_signed_error':abs(mu_reference-mu),'fd_A_refinement_error':fd_refinement,
            'fd_step_arcsec':fdstep,'ray_residual_arcsec':float(residual[i]),
            'kappa_true':kap,'gamma1_true':g1,'gamma2_true':g2,'gamma_abs_true':float(np.hypot(g1,g2)),
            'A11_true':A[0,0],'A12_true':A[0,1],'A21_true':A[1,0],'A22_true':A[1,1],
            'detA_true':det,'lambda_1_true':ev[0],'lambda_2_true':ev[1],
            'singular_value_max_true':sv[0],'singular_value_min_true':sv[-1],
            'jacobian_condition_true':float(sv[0]/sv[-1]),'parity_true':int(np.sign(det)),
            'morse_index_true':int(np.sum(ev<0)),
            'critical_distance_true':distance,'critical_distance_arcsec_true':distance,
            'critical_distance_over_theta_E_true':distance/theta,
            'critical_distance_refinement_error_arcsec':distance_err,
            'critical_distance_method':method,'fermat_potential_true':float(fermat[i]),
            'theta_E_true':theta,'source_phi_true':phi,**meta})
    return rows, None


def observed_event(rows, prior, eid, rng, a):
    p, s = sorted(rows,key=lambda r:-r['mu_abs_true'])[:2]
    cx, cy = rng.normal(0,a.lens_center_sigma,2)
    xo = np.array([p['x_true'],s['x_true']])+rng.normal(0,a.astrometry_sigma,2)
    yo = np.array([p['y_image_true'],s['y_image_true']])+rng.normal(0,a.astrometry_sigma,2)
    ro = np.hypot(xo-cx,yo-cy)
    sep = float(np.hypot(xo[0]-xo[1],yo[0]-yo[1]))
    asym = float((ro[0]-ro[1])/max(float(ro.sum()),1e-15))
    zl = max(float(prior.z_l_true+rng.normal(0,a.redshift_sigma)),1e-5)
    zs = max(float(prior.z_s_true+rng.normal(0,a.redshift_sigma)),zl+1e-3)
    sig = max(float(prior.sigma_v_true*(1+rng.normal(0,a.sigma_v_fraction))),1e-6)
    theta_direct = max(float(prior.theta_E_true*(1+rng.normal(0,a.theta_e_fraction))),1e-6)
    return {'event_id':eid,'lens_family':p['lens_family'],'source_event_id':int(prior.event_id),
        'mu0_true':p['mu_abs_true'],'mu1_true':s['mu_abs_true'],'mu_total_abs_true':sum(r['mu_abs_true'] for r in rows),
        'mu0_signed_true':p['mu_signed_true'],'mu1_signed_true':s['mu_signed_true'],
        'primary_image_id':p['image_id'],'secondary_image_id':s['image_id'],'n_images_true':len(rows),
        'y_true':p['source_offset_y_true'],'source_x_true':p['source_x_true'],'source_y_true':p['source_y_true'],
        'z_l_true':float(prior.z_l_true),'z_s_true':float(prior.z_s_true),
        'sigma_v_true':float(prior.sigma_v_true),'theta_E_true':float(prior.theta_E_true),
        'source_luminosity_distance_true_Mpc':float(prior.source_luminosity_distance_true_Mpc),
        'time_delay_pair_true_s':abs(p['time_delay_true_s']-s['time_delay_true_s']),
        'obs__z_s_observed':zs,'obs__z_l_observed':zl,'obs__z_l_over_z_s':zl/zs,
        'obs__sigma_v_observed':sig,'obs__log_sigma_v_observed':float(np.log(sig)),
        'obs__theta_E_observed':max(sep/2,1e-8),'obs__image_separation_observed':sep,
        'obs__theta_plus_abs_observed':float(ro[0]),'obs__theta_minus_abs_observed':float(ro[1]),
        'obs__image_position_asymmetry_observed':asym,
        'obs__lens_center_x_observed':float(cx),'obs__lens_center_y_observed':float(cy),
        'obs__image_x_0_observed':float(xo[0]),'obs__image_y_0_observed':float(yo[0]),
        'obs__image_x_1_observed':float(xo[1]),'obs__image_y_1_observed':float(yo[1]),
        'obs__theta_E_observed_direct':theta_direct,
        'analytic_mu0_from_image_asymmetry':1+1/max(abs(asym),1e-6),
        'waveform_features_available':False}


def main():
    from astropy.cosmology import Planck18
    from astropy.constants import c
    import astropy.units as u
    a=args(); prior, prior_columns = load_sis_prior(a.sis_root)
    prior['source_luminosity_distance_true_Mpc'] = Planck18.luminosity_distance(prior.z_s_true.to_numpy()).value
    dl = Planck18.angular_diameter_distance(prior.z_l_true.to_numpy())
    ds = Planck18.angular_diameter_distance(prior.z_s_true.to_numpy())
    dls = Planck18.angular_diameter_distance_z1z2(prior.z_l_true.to_numpy(),prior.z_s_true.to_numpy())
    prior['time_delay_factor_s_per_arcsec2'] = ((1+prior.z_l_true.to_numpy())*dl*ds/dls/c).to_value(u.s)*(np.pi/(180*3600))**2
    partitions = group_partition(prior,a.split_seed)
    prior['split'] = prior.event_id.map(partitions)
    counts = split_counts(a.n_events)
    out = a.out_dir or a.out_root/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    manifest={'generator':'universal_latent_benchmark_v2','status':'running','waveform_features_available':False,
        'families':a.families,'n_events_per_family':a.n_events,'split_counts_per_family':counts,
        'split_rule':'Fixed grouped 70/15/15 by source_event_id; same reused prior row never crosses train/validation/test or families.',
        'identity_note':'source_event_id identifies an empirical lens-prior row, not a new independent source waveform. group_id=prior:<id>. source_waveform_id is absent; do not copy old waveform rows.',
        'source_prior_columns':prior_columns,'prior_file':str((a.sis_root/'lens_params.csv').resolve()),
        'prior_sha256':hashlib.sha256((a.sis_root/'lens_params.csv').read_bytes()).hexdigest(),
        'source_prior':'Uniform source radial offset y and angle, conditional on >=2 finite solved images; this is not uniform source-plane area.',
        'target':'mu0_true = highest |mu| among all solved images; mu1_true = second highest; mu_total_abs_true separately sums all images.',
        'observable_order':'Two brightest true images define labels. Radii keep this brightness ordering; separation is true 2D Euclidean distance. asymmetry is signed radius difference/sum.',
        'critical_distance':'Geometric distance in arcsec. SIS analytic; NFW radial eigenvalue roots; SIE/shear polar critical locus, nearest 2N-segment polyline. N-versus-2N refinement error recorded, not a rigorous error bound.',
        'mu_validation':'SIS analytic signed mu versus Hessian determinant; other families library magnification (shares library Hessian) plus independent fourth-order finite differences of the ray-shooting map for ALL families.',
        'time_delay':'Planck18 thin-lens D_dt/c times relative Fermat potential; seconds, relative to earliest image. Includes NFW potential.',
        'nfw_note':'Controlled spherical NFW Rs and alpha_Rs in angular units; theta_E_true is a reference scale, not its tangential critical radius. sigma_v is an empirical scale covariate, not an NFW Jeans prediction.',
        'followup_note':'SIE/shear share SIS empirical redshift and velocity-scale priors. Direct theta_E_observed_direct measures reference normalization, not a universal Einstein-ring radius.',
        'selection':'At least 2 solved images, finite nonsingular local Jacobian; no magnification/SNR threshold. Expected rejection reasons counted; numerical validation errors fail the run.',
        'cosmology':str(Planck18),'units':{'angles':'arcsec','sigma_v':'km/s','time_delay':'s','fermat':'arcsec^2'},
        'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','astropy','lenstronomy']},
        'args':vars(a)}
    def save_manifest():
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    save_manifest(); all_validation=[]
    try:
        for family in a.families:
            fi=FAMILIES.index(family); seed=a.seed+1009*fi; rng=np.random.default_rng(seed)
            image_all=[]; event_all=[]; attempts=0; rejection=Counter(); accepted=0
            bar=tqdm(total=a.n_events,desc=f'{family} accepted',unit='event')
            try:
                for split,n_target in counts.items():
                    pool=prior[prior.split==split]
                    for _ in range(n_target):
                        while True:
                            if attempts >= a.max_attempts:
                                raise RuntimeError(f'{family} accepted {accepted}/{a.n_events} in {attempts} attempts; rejection={dict(rejection)}')
                            attempts+=1; pr=pool.iloc[int(rng.integers(0,len(pool)))]
                            theta=float(pr.theta_E_true); ys=float(rng.uniform(a.source_y_min,a.source_y_max)); phi=float(rng.uniform(0,2*np.pi))
                            model,kw,meta=model_for(family,theta,rng,a)
                            eid=fi*100000000+accepted
                            rows, reason=rows_for(model,kw,family,eid,theta,ys,phi,meta,a)
                            if reason:
                                rejection[reason]+=1
                                bar.set_postfix(attempts=attempts,rejected=sum(rejection.values()),refresh=False)
                                continue
                            group_id=f'prior:{int(pr.event_id)}'
                            minimum=min(r['fermat_potential_true'] for r in rows)
                            for r in rows:
                                r.update(source_event_id=int(pr.event_id),group_id=group_id,split=split,
                                    time_delay_true_s=(r['fermat_potential_true']-minimum)*float(pr.time_delay_factor_s_per_arcsec2),
                                    lens_kwargs_json=json.dumps(kw,separators=(',',':')))
                            ev=observed_event(rows,pr,eid,rng,a)
                            ev.update(split=split,group_id=group_id,independent_lens_realization_id=f'{a.seed}:{family}:{accepted}',
                                      q_true=meta['q_true'],external_shear_true=meta['external_shear_true'],
                                      nfw_rs_over_theta_e=meta['nfw_rs_over_theta_e'])
                            primary=ev['primary_image_id']; secondary=ev['secondary_image_id']
                            for r in rows:
                                r.update(is_primary=r['image_id']==primary,is_secondary=r['image_id']==secondary)
                            event_all.append(ev); image_all.extend(rows); accepted+=1; bar.update(1)
                            bar.set_postfix(attempts=attempts,rejected=sum(rejection.values()),refresh=False)
                            break
            finally:
                bar.close()
            fd=out/family; fd.mkdir(); img=pd.DataFrame(image_all); obs=pd.DataFrame(event_all)
            primary=img[img.is_primary].rename(columns={'x_true':'primary_x_true','y_image_true':'primary_y_image_true'}).copy()
            additions=[c for c in obs if c not in primary or c=='event_id']
            event=primary.merge(obs[additions],on='event_id',validate='one_to_one')
            for name,frame in [('image_latent',img),('event_latent',event),('observables',obs)]:
                frame.to_csv(fd/f'{name}.csv',index=False)
                for split in counts:
                    sd=fd/split; sd.mkdir(exist_ok=True); frame[frame.split==split].to_csv(sd/f'{name}.csv',index=False)
            obs[['event_id','lens_family','source_event_id','group_id','split','independent_lens_realization_id']].to_csv(fd/'event_manifest.csv',index=False)
            validation={'lens_family':family,'n_events':accepted,'n_images':len(img),'n_unique_prior_groups':obs.group_id.nunique()}
            for name,values in [('abs_mu_ray_minus_mu_detA',img.mu_ray_abs_error),
                                ('signed_mu_ray_minus_mu_detA',img.mu_ray_signed_error),
                                ('abs_mu_fd_minus_mu_detA',np.abs(np.abs(img.mu_fd_signed_reference)-img.mu_abs_true)),
                                ('relative_mu_fd_error',img.mu_fd_relative_error)]:
                for q,stat in [('median',np.median),('p90',lambda v:np.quantile(v,.9)),('max',np.max)]:
                    validation[f'{q}_{name}']=float(stat(values))
            validation['max_ray_residual_arcsec']=float(img.ray_residual_arcsec.max())
            validation['max_critical_distance_refinement_error_arcsec']=float(img.critical_distance_refinement_error_arcsec.max())
            pd.DataFrame([validation]).to_csv(fd/'latent_validation_summary.csv',index=False); all_validation.append(validation)
            config={'family':family,'seed':seed,'attempts':attempts,'accepted':accepted,'rejections':dict(rejection),
                    'split_counts':counts,'args':vars(a)}
            (fd/'generation_config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
            print(f'Completed {family}: {fd}',flush=True)
        pd.DataFrame(all_validation).to_csv(out/'latent_validation_summary.csv',index=False)
        manifest['status']='complete'; manifest['completed_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()); save_manifest()
    except Exception as exc:
        manifest['status']='failed'; manifest['failure']=repr(exc); save_manifest(); raise
    print(f'UNIVERSAL_LATENT_RUN={out.resolve()}',flush=True)


if __name__=='__main__':
    main()

#!/usr/bin/env python
# coding: utf-8
"""
Physics-oriented SIS gravitational-wave lensing baseline for mu0 prediction.

This script keeps the file contract used by the current mu0 training pipeline,
but moves the generator toward a cleaner SIS-theory simulation baseline:

* sources are still idealized BBH draws in a fixed cosmology,
* lenses are drawn from simple lensing-weighted SIS-inspired distributions,
* source impact parameter y is sampled by lensing cross-section area,
* each image receives SIS magnification, time delay, and Morse phase,
* detector response and ideal stationary Gaussian PSD noise are injected,
* catalog-level optical and timing observables are noisy proxies.

This is not an end-to-end realistic survey simulation. It is a controlled SIS
geometric-optics simulation intended for physics-baseline experiments.

The output keeps compatibility with the current mu0 training scripts while
separating:

1. training observables,
2. label/truth tables,
3. diagnostic privileged quantities,
4. quality-control reports.

Default compatible training files:

    lens.csv
    lens_params.csv
    observable_features.csv
    SIS_data_strain_1.npy
    SIS_data_strain_2.npy
    SIS_h_strain_1.npy
    SIS_h_strain_2.npy
    SIS_noise_strain_1.npy
    SIS_noise_strain_2.npy

The default `SIS_data_strain_*.npy` files are signal-centered, noisy,
whitened, one-second windows.  Full 24-second arrays are optional because they
are expensive and often dominated by background noise for amplitude statistics.
"""

from __future__ import annotations
from pathlib import Path as _ReleasePath

import argparse
import json
import logging
import math
import os
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bilby
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units
from astropy.cosmology import Planck18 as cosmo
from bilby.gw.conversion import (
    luminosity_distance_to_redshift,
    redshift_to_luminosity_distance,
)
from gwpy.time import Time as GWTime
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from tqdm import tqdm


warnings.filterwarnings("ignore")
for logger_name in ("", "bilby", "bilby.gw", "bilby.gw.source", "astropy", "gwpy"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Dataset identity and output
    # ------------------------------------------------------------------
    save_dir: str = (
        str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_sis_gw_physics_baseline')
    )
    seed: int = 238
    target_events: int = 2500
    max_attempts: int = 25000
    overwrite: bool = False

    # The paired train_mu0_physics_baseline.py entry point overrides the base
    # feature-kind check to expect this physics-baseline value.
    feature_data_kind: str = "sis_physics_baseline_noisy_whitened"
    generator_name: str = "sis_gw_physics_baseline_v1"

    # ------------------------------------------------------------------
    # Cosmology / source / lens sampling
    # ------------------------------------------------------------------
    z_min: float = 0.05
    z_max: float = 1.5
    lens_z_min: float = 0.01
    min_lens_source_z_gap: float = 1e-3

    m1_source_min: float = 20.0
    m1_source_max: float = 70.0
    m2_source_min: float = 10.0
    m2_source_max: float = 50.0

    sigma_v_min: float = 120.0
    sigma_v_max: float = 380.0

    # SIS physics baseline sampling.  y is sampled by area in the multiply
    # imaged region, p(y) proportional to y, rather than by a uniform A21 grid.
    y_min: float = 0.05
    y_max: float = 0.98
    y_bins: int = 8

    # Simple lensing-selected velocity-dispersion function.  The weight is
    # approximately VDF(sigma) * sigma^4 to reflect the SIS lensing cross-section.
    sigma_v_star: float = 220.0
    sigma_vdf_alpha: float = 2.32
    sigma_vdf_beta: float = 2.67

    # Rejection-sampling controls for the approximate lens-redshift kernel.
    lens_z_kernel_grid: int = 96
    lens_z_kernel_max_draws: int = 256
    sigma_v_max_draws: int = 256

    # Reject candidates whose SIS delay is outside the requested observable
    # window.  Long-delay events can be generated in a dedicated bin later.
    td_min_sec: float = 1.0e2
    td_max_sec: float = 1.0e7

    # ------------------------------------------------------------------
    # Detector and waveform generation
    # ------------------------------------------------------------------
    detector_names: Tuple[str, ...] = ("ET",)
    reference_detector_index: int = 0
    sampling_frequency: int = 4096
    duration: float = 24.0
    minimum_frequency: float = 20.0
    waveform_approximant: str = "IMRPhenomXPHM"
    reference_frequency: float = 10.0

    # The full simulated frame starts at geocent_time - pre_trigger_seconds.
    # The one-second training window is still centered by the injected signal
    # peak to keep a stable ML file contract, but timing observables below are
    # estimated from the noisy window before adding catalog-level uncertainty.
    pre_trigger_seconds: float = 6.0
    train_window_seconds: float = 1.0
    main_feature_seconds: float = 0.5
    arrival_search_seconds: float = 0.35
    extra_feature_windows: Tuple[float, ...] = (0.25, 0.5, 0.9)

    add_detector_noise: bool = True
    force_same_detector_response_time: bool = False
    geocent_time_includes_delay: bool = True
    save_windowed_raw: bool = True
    save_full_strain: bool = False
    save_time_arrays: bool = True

    # ------------------------------------------------------------------
    # Observation noise models
    # ------------------------------------------------------------------
    redshift_obs_sigma: float = 5.0e-4
    sigma_v_obs_frac_sigma: float = 0.05
    theta_E_obs_frac_sigma: float = 0.02
    astrometric_sigma_arcsec: float = 0.01
    lens_center_sigma_arcsec: float = 0.02

    # Arrival-time observation uncertainty.  This is intentionally simple and
    # tied to SNR so the feature is observable, not theoretical t_d.
    time_obs_floor_sec: float = 1.0 / 4096.0
    time_obs_snr_scale_sec: float = 0.03

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------
    min_snr_image_1: float = 12.0
    min_snr_image_2: float = 8.0
    min_network_snr_pair: float = 18.0
    max_abs_peak_time_error_sec: float = 0.25
    min_window_signal_noise_rms: float = 0.015

    # ------------------------------------------------------------------
    # Optional source-parameter simplification for curriculum variants.
    # The default is realistic/random.
    # ------------------------------------------------------------------
    random_spin: bool = True
    random_external: bool = True
    random_distance: bool = True
    fixed_luminosity_distance_mpc: float = 3000.0


CFG = Config()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Generate a physics-oriented SIS geometric-optics GW lensing baseline for mu0."
    )
    parser.add_argument("--save-dir", type=str, default=CFG.save_dir)
    parser.add_argument("--n-events", type=int, default=CFG.target_events)
    parser.add_argument("--max-attempts", type=int, default=CFG.max_attempts)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-full-strain", action="store_true")
    parser.add_argument("--same-response", action="store_true")
    parser.add_argument("--no-detector-noise", action="store_true")
    parser.add_argument("--td-min", type=float, default=CFG.td_min_sec)
    parser.add_argument("--td-max", type=float, default=CFG.td_max_sec)
    parser.add_argument("--y-min", type=float, default=CFG.y_min)
    parser.add_argument("--y-max", type=float, default=CFG.y_max)
    parser.add_argument("--sigma-v-min", type=float, default=CFG.sigma_v_min)
    parser.add_argument("--sigma-v-max", type=float, default=CFG.sigma_v_max)
    args = parser.parse_args()

    cfg = Config()
    cfg.save_dir = args.save_dir
    cfg.target_events = args.n_events
    cfg.max_attempts = args.max_attempts
    cfg.seed = args.seed
    cfg.overwrite = args.overwrite
    cfg.save_full_strain = args.save_full_strain
    cfg.force_same_detector_response_time = args.same_response
    cfg.add_detector_noise = not args.no_detector_noise
    cfg.td_min_sec = args.td_min
    cfg.td_max_sec = args.td_max
    cfg.y_min = args.y_min
    cfg.y_max = args.y_max
    cfg.sigma_v_min = args.sigma_v_min
    cfg.sigma_v_max = args.sigma_v_max
    if not (0.0 < cfg.y_min < cfg.y_max < 1.0):
        raise ValueError("--y-min/--y-max must satisfy 0 < y_min < y_max < 1 for double-image SIS.")
    if not (0.0 < cfg.sigma_v_min < cfg.sigma_v_max):
        raise ValueError("--sigma-v-min/--sigma-v-max must be positive and increasing.")
    return cfg


def cfg_to_jsonable(cfg: Config) -> Dict[str, object]:
    out = asdict(cfg)
    for key, value in list(out.items()):
        if isinstance(value, tuple):
            out[key] = list(value)
    return out


def seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)
    return np.random.default_rng(seed)


def prepare_output_dir(cfg: Config) -> None:
    if os.path.exists(cfg.save_dir):
        if not cfg.overwrite:
            raise RuntimeError(
                f"Output directory already exists: {cfg.save_dir}. "
                "Use --overwrite or choose --save-dir."
            )
        shutil.rmtree(cfg.save_dir)
    os.makedirs(cfg.save_dir, exist_ok=True)


def safe_ratio(a: float, b: float, eps: float = 1e-12) -> float:
    return float((a + eps) / (b + eps))


def finite_or_nan(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def observed_redshift(rng: np.random.Generator, z_true: float, cfg: Config) -> float:
    z_obs = z_true + rng.normal(0.0, cfg.redshift_obs_sigma)
    return float(max(z_obs, 0.0))


def observed_positive_fraction(
    rng: np.random.Generator,
    x_true: float,
    frac_sigma: float,
) -> float:
    x_obs = x_true * (1.0 + rng.normal(0.0, frac_sigma))
    return float(max(x_obs, 1e-12))


def simple_envelope(x: np.ndarray, win: int) -> np.ndarray:
    a = np.abs(x).astype(np.float64)
    if win <= 1:
        return a
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(a, kernel, mode="same")


def crop_with_padding(x: np.ndarray, center: int, length: int) -> np.ndarray:
    half_left = length // 2
    start = center - half_left
    end = start + length

    left_pad = max(0, -start)
    right_pad = max(0, end - len(x))
    start = max(0, start)
    end = min(len(x), end)

    y = x[start:end]
    if left_pad or right_pad:
        y = np.pad(y, (left_pad, right_pad), mode="constant")
    return y


def centered_segment(x: np.ndarray, fs: int, seconds: float) -> np.ndarray:
    n = max(8, int(round(seconds * fs)))
    n = min(n, len(x))
    center = len(x) // 2
    return crop_with_padding(x, center, n).astype(np.float64)


def normalized_xcorr(x: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    n = len(x0) + len(y0) - 1
    nfft = int(2 ** np.ceil(np.log2(max(n, 2))))
    X = np.fft.rfft(x0, n=nfft)
    Y = np.fft.rfft(y0, n=nfft)
    corr = np.fft.irfft(X.conj() * Y, n=nfft)[:n]
    corr = np.concatenate([corr[-(len(x0) - 1):], corr[:len(y0)]])
    denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2)) + 1e-12
    norm_corr = corr / denom
    lag = int(np.argmax(norm_corr) - (len(x0) - 1))
    return lag, float(np.max(corr)), float(np.max(norm_corr))


def aligned_residual(x1: np.ndarray, x2: np.ndarray) -> Tuple[float, float]:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    scale = float(np.dot(x2, x1) / (np.dot(x1, x1) + 1e-12))
    resid = x2 - scale * x1
    return float(np.mean(np.abs(resid))), float(np.sqrt(np.mean(resid**2)))


def spectrum_features(x: np.ndarray, fs: float) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spec_sum = np.sum(spec) + 1e-12
    centroid = float(np.sum(freqs * spec) / spec_sum)
    low = float(np.sum(spec[(freqs >= 20) & (freqs < 80)]))
    mid = float(np.sum(spec[(freqs >= 80) & (freqs < 200)]))
    high = float(np.sum(spec[(freqs >= 200) & (freqs < min(512, fs / 2))]))
    return centroid, low, mid, high


def window_stats_pair(x1: np.ndarray, x2: np.ndarray, fs: int, seconds: float) -> Dict[str, float]:
    s1 = centered_segment(x1, fs, seconds)
    s2 = centered_segment(x2, fs, seconds)
    env_win = max(5, int(round(0.03 * fs)))
    if env_win % 2 == 0:
        env_win += 1
    env1 = simple_envelope(s1, env_win)
    env2 = simple_envelope(s2, env_win)
    return {
        "peak_amp_ratio": safe_ratio(float(np.max(np.abs(s2))), float(np.max(np.abs(s1)))),
        "rms_ratio": safe_ratio(
            float(np.sqrt(np.mean(s2**2) + 1e-12)),
            float(np.sqrt(np.mean(s1**2) + 1e-12)),
        ),
        "energy_ratio": safe_ratio(float(np.sum(s2**2)), float(np.sum(s1**2))),
        "env_area_ratio": safe_ratio(float(np.sum(env2)), float(np.sum(env1))),
    }


def edge_noise_rms(noise_window: np.ndarray, fs: int) -> float:
    edge = max(8, int(round(0.15 * fs)))
    edge = min(edge, len(noise_window) // 3)
    seg = np.concatenate([noise_window[:edge], noise_window[-edge:]])
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))


def observed_window_snr_proxy(data_window: np.ndarray, fs: int, signal_seconds: float = 0.25) -> float:
    """Estimate a simple SNR-like quantity from observed data only.

    The signal is centered by the dataset construction trigger.  The edge RMS is
    an off-source noise estimate from the same noisy window, so this proxy avoids
    using the separately stored true noise or clean signal.
    """
    x = np.asarray(data_window, dtype=np.float64)
    noise_rms = edge_noise_rms(x, fs)
    sig = centered_segment(x, fs, signal_seconds)
    peak = float(np.max(np.abs(sig)))
    return safe_ratio(peak, noise_rms)


def observed_peak_time_from_window(
    data_window: np.ndarray,
    time_window: np.ndarray,
    fs: int,
    search_seconds: float,
) -> float:
    """Estimate arrival time from noisy data within the centered trigger window."""
    x = np.asarray(data_window, dtype=np.float64)
    t = np.asarray(time_window, dtype=np.float64)
    if len(x) == 0 or len(t) != len(x):
        return float("nan")
    n = max(8, int(round(search_seconds * fs)))
    n = min(n, len(x))
    center = len(x) // 2
    start = max(0, center - n // 2)
    end = min(len(x), start + n)
    start = max(0, end - n)
    local = x[start:end]
    local_t = t[start:end]
    env_win = max(5, int(round(0.02 * fs)))
    if env_win % 2 == 0:
        env_win += 1
    env = simple_envelope(local, env_win)
    return float(local_t[int(np.argmax(env))])


def draw_comoving_distance(
    rng: np.random.Generator,
    cfg: Config,
    distance_prior: bilby.core.prior.Prior,
) -> float:
    if cfg.random_distance:
        return float(distance_prior.sample())
    return float(cfg.fixed_luminosity_distance_mpc)


def draw_source(
    rng: np.random.Generator,
    cfg: Config,
    distance_prior: bilby.core.prior.Prior,
) -> Dict[str, float]:
    d_l = draw_comoving_distance(rng, cfg, distance_prior)
    z_s = float(luminosity_distance_to_redshift(d_l, cosmology=cosmo))

    m1_source = float(rng.uniform(cfg.m1_source_min, cfg.m1_source_max))
    m2_high = min(cfg.m2_source_max, m1_source)
    m2_source = float(rng.uniform(cfg.m2_source_min, m2_high))
    if m2_source > m1_source:
        m1_source, m2_source = m2_source, m1_source

    if cfg.random_spin:
        a_1 = float(rng.uniform(0.0, 0.99))
        a_2 = float(rng.uniform(0.0, 0.99))
        tilt_1 = float(np.arccos(rng.uniform(-1.0, 1.0)))
        tilt_2 = float(np.arccos(rng.uniform(-1.0, 1.0)))
        phi_12 = float(rng.uniform(0.0, 2.0 * np.pi))
        phi_jl = float(rng.uniform(0.0, 2.0 * np.pi))
    else:
        a_1 = a_2 = 0.5
        tilt_1 = tilt_2 = np.pi / 2.0
        phi_12 = phi_jl = 0.0

    if cfg.random_external:
        ra = float(rng.uniform(0.0, 2.0 * np.pi))
        dec = float(np.arcsin(rng.uniform(-1.0, 1.0)))
        theta_jn = float(np.arccos(rng.uniform(-1.0, 1.0)))
        psi = float(rng.uniform(0.0, np.pi))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
    else:
        ra = np.pi
        dec = 0.0
        theta_jn = np.pi / 2.0
        psi = np.pi / 2.0
        phase = 0.0

    time_start = GWTime("2015-09-14 09:50:45.39", scale="utc").gps
    time_end = GWTime("2025-12-10 17:18:45.39", scale="utc").gps
    geocent_time = float(rng.uniform(time_start, time_end))

    redshift_factor = 1.0 + z_s
    return {
        "luminosity_distance": d_l,
        "source_redshift": z_s,
        "z_s_true": z_s,
        "mass_1_source": m1_source,
        "mass_2_source": m2_source,
        "mass_1": m1_source * redshift_factor,
        "mass_2": m2_source * redshift_factor,
        "a_1": a_1,
        "a_2": a_2,
        "tilt_1": tilt_1,
        "tilt_2": tilt_2,
        "phi_12": phi_12,
        "phi_jl": phi_jl,
        "ra": ra,
        "dec": dec,
        "theta_jn": theta_jn,
        "psi": psi,
        "phase": phase,
        "geocent_time": geocent_time,
    }


def draw_y_cross_section(rng: np.random.Generator, cfg: Config) -> Tuple[float, int]:
    """Draw SIS impact parameter from p(y) proportional to y on [y_min, y_max]."""
    y2 = rng.uniform(cfg.y_min**2, cfg.y_max**2)
    y = float(np.sqrt(y2))
    edges = np.linspace(cfg.y_min, cfg.y_max, max(1, cfg.y_bins) + 1)
    bin_id = int(np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(edges) - 2))
    return y, bin_id


def lens_redshift_kernel(z_l: float, z_s: float) -> float:
    """Approximate SIS lensing optical-depth kernel for fixed sigma_v."""
    if z_l <= 0.0 or z_l >= z_s:
        return 0.0
    try:
        Dl = cosmo.angular_diameter_distance(z_l).to(units.Mpc).value
        Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(units.Mpc).value
        Ds = cosmo.angular_diameter_distance(z_s).to(units.Mpc).value
        dvc_dz = cosmo.differential_comoving_volume(z_l).to(units.Mpc**3 / units.sr).value
    except Exception:
        return 0.0
    if not np.isfinite(Dl + Dls + Ds + dvc_dz) or Ds <= 0.0 or Dls <= 0.0:
        return 0.0
    geometry = (Dl * Dls / Ds) ** 2
    return float(max(dvc_dz * geometry, 0.0))


def draw_lens_redshift(rng: np.random.Generator, cfg: Config, z_s: float) -> Tuple[Optional[float], Optional[str]]:
    z_hi = z_s - cfg.min_lens_source_z_gap
    if z_hi <= cfg.lens_z_min:
        return None, "no_valid_lens_redshift"

    grid = np.linspace(cfg.lens_z_min, z_hi, max(8, cfg.lens_z_kernel_grid))
    weights = np.array([lens_redshift_kernel(float(z), z_s) for z in grid], dtype=np.float64)
    wmax = float(np.nanmax(weights)) if len(weights) else 0.0
    if not np.isfinite(wmax) or wmax <= 0.0:
        return float(rng.uniform(cfg.lens_z_min, z_hi)), None

    for _ in range(max(1, cfg.lens_z_kernel_max_draws)):
        z = float(rng.uniform(cfg.lens_z_min, z_hi))
        if rng.uniform(0.0, wmax) <= lens_redshift_kernel(z, z_s):
            return z, None
    return float(rng.choice(grid, p=weights / np.sum(weights))), None


def sigma_v_lensing_weight(sigma_v: float, cfg: Config) -> float:
    """Approximate lensing-selected early-type velocity-dispersion weight."""
    x = max(float(sigma_v), 1e-12) / cfg.sigma_v_star
    vdf = x ** cfg.sigma_vdf_alpha * np.exp(-(x ** cfg.sigma_vdf_beta)) / max(x, 1e-12)
    cross_section = float(sigma_v) ** 4
    return float(max(vdf * cross_section, 0.0))


def draw_sigma_v_lensing_selected(rng: np.random.Generator, cfg: Config) -> float:
    grid = np.linspace(cfg.sigma_v_min, cfg.sigma_v_max, 512)
    weights = np.array([sigma_v_lensing_weight(float(s), cfg) for s in grid], dtype=np.float64)
    wmax = float(np.nanmax(weights)) if len(weights) else 0.0
    if not np.isfinite(wmax) or wmax <= 0.0:
        return float(rng.uniform(cfg.sigma_v_min, cfg.sigma_v_max))

    for _ in range(max(1, cfg.sigma_v_max_draws)):
        sigma_v = float(rng.uniform(cfg.sigma_v_min, cfg.sigma_v_max))
        if rng.uniform(0.0, wmax) <= sigma_v_lensing_weight(sigma_v, cfg):
            return sigma_v
    return float(rng.choice(grid, p=weights / np.sum(weights)))


def draw_lens(
    rng: np.random.Generator,
    cfg: Config,
    source: Dict[str, float],
    attempt_id: int,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    z_s = float(source["z_s_true"])
    z_l_max = z_s - cfg.min_lens_source_z_gap
    if z_l_max <= cfg.lens_z_min:
        return None, "no_valid_lens_redshift"

    z_l, reason = draw_lens_redshift(rng, cfg, z_s)
    if z_l is None:
        return None, reason
    sigma_v = draw_sigma_v_lensing_selected(rng, cfg)
    lens_cosmo = LensCosmo(z_lens=z_l, z_source=z_s, cosmo=cosmo)

    theta_E = float(lens_cosmo.sis_sigma_v2theta_E(sigma_v))
    y, y_bin = draw_y_cross_section(rng, cfg)
    R = (1.0 - y) / (1.0 + y)
    A21 = float(np.sqrt(max(R, 0.0)))
    u = 1.0 / y
    log_u = float(np.log(u))
    mu0 = 1.0 + u
    mu1 = 1.0 - u

    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    dx = float(np.cos(phi))
    dy = float(np.sin(phi))
    beta_x = y * theta_E * dx
    beta_y = y * theta_E * dy
    beta = float(np.sqrt(beta_x**2 + beta_y**2))

    image_x_0 = (1.0 + y) * theta_E * dx
    image_y_0 = (1.0 + y) * theta_E * dy
    image_x_1 = (y - 1.0) * theta_E * dx
    image_y_1 = (y - 1.0) * theta_E * dy

    Dls = (lens_cosmo.dds * units.Mpc).to(units.m).value
    Dl = (lens_cosmo.dd * units.Mpc).to(units.m).value
    Ds = (lens_cosmo.ds * units.Mpc).to(units.m).value
    t_d = (
        32.0
        * np.pi**2
        * (sigma_v * 1000.0) ** 4
        * (1.0 + z_l)
        * Dls
        * Dl
        * y
        / (Ds * const.c.value**5)
    )
    t_d = float(t_d)

    if t_d < cfg.td_min_sec or t_d > cfg.td_max_sec:
        return None, "td_out_of_range"

    z_l_obs = observed_redshift(rng, z_l, cfg)
    z_s_obs = observed_redshift(rng, z_s, cfg)
    if z_s_obs <= z_l_obs:
        z_s_obs = z_l_obs + cfg.min_lens_source_z_gap
    sigma_v_obs = observed_positive_fraction(rng, sigma_v, cfg.sigma_v_obs_frac_sigma)
    theta_E_obs_direct = observed_positive_fraction(rng, theta_E, cfg.theta_E_obs_frac_sigma)

    lens_center_x = 0.0
    lens_center_y = 0.0
    lens_center_x_obs = float(lens_center_x + rng.normal(0.0, cfg.lens_center_sigma_arcsec))
    lens_center_y_obs = float(lens_center_y + rng.normal(0.0, cfg.lens_center_sigma_arcsec))

    image_x_0_obs = float(image_x_0 + rng.normal(0.0, cfg.astrometric_sigma_arcsec))
    image_y_0_obs = float(image_y_0 + rng.normal(0.0, cfg.astrometric_sigma_arcsec))
    image_x_1_obs = float(image_x_1 + rng.normal(0.0, cfg.astrometric_sigma_arcsec))
    image_y_1_obs = float(image_y_1 + rng.normal(0.0, cfg.astrometric_sigma_arcsec))

    theta_plus_abs_obs = float(
        np.sqrt((image_x_0_obs - lens_center_x_obs) ** 2 + (image_y_0_obs - lens_center_y_obs) ** 2)
    )
    theta_minus_abs_obs = float(
        np.sqrt((image_x_1_obs - lens_center_x_obs) ** 2 + (image_y_1_obs - lens_center_y_obs) ** 2)
    )
    image_separation_obs = theta_plus_abs_obs + theta_minus_abs_obs
    theta_E_obs_from_images = 0.5 * image_separation_obs
    image_asym_obs = safe_ratio(
        theta_plus_abs_obs - theta_minus_abs_obs,
        image_separation_obs,
    )
    image_asym_obs = float(np.clip(image_asym_obs, 1e-4, 0.999))
    mu0_from_img_obs = 1.0 + 1.0 / image_asym_obs

    return {
        "z_l": z_l,
        "z_s": z_s,
        "z_l_observed": z_l_obs,
        "z_s_observed": z_s_obs,
        "z_l_over_z_s": safe_ratio(z_l_obs, z_s_obs),
        "log1p_z_l_observed": float(np.log1p(z_l_obs)),
        "log1p_z_s_observed": float(np.log1p(z_s_obs)),
        "sigma_v": sigma_v,
        "sigma_v_true": sigma_v,
        "sigma_v_observed": sigma_v_obs,
        "log_sigma_v_observed": float(np.log(max(sigma_v_obs, 1e-12))),
        "theta_E_arcsec": theta_E,
        "theta_E_observed_arcsec": theta_E_obs_from_images,
        "theta_E_observed_direct_arcsec": theta_E_obs_direct,
        "image_separation_arcsec": 2.0 * theta_E,
        "image_separation_observed_arcsec": image_separation_obs,
        "theta_plus_abs_observed": theta_plus_abs_obs,
        "theta_minus_abs_observed": theta_minus_abs_obs,
        "image_position_asymmetry_observed": image_asym_obs,
        "mu0_from_image_asymmetry_observed": mu0_from_img_obs,
        "lens_center_x": lens_center_x,
        "lens_center_y": lens_center_y,
        "lens_center_x_observed": lens_center_x_obs,
        "lens_center_y_observed": lens_center_y_obs,
        "image_x_0": image_x_0,
        "image_y_0": image_y_0,
        "image_x_1": image_x_1,
        "image_y_1": image_y_1,
        "image_x_0_observed": image_x_0_obs,
        "image_y_0_observed": image_y_0_obs,
        "image_x_1_observed": image_x_1_obs,
        "image_y_1_observed": image_y_1_obs,
        "beta": beta,
        "beta_x": beta_x,
        "beta_y": beta_y,
        "phi": phi,
        "u": u,
        "log_u": log_u,
        "y": y,
        "mu_0": mu0,
        "mu_1": mu1,
        "abs_mu_0": abs(mu0),
        "abs_mu_1": abs(mu1),
        "mu_total_abs": abs(mu0) + abs(mu1),
        "R_abs_21": R,
        "mu_ratio_abs_21": R,
        "amp_ratio_21": A21,
        "A21_true": A21,
        "a21_bin": y_bin,
        "y_bin": y_bin,
        "t_d": t_d,
    }, None


def lens_amplification(
    frequency_array: np.ndarray,
    mu_0: float,
    mu_1: float,
    t_d: float,
    which_image: int,
    cfg: Config,
) -> np.ndarray:
    mu = mu_0 if which_image == 0 else mu_1
    morse_n = 0.0 if which_image == 0 else 0.5
    if cfg.geocent_time_includes_delay:
        t_eff = 0.0
    else:
        t_eff = 0.0 if which_image == 0 else t_d
    return np.sqrt(abs(mu)) * np.exp(-1j * np.pi * (2.0 * frequency_array * t_eff + morse_n))


def lensed_bbh_source(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    theta_jn,
    phase,
    psi,
    ra,
    dec,
    geocent_time,
    mu_0,
    mu_1,
    t_d,
    which_image,
    **kwargs,
):
    waveform_keys = {
        "waveform_approximant",
        "reference_frequency",
        "minimum_frequency",
        "maximum_frequency",
        "catch_waveform_errors",
        "pn_spin_order",
        "pn_tidal_order",
        "pn_phase_order",
        "pn_amplitude_order",
        "mode_array",
        "lal_waveform_dictionary",
    }
    waveform_kwargs = {k: v for k, v in kwargs.items() if k in waveform_keys}
    params = {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "luminosity_distance": luminosity_distance,
        "a_1": a_1,
        "a_2": a_2,
        "tilt_1": tilt_1,
        "tilt_2": tilt_2,
        "phi_12": phi_12,
        "phi_jl": phi_jl,
        "theta_jn": theta_jn,
        "phase": phase,
        "psi": psi,
        "ra": ra,
        "dec": dec,
        "geocent_time": geocent_time,
    }
    h = bilby.gw.source.lal_binary_black_hole(frequency_array, **params, **waveform_kwargs)
    amp = lens_amplification(frequency_array, mu_0, mu_1, t_d, int(which_image), CFG)
    return {"plus": h["plus"] * amp, "cross": h["cross"] * amp}


def build_waveform_generator(cfg: Config) -> bilby.gw.waveform_generator.WaveformGenerator:
    args = {
        "waveform_approximant": cfg.waveform_approximant,
        "reference_frequency": cfg.reference_frequency,
        "minimum_frequency": cfg.minimum_frequency,
    }
    return bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=cfg.sampling_frequency,
        duration=cfg.duration,
        frequency_domain_source_model=lensed_bbh_source,
        waveform_arguments=args,
    )


def simulate_image(
    params: Dict[str, float],
    which_image: int,
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator,
    cfg: Config,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    N = int(round(cfg.sampling_frequency * cfg.duration))
    win_len = int(round(cfg.sampling_frequency * cfg.train_window_seconds))
    if win_len <= 0:
        return None, "invalid_train_window"

    ifos = bilby.gw.detector.InterferometerList(list(cfg.detector_names))
    if len(ifos) == 0:
        return None, "no_detector"
    if cfg.reference_detector_index >= len(ifos):
        return None, "bad_reference_detector"

    snr_sq_sum = 0.0
    detector_snr = {}
    detector_outputs = {}

    for det_idx, ifo in enumerate(ifos):
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=cfg.sampling_frequency,
            duration=cfg.duration,
            start_time=params["geocent_time"] - cfg.pre_trigger_seconds,
        )
        if not cfg.add_detector_noise:
            ifo.strain_data.frequency_domain_strain *= 0.0

        noise_raw = np.array(ifo.strain_data.time_domain_strain, dtype=np.float64)
        noise_white = np.fft.irfft(ifo.whitened_frequency_domain_strain, n=N)
        time_array = np.array(ifo.strain_data.time_array, dtype=np.float64)

        fd_waveform = waveform_generator.frequency_domain_strain(params)
        response = ifo.get_detector_response(fd_waveform, params)
        snr_sq = float(np.real(ifo.optimal_snr_squared(response)))
        snr_sq = max(snr_sq, 0.0)
        snr = float(np.sqrt(snr_sq))
        snr_sq_sum += snr_sq
        detector_snr[f"snr_{ifo.name}"] = snr

        ifo.strain_data.frequency_domain_strain += response
        data_raw = np.array(ifo.strain_data.time_domain_strain, dtype=np.float64)
        data_white = np.fft.irfft(ifo.whitened_frequency_domain_strain, n=N)

        signal_raw = data_raw - noise_raw
        signal_white = data_white - noise_white
        peak_idx = int(np.argmax(np.abs(signal_white)))
        peak_time = float(time_array[peak_idx])

        out = {
            "data_white": crop_with_padding(data_white, peak_idx, win_len),
            "signal_white": crop_with_padding(signal_white, peak_idx, win_len),
            "noise_white": crop_with_padding(noise_white, peak_idx, win_len),
            "data_raw": crop_with_padding(data_raw, peak_idx, win_len),
            "signal_raw": crop_with_padding(signal_raw, peak_idx, win_len),
            "noise_raw": crop_with_padding(noise_raw, peak_idx, win_len),
            "time_window": crop_with_padding(time_array, peak_idx, win_len).astype(np.float64),
            "clean_peak_time": peak_time,
            "clean_peak_index": peak_idx,
            "clean_white_rms": float(np.sqrt(np.mean(signal_white**2) + 1e-12)),
            "noise_white_rms": float(np.sqrt(np.mean(noise_white**2) + 1e-12)),
            "window_clean_white_rms": float(
                np.sqrt(np.mean(crop_with_padding(signal_white, peak_idx, win_len).astype(np.float64) ** 2) + 1e-12)
            ),
            "window_noise_white_rms": float(
                np.sqrt(np.mean(crop_with_padding(noise_white, peak_idx, win_len).astype(np.float64) ** 2) + 1e-12)
            ),
        }
        if cfg.save_full_strain:
            out.update(
                {
                    "data_white_full": data_white.astype(np.float32),
                    "signal_white_full": signal_white.astype(np.float32),
                    "noise_white_full": noise_white.astype(np.float32),
                    "data_raw_full": data_raw.astype(np.float32),
                    "signal_raw_full": signal_raw.astype(np.float32),
                    "noise_raw_full": noise_raw.astype(np.float32),
                    "time_full": time_array.astype(np.float64),
                }
            )
        detector_outputs[ifo.name] = out

    network_snr = float(np.sqrt(max(snr_sq_sum, 0.0)))
    ref_name = ifos[cfg.reference_detector_index].name
    ref = detector_outputs[ref_name]
    return {
        "network_snr": network_snr,
        "detector_snr": detector_snr,
        "reference_detector": ref_name,
        "reference": ref,
    }, None


def build_image_parameters(
    source: Dict[str, float],
    lens: Dict[str, float],
    which_image: int,
    cfg: Config,
) -> Dict[str, float]:
    params = dict(source)
    if which_image == 1 and not cfg.force_same_detector_response_time:
        params["geocent_time"] = float(source["geocent_time"] + lens["t_d"])
    params.update(
        {
            "mu_0": lens["mu_0"],
            "mu_1": lens["mu_1"],
            "t_d": lens["t_d"],
            "which_image": which_image,
        }
    )
    return params


def arrival_time_observation(
    rng: np.random.Generator,
    peak_time_estimate: float,
    observed_snr_proxy: float,
    cfg: Config,
) -> Tuple[float, float]:
    sigma = max(
        cfg.time_obs_floor_sec,
        cfg.time_obs_snr_scale_sec / max(observed_snr_proxy, 1e-6),
    )
    return float(peak_time_estimate + rng.normal(0.0, sigma)), float(sigma)


def simulate_pair(
    rng: np.random.Generator,
    source: Dict[str, float],
    lens: Dict[str, float],
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator,
    cfg: Config,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    p0 = build_image_parameters(source, lens, 0, cfg)
    p1 = build_image_parameters(source, lens, 1, cfg)

    img0, reason = simulate_image(p0, 0, waveform_generator, cfg)
    if img0 is None:
        return None, f"image1_{reason}"
    img1, reason = simulate_image(p1, 1, waveform_generator, cfg)
    if img1 is None:
        return None, f"image2_{reason}"

    snr1 = float(img0["network_snr"])
    snr2 = float(img1["network_snr"])
    pair_snr = float(np.sqrt(snr1**2 + snr2**2))

    if snr1 < cfg.min_snr_image_1:
        return None, "low_snr_image_1"
    if snr2 < cfg.min_snr_image_2:
        return None, "low_snr_image_2"
    if pair_snr < cfg.min_network_snr_pair:
        return None, "low_pair_network_snr"

    ref0 = img0["reference"]
    ref1 = img1["reference"]
    true_window_snr_like_0 = safe_ratio(ref0["window_clean_white_rms"], ref0["window_noise_white_rms"])
    true_window_snr_like_1 = safe_ratio(ref1["window_clean_white_rms"], ref1["window_noise_white_rms"])
    if min(true_window_snr_like_0, true_window_snr_like_1) < cfg.min_window_signal_noise_rms:
        return None, "low_window_signal_noise_rms"

    observed_window_snr_like_0 = observed_window_snr_proxy(
        ref0["data_white"], cfg.sampling_frequency, cfg.main_feature_seconds
    )
    observed_window_snr_like_1 = observed_window_snr_proxy(
        ref1["data_white"], cfg.sampling_frequency, cfg.main_feature_seconds
    )

    observed_peak_time_0 = observed_peak_time_from_window(
        ref0["data_white"], ref0["time_window"], cfg.sampling_frequency, cfg.arrival_search_seconds
    )
    observed_peak_time_1 = observed_peak_time_from_window(
        ref1["data_white"], ref1["time_window"], cfg.sampling_frequency, cfg.arrival_search_seconds
    )
    if not (np.isfinite(observed_peak_time_0) and np.isfinite(observed_peak_time_1)):
        return None, "arrival_time_obs_bad"

    arr0, arr0_sigma = arrival_time_observation(
        rng, observed_peak_time_0, observed_window_snr_like_0, cfg
    )
    arr1, arr1_sigma = arrival_time_observation(
        rng, observed_peak_time_1, observed_window_snr_like_1, cfg
    )
    peak_time_diff_obs = arr1 - arr0
    if abs((peak_time_diff_obs - lens["t_d"])) > max(cfg.max_abs_peak_time_error_sec, 5.0 * (arr0_sigma + arr1_sigma)):
        return None, "arrival_time_obs_bad"

    return {
        "image1": img0,
        "image2": img1,
        "snr_1": snr1,
        "snr_2": snr2,
        "snr_pair": pair_snr,
        "arrival_time_1_observed": arr0,
        "arrival_time_2_observed": arr1,
        "arrival_time_1_sigma": arr0_sigma,
        "arrival_time_2_sigma": arr1_sigma,
        "arrival_time_1_peak_noisy": observed_peak_time_0,
        "arrival_time_2_peak_noisy": observed_peak_time_1,
        "peak_time_diff_observed": peak_time_diff_obs,
        "env_peak_time_diff_observed": peak_time_diff_obs,
        "window_snr_like_1": observed_window_snr_like_0,
        "window_snr_like_2": observed_window_snr_like_1,
        "true_window_snr_like_1": true_window_snr_like_0,
        "true_window_snr_like_2": true_window_snr_like_1,
    }, None


def observable_feature_row(
    event_id: int,
    source: Dict[str, float],
    lens: Dict[str, float],
    sim: Dict[str, object],
    cfg: Config,
) -> Dict[str, float]:
    ref1 = sim["image1"]["reference"]
    ref2 = sim["image2"]["reference"]
    x1 = np.asarray(ref1["data_white"], dtype=np.float32)
    x2 = np.asarray(ref2["data_white"], dtype=np.float32)
    fs = cfg.sampling_frequency

    main = window_stats_pair(x1, x2, fs, cfg.main_feature_seconds)
    local = window_stats_pair(x1, x2, fs, cfg.main_feature_seconds)

    xcorr_lag, _, norm_xcorr_max = normalized_xcorr(x1, x2)
    local_half_seconds = min(0.25, cfg.train_window_seconds / 2.0)
    xs1 = centered_segment(x1, fs, 2.0 * local_half_seconds)
    xs2 = centered_segment(x2, fs, 2.0 * local_half_seconds)
    xcorr_lag_win65, _, norm_xcorr_max_win65 = normalized_xcorr(xs1, xs2)
    aligned_l1, aligned_l2 = aligned_residual(xs1, xs2)

    c1, l1, m1, h1 = spectrum_features(xs1, fs)
    c2, l2, m2, h2 = spectrum_features(xs2, fs)

    # Estimate local noise from observed window edges rather than using the
    # separately stored true noise realization.
    local_noise_rms_1 = edge_noise_rms(x1, fs)
    local_noise_rms_2 = edge_noise_rms(x2, fs)

    row = {
        "event_id": event_id,
        "feature_data_kind": cfg.feature_data_kind,
        "generator_data_kind": cfg.generator_name,
        "reference_detector": sim["image1"]["reference_detector"],

        # SNR columns are saved for diagnostics/ablation.  Current main
        # training keeps USE_OPTIMAL_SNR_FEATURES=False.
        "snr_1": float(sim["snr_1"]),
        "snr_2": float(sim["snr_2"]),
        "snr_ratio_21": safe_ratio(float(sim["snr_2"]), float(sim["snr_1"])),
        "snr_sum": float(sim["snr_1"] + sim["snr_2"]),
        "snr_diff": float(sim["snr_2"] - sim["snr_1"]),
        "snr_pair": float(sim["snr_pair"]),

        # Optical/lens observables.
        "z_s_observed": float(lens["z_s_observed"]),
        "z_l_observed": float(lens["z_l_observed"]),
        "z_l_over_z_s": float(lens["z_l_over_z_s"]),
        "log1p_z_s_observed": float(lens["log1p_z_s_observed"]),
        "log1p_z_l_observed": float(lens["log1p_z_l_observed"]),
        "source_luminosity_distance": float(source["luminosity_distance"]),
        "sigma_v_observed": float(lens["sigma_v_observed"]),
        "log_sigma_v_observed": float(lens["log_sigma_v_observed"]),
        "theta_E_observed": float(lens["theta_E_observed_arcsec"]),
        "image_separation_observed": float(lens["image_separation_observed_arcsec"]),
        "theta_plus_abs_observed": float(lens["theta_plus_abs_observed"]),
        "theta_minus_abs_observed": float(lens["theta_minus_abs_observed"]),
        "image_position_asymmetry_observed": float(lens["image_position_asymmetry_observed"]),
        "lens_center_x_observed": float(lens["lens_center_x_observed"]),
        "lens_center_y_observed": float(lens["lens_center_y_observed"]),
        "image_x_0_observed": float(lens["image_x_0_observed"]),
        "image_y_0_observed": float(lens["image_y_0_observed"]),
        "image_x_1_observed": float(lens["image_x_1_observed"]),
        "image_y_1_observed": float(lens["image_y_1_observed"]),

        # Observed time-delay proxies.
        "arrival_time_1_observed": float(sim["arrival_time_1_observed"]),
        "arrival_time_2_observed": float(sim["arrival_time_2_observed"]),
        "arrival_time_1_sigma": float(sim["arrival_time_1_sigma"]),
        "arrival_time_2_sigma": float(sim["arrival_time_2_sigma"]),
        "peak_time_diff": float(sim["peak_time_diff_observed"]),
        "env_peak_time_diff": float(sim["env_peak_time_diff_observed"]),

        # Default waveform-stat columns expected by existing training.
        "peak_amp_ratio_21": float(main["peak_amp_ratio"]),
        "rms_ratio_21": float(main["rms_ratio"]),
        "energy_ratio_21": float(main["energy_ratio"]),
        "xcorr_lag": int(xcorr_lag),
        "norm_xcorr_max": float(norm_xcorr_max),
        "xcorr_lag_win65": int(xcorr_lag_win65),
        "norm_xcorr_max_win65": float(norm_xcorr_max_win65),
        "peak_amp_ratio_local_65": float(local["peak_amp_ratio"]),
        "rms_ratio_local_65": float(local["rms_ratio"]),
        "energy_ratio_local_65": float(local["energy_ratio"]),
        "env_area_ratio_21": float(main["env_area_ratio"]),
        "aligned_l1_residual": float(aligned_l1),
        "aligned_l2_residual": float(aligned_l2),
        "spec_centroid_diff": float(c2 - c1),
        "band_energy_ratio_low_21": safe_ratio(l2, l1),
        "band_energy_ratio_mid_21": safe_ratio(m2, m1),
        "band_energy_ratio_high_21": safe_ratio(h2, h1),

        "local_noise_rms_1": local_noise_rms_1,
        "local_noise_rms_2": local_noise_rms_2,
        "window_snr_like_1": float(sim["window_snr_like_1"]),
        "window_snr_like_2": float(sim["window_snr_like_2"]),
    }

    for seconds in cfg.extra_feature_windows:
        stats = window_stats_pair(x1, x2, fs, seconds)
        tag = str(seconds).replace(".", "p")
        row[f"win{tag}_peak_amp_ratio_21"] = float(stats["peak_amp_ratio"])
        row[f"win{tag}_rms_ratio_21"] = float(stats["rms_ratio"])
        row[f"win{tag}_energy_ratio_21"] = float(stats["energy_ratio"])
        row[f"win{tag}_env_area_ratio_21"] = float(stats["env_area_ratio"])

    return row


def lens_csv_row(event_id: int, lens: Dict[str, float]) -> Dict[str, float]:
    return {
        "event_id": event_id,
        "mu_0": float(lens["mu_0"]),
        "mu_1": float(lens["mu_1"]),
        "abs_mu_0": float(lens["abs_mu_0"]),
        "abs_mu_1": float(lens["abs_mu_1"]),
        "mu_total_abs": float(lens["mu_total_abs"]),
        "mu_ratio_abs_21": float(lens["mu_ratio_abs_21"]),
        "R_abs_21": float(lens["R_abs_21"]),
        "amp_ratio_21": float(lens["amp_ratio_21"]),
        "u": float(lens["u"]),
        "log_u": float(lens["log_u"]),
        "y": float(lens["y"]),
        "t_d": float(lens["t_d"]),
    }


def lens_params_row(event_id: int, source: Dict[str, float], lens: Dict[str, float]) -> Dict[str, float]:
    return {
        "event_id": event_id,
        "z_l": float(lens["z_l"]),
        "z_s": float(lens["z_s"]),
        "z_l_true": float(lens["z_l"]),
        "z_s_true": float(lens["z_s"]),
        "z_l_observed": float(lens["z_l_observed"]),
        "z_s_observed": float(lens["z_s_observed"]),
        "z_l_over_z_s": float(lens["z_l_over_z_s"]),
        "log1p_z_l_observed": float(lens["log1p_z_l_observed"]),
        "log1p_z_s_observed": float(lens["log1p_z_s_observed"]),
        "source_luminosity_distance": float(source["luminosity_distance"]),
        "log_source_luminosity_distance": float(np.log(max(source["luminosity_distance"], 1e-12))),
        "sigma_v": float(lens["sigma_v"]),
        "sigma_v_true": float(lens["sigma_v_true"]),
        "sigma_v_observed": float(lens["sigma_v_observed"]),
        "log_sigma_v_observed": float(lens["log_sigma_v_observed"]),
        "theta_E(arcsec)": float(lens["theta_E_arcsec"]),
        "theta_E_observed(arcsec)": float(lens["theta_E_observed_arcsec"]),
        "theta_E_observed": float(lens["theta_E_observed_arcsec"]),
        "theta_E_observed_direct(arcsec)": float(lens["theta_E_observed_direct_arcsec"]),
        "log_theta_E": float(np.log(max(lens["theta_E_arcsec"], 1e-12))),
        "image_separation(arcsec)": float(lens["image_separation_arcsec"]),
        "image_separation_observed(arcsec)": float(lens["image_separation_observed_arcsec"]),
        "image_separation_observed": float(lens["image_separation_observed_arcsec"]),
        "theta_plus_abs_observed": float(lens["theta_plus_abs_observed"]),
        "theta_minus_abs_observed": float(lens["theta_minus_abs_observed"]),
        "image_position_asymmetry_observed": float(lens["image_position_asymmetry_observed"]),
        "u": float(lens["u"]),
        "log_u": float(lens["log_u"]),
        "y": float(lens["y"]),
        "phi": float(lens["phi"]),
        "beta": float(lens["beta"]),
        "beta_x": float(lens["beta_x"]),
        "beta_y": float(lens["beta_y"]),
        "image_x_0": float(lens["image_x_0"]),
        "image_y_0": float(lens["image_y_0"]),
        "image_x_1": float(lens["image_x_1"]),
        "image_y_1": float(lens["image_y_1"]),
        "lens_center_x": float(lens["lens_center_x"]),
        "lens_center_y": float(lens["lens_center_y"]),
        "lens_center_x_observed": float(lens["lens_center_x_observed"]),
        "lens_center_y_observed": float(lens["lens_center_y_observed"]),
        "image_x_0_observed": float(lens["image_x_0_observed"]),
        "image_y_0_observed": float(lens["image_y_0_observed"]),
        "image_x_1_observed": float(lens["image_x_1_observed"]),
        "image_y_1_observed": float(lens["image_y_1_observed"]),
        "lens_id": event_id,
        "fixed_lens_flag": 0,
        "double_image_flag": 1,
        "a21_bin": int(lens["a21_bin"]),
        "y_bin": int(lens["y_bin"]),
    }


def source_row(event_id: int, source: Dict[str, float]) -> Dict[str, float]:
    row = {"event_id": event_id}
    row.update({k: finite_or_nan(v) for k, v in source.items() if isinstance(v, (int, float, np.floating))})
    return row


def diagnostic_row(
    event_id: int,
    source: Dict[str, float],
    lens: Dict[str, float],
    sim: Dict[str, object],
) -> Dict[str, float]:
    t_d_obs = float(sim["peak_time_diff_observed"])
    K_obs = lens["t_d"] / max(lens["y"], 1e-12)
    y_from_td_obs = t_d_obs / max(K_obs, 1e-12)
    if y_from_td_obs <= 0:
        mu0_from_td = np.nan
    else:
        mu0_from_td = 1.0 + 1.0 / y_from_td_obs
    return {
        "event_id": event_id,
        "A21_true": float(lens["A21_true"]),
        "R_abs_21_true": float(lens["R_abs_21"]),
        "y_true": float(lens["y"]),
        "u_true": float(lens["u"]),
        "log_u_true": float(lens["log_u"]),
        "mu0_true": float(lens["mu_0"]),
        "mu1_true": float(lens["mu_1"]),
        "t_d_true": float(lens["t_d"]),
        "t_d_observed": t_d_obs,
        "mu0_from_observed_image_asymmetry": float(lens["mu0_from_image_asymmetry_observed"]),
        "mu0_from_observed_t_d_using_true_K": finite_or_nan(mu0_from_td),
        "snr_1": float(sim["snr_1"]),
        "snr_2": float(sim["snr_2"]),
        "snr_ratio_21": safe_ratio(float(sim["snr_2"]), float(sim["snr_1"])),
        "window_snr_like_1": float(sim["window_snr_like_1"]),
        "window_snr_like_2": float(sim["window_snr_like_2"]),
        "true_window_snr_like_1": float(sim.get("true_window_snr_like_1", np.nan)),
        "true_window_snr_like_2": float(sim.get("true_window_snr_like_2", np.nan)),
        "geocent_time_1": float(source["geocent_time"]),
        "geocent_time_2": float(source["geocent_time"] + lens["t_d"]),
    }


def append_event_arrays(
    arrays: Dict[str, List[np.ndarray]],
    sim: Dict[str, object],
    cfg: Config,
) -> None:
    ref1 = sim["image1"]["reference"]
    ref2 = sim["image2"]["reference"]
    arrays["data_white_1"].append(np.asarray(ref1["data_white"], dtype=np.float32))
    arrays["data_white_2"].append(np.asarray(ref2["data_white"], dtype=np.float32))
    arrays["h_white_1"].append(np.asarray(ref1["signal_white"], dtype=np.float32))
    arrays["h_white_2"].append(np.asarray(ref2["signal_white"], dtype=np.float32))
    arrays["noise_white_1"].append(np.asarray(ref1["noise_white"], dtype=np.float32))
    arrays["noise_white_2"].append(np.asarray(ref2["noise_white"], dtype=np.float32))

    if cfg.save_windowed_raw:
        arrays["data_raw_1"].append(np.asarray(ref1["data_raw"], dtype=np.float32))
        arrays["data_raw_2"].append(np.asarray(ref2["data_raw"], dtype=np.float32))
        arrays["h_raw_1"].append(np.asarray(ref1["signal_raw"], dtype=np.float32))
        arrays["h_raw_2"].append(np.asarray(ref2["signal_raw"], dtype=np.float32))
        arrays["noise_raw_1"].append(np.asarray(ref1["noise_raw"], dtype=np.float32))
        arrays["noise_raw_2"].append(np.asarray(ref2["noise_raw"], dtype=np.float32))

    if cfg.save_time_arrays:
        arrays["time_1"].append(np.asarray(ref1["time_window"], dtype=np.float64))
        arrays["time_2"].append(np.asarray(ref2["time_window"], dtype=np.float64))

    if cfg.save_full_strain:
        arrays["data_white_full_1"].append(np.asarray(ref1["data_white_full"], dtype=np.float32))
        arrays["data_white_full_2"].append(np.asarray(ref2["data_white_full"], dtype=np.float32))
        arrays["h_white_full_1"].append(np.asarray(ref1["signal_white_full"], dtype=np.float32))
        arrays["h_white_full_2"].append(np.asarray(ref2["signal_white_full"], dtype=np.float32))
        arrays["noise_white_full_1"].append(np.asarray(ref1["noise_white_full"], dtype=np.float32))
        arrays["noise_white_full_2"].append(np.asarray(ref2["noise_white_full"], dtype=np.float32))
        arrays["data_raw_full_1"].append(np.asarray(ref1["data_raw_full"], dtype=np.float32))
        arrays["data_raw_full_2"].append(np.asarray(ref2["data_raw_full"], dtype=np.float32))
        arrays["h_raw_full_1"].append(np.asarray(ref1["signal_raw_full"], dtype=np.float32))
        arrays["h_raw_full_2"].append(np.asarray(ref2["signal_raw_full"], dtype=np.float32))
        arrays["noise_raw_full_1"].append(np.asarray(ref1["noise_raw_full"], dtype=np.float32))
        arrays["noise_raw_full_2"].append(np.asarray(ref2["noise_raw_full"], dtype=np.float32))
        if cfg.save_time_arrays:
            arrays["time_full_1"].append(np.asarray(ref1["time_full"], dtype=np.float64))
            arrays["time_full_2"].append(np.asarray(ref2["time_full"], dtype=np.float64))


def save_array_outputs(arrays: Dict[str, List[np.ndarray]], cfg: Config) -> None:
    mapping = {
        "data_white_1": "SIS_data_strain_1.npy",
        "data_white_2": "SIS_data_strain_2.npy",
        "h_white_1": "SIS_h_strain_1.npy",
        "h_white_2": "SIS_h_strain_2.npy",
        "noise_white_1": "SIS_noise_strain_1.npy",
        "noise_white_2": "SIS_noise_strain_2.npy",
        "data_raw_1": "SIS_data_strain_raw_1.npy",
        "data_raw_2": "SIS_data_strain_raw_2.npy",
        "h_raw_1": "SIS_h_strain_raw_1.npy",
        "h_raw_2": "SIS_h_strain_raw_2.npy",
        "noise_raw_1": "SIS_noise_strain_raw_1.npy",
        "noise_raw_2": "SIS_noise_strain_raw_2.npy",
        "time_1": "SIS_time_array_1.npy",
        "time_2": "SIS_time_array_2.npy",
        "data_white_full_1": "SIS_data_strain_full_1.npy",
        "data_white_full_2": "SIS_data_strain_full_2.npy",
        "h_white_full_1": "SIS_h_strain_full_1.npy",
        "h_white_full_2": "SIS_h_strain_full_2.npy",
        "noise_white_full_1": "SIS_noise_strain_full_1.npy",
        "noise_white_full_2": "SIS_noise_strain_full_2.npy",
        "data_raw_full_1": "SIS_data_strain_raw_full_1.npy",
        "data_raw_full_2": "SIS_data_strain_raw_full_2.npy",
        "h_raw_full_1": "SIS_h_strain_raw_full_1.npy",
        "h_raw_full_2": "SIS_h_strain_raw_full_2.npy",
        "noise_raw_full_1": "SIS_noise_strain_raw_full_1.npy",
        "noise_raw_full_2": "SIS_noise_strain_raw_full_2.npy",
        "time_full_1": "SIS_time_array_full_1.npy",
        "time_full_2": "SIS_time_array_full_2.npy",
    }
    for key, filename in mapping.items():
        if key not in arrays or not arrays[key]:
            continue
        arr = np.stack(arrays[key], axis=0)
        np.save(os.path.join(cfg.save_dir, filename), arr)


def numeric_summary(series: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "min": float(x.min()),
        "p05": float(x.quantile(0.05)),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def corr_value(df: pd.DataFrame, a: str, b: str, method: str = "spearman") -> Optional[float]:
    if a not in df.columns or b not in df.columns:
        return None
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5:
        return None
    if x[mask].nunique() <= 1 or y[mask].nunique() <= 1:
        return None
    return float(x[mask].corr(y[mask], method=method))


def build_quality_report(
    cfg: Config,
    attempts: int,
    accepted: int,
    reject_counts: Dict[str, int],
    lens_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    diag_df: pd.DataFrame,
) -> Dict[str, object]:
    merged = obs_df.merge(diag_df, on="event_id", suffixes=("", "_diag"))
    report = {
        "generator": cfg.generator_name,
        "attempts": int(attempts),
        "accepted": int(accepted),
        "acceptance_rate": float(accepted / max(attempts, 1)),
        "reject_counts": dict(sorted(reject_counts.items())),
        "config": cfg_to_jsonable(cfg),
        "distributions": {
            "mu0": numeric_summary(lens_df["mu_0"]),
            "A21_true": numeric_summary(diag_df["A21_true"]),
            "y_true": numeric_summary(diag_df["y_true"]),
            "t_d_true": numeric_summary(diag_df["t_d_true"]),
            "t_d_observed": numeric_summary(diag_df["t_d_observed"]),
            "snr_1": numeric_summary(obs_df["snr_1"]),
            "snr_2": numeric_summary(obs_df["snr_2"]),
            "peak_amp_ratio_21": numeric_summary(obs_df["peak_amp_ratio_21"]),
            "rms_ratio_21": numeric_summary(obs_df["rms_ratio_21"]),
            "image_position_asymmetry_observed": numeric_summary(obs_df["image_position_asymmetry_observed"]),
        },
        "diagnostic_correlations_spearman": {
            "peak_amp_ratio_21_vs_A21_true": corr_value(merged, "peak_amp_ratio_21", "A21_true"),
            "rms_ratio_21_vs_A21_true": corr_value(merged, "rms_ratio_21", "A21_true"),
            "energy_ratio_21_vs_R_abs_21_true": corr_value(merged, "energy_ratio_21", "R_abs_21_true"),
            "env_area_ratio_21_vs_A21_true": corr_value(merged, "env_area_ratio_21", "A21_true"),
            "snr_ratio_21_vs_A21_true": corr_value(merged, "snr_ratio_21", "A21_true"),
            "image_asymmetry_vs_y_true": corr_value(merged, "image_position_asymmetry_observed", "y_true"),
            "t_d_observed_vs_t_d_true": corr_value(merged, "t_d_observed", "t_d_true"),
            "peak_time_diff_vs_t_d_true": corr_value(merged, "peak_time_diff", "t_d_true"),
        },
        "nan_counts_observable_features": {
            c: int(obs_df[c].isna().sum()) for c in obs_df.columns if obs_df[c].isna().sum() > 0
        },
    }
    return report


def write_quality_markdown(report: Dict[str, object], path: str) -> None:
    lines = [
        "# Realobs Quality Mu0 Dataset Report",
        "",
        f"Generator: `{report['generator']}`",
        f"Attempts: `{report['attempts']}`",
        f"Accepted: `{report['accepted']}`",
        f"Acceptance rate: `{report['acceptance_rate']:.4f}`",
        "",
        "## Reject Counts",
        "",
    ]
    reject_counts = report.get("reject_counts", {})
    if reject_counts:
        lines += ["| reason | count |", "|---|---:|"]
        for k, v in reject_counts.items():
            lines.append(f"| `{k}` | {v} |")
    else:
        lines.append("No rejected candidates.")

    lines += ["", "## Diagnostic Spearman Correlations", "", "| pair | corr |", "|---|---:|"]
    for k, v in report.get("diagnostic_correlations_spearman", {}).items():
        val = "NA" if v is None else f"{v:.6f}"
        lines.append(f"| `{k}` | {val} |")

    lines += ["", "## Key Distributions", ""]
    for name, summary in report.get("distributions", {}).items():
        if not summary or summary.get("n", 0) == 0:
            continue
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(
            "`min={min:.6g}, p05={p05:.6g}, median={median:.6g}, "
            "mean={mean:.6g}, p95={p95:.6g}, max={max:.6g}`".format(**summary)
        )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    global CFG
    CFG = parse_args()
    rng = seed_everything(CFG.seed)
    prepare_output_dir(CFG)

    bilby.gw.cosmology.DEFAULT_COSMOLOGY = cosmo
    bilby.gw.cosmology.COSMOLOGY = [cosmo, cosmo.name]
    bilby.gw.cosmology.get_cosmology()

    d_l_min = redshift_to_luminosity_distance(CFG.z_min)
    d_l_max = redshift_to_luminosity_distance(CFG.z_max)
    distance_prior = bilby.gw.prior.UniformComovingVolume(
        name="luminosity_distance",
        minimum=d_l_min,
        maximum=d_l_max,
    )
    waveform_generator = build_waveform_generator(CFG)

    with open(os.path.join(CFG.save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "generator": CFG.generator_name,
                "notes": (
                    "observable_features.csv contains observed features only; "
                    "diagnostic_features.csv contains privileged truth diagnostics."
                ),
                "config": cfg_to_jsonable(CFG),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    arrays: Dict[str, List[np.ndarray]] = {
        "data_white_1": [],
        "data_white_2": [],
        "h_white_1": [],
        "h_white_2": [],
        "noise_white_1": [],
        "noise_white_2": [],
        "data_raw_1": [],
        "data_raw_2": [],
        "h_raw_1": [],
        "h_raw_2": [],
        "noise_raw_1": [],
        "noise_raw_2": [],
        "time_1": [],
        "time_2": [],
        "data_white_full_1": [],
        "data_white_full_2": [],
        "h_white_full_1": [],
        "h_white_full_2": [],
        "noise_white_full_1": [],
        "noise_white_full_2": [],
        "data_raw_full_1": [],
        "data_raw_full_2": [],
        "h_raw_full_1": [],
        "h_raw_full_2": [],
        "noise_raw_full_1": [],
        "noise_raw_full_2": [],
        "time_full_1": [],
        "time_full_2": [],
    }
    source_rows: List[Dict[str, float]] = []
    lens_rows: List[Dict[str, float]] = []
    lens_param_rows: List[Dict[str, float]] = []
    obs_rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []
    rejection_rows: List[Dict[str, object]] = []
    reject_counts: Dict[str, int] = {}

    accepted = 0
    attempts = 0
    pbar = tqdm(total=CFG.target_events, desc="Accepted events")
    while accepted < CFG.target_events and attempts < CFG.max_attempts:
        attempts += 1
        source = draw_source(rng, CFG, distance_prior)
        lens, reason = draw_lens(rng, CFG, source, attempts)
        if lens is None:
            reject_counts[reason or "lens_rejected"] = reject_counts.get(reason or "lens_rejected", 0) + 1
            rejection_rows.append({"attempt": attempts, "reason": reason or "lens_rejected"})
            continue

        try:
            sim, reason = simulate_pair(rng, source, lens, waveform_generator, CFG)
        except Exception as exc:  # keep a generation run alive after rare LAL failures
            reason = f"waveform_exception_{type(exc).__name__}"
            sim = None

        if sim is None:
            reject_counts[reason or "simulation_rejected"] = reject_counts.get(reason or "simulation_rejected", 0) + 1
            rejection_rows.append({"attempt": attempts, "reason": reason or "simulation_rejected"})
            continue

        event_id = accepted
        append_event_arrays(arrays, sim, CFG)
        source_rows.append(source_row(event_id, source))
        lens_rows.append(lens_csv_row(event_id, lens))
        lens_param_rows.append(lens_params_row(event_id, source, lens))
        obs_rows.append(observable_feature_row(event_id, source, lens, sim, CFG))
        diag_rows.append(diagnostic_row(event_id, source, lens, sim))
        accepted += 1
        pbar.update(1)
    pbar.close()

    if accepted < CFG.target_events:
        raise RuntimeError(
            f"Only accepted {accepted}/{CFG.target_events} events after "
            f"{attempts} attempts. Relax quality gates or increase --max-attempts."
        )

    save_array_outputs(arrays, CFG)

    source_df = pd.DataFrame(source_rows)
    lens_df = pd.DataFrame(lens_rows)
    lens_params_df = pd.DataFrame(lens_param_rows)
    obs_df = pd.DataFrame(obs_rows)
    diag_df = pd.DataFrame(diag_rows)
    reject_df = pd.DataFrame(rejection_rows)

    source_df.to_csv(os.path.join(CFG.save_dir, "source_truth.csv"), index=False, encoding="utf-8-sig")
    source_df.to_csv(os.path.join(CFG.save_dir, "source_samples.csv"), index=False, encoding="utf-8-sig")
    lens_df.to_csv(os.path.join(CFG.save_dir, "lens.csv"), index=False, encoding="utf-8-sig")
    lens_params_df.to_csv(os.path.join(CFG.save_dir, "lens_params.csv"), index=False, encoding="utf-8-sig")
    obs_df.to_csv(os.path.join(CFG.save_dir, "observable_features.csv"), index=False, encoding="utf-8-sig")
    # Explicit deployment contract.  The full table retains diagnostics such
    # as optimal SNR and source distance for QC, while this artifact contains
    # only quantities that can be reconstructed from noisy follow-up/GW data.
    privileged = {
        "source_luminosity_distance", "snr_1", "snr_2", "snr_ratio_21",
        "snr_sum", "snr_diff", "snr_pair",
    }
    deployable_cols = [c for c in obs_df.columns if c not in privileged]
    obs_df[deployable_cols].to_csv(
        os.path.join(CFG.save_dir, "observable_features_deployable.csv"),
        index=False, encoding="utf-8-sig",
    )
    diag_df.to_csv(os.path.join(CFG.save_dir, "diagnostic_features.csv"), index=False, encoding="utf-8-sig")
    reject_df.to_csv(os.path.join(CFG.save_dir, "rejected_events.csv"), index=False, encoding="utf-8-sig")

    lensed_index = pd.DataFrame({"lensed_index": np.arange(accepted, dtype=np.int64)})
    lensed_index.to_csv(os.path.join(CFG.save_dir, "lensed_index.csv"), index=False)

    lens_obs_cols = [
        "event_id",
        "feature_data_kind",
        "z_s_observed",
        "z_l_observed",
        "z_l_over_z_s",
        "log1p_z_s_observed",
        "log1p_z_l_observed",
        "sigma_v_observed",
        "log_sigma_v_observed",
        "theta_E_observed",
        "image_separation_observed",
        "theta_plus_abs_observed",
        "theta_minus_abs_observed",
        "image_position_asymmetry_observed",
    ]
    obs_df[lens_obs_cols].to_csv(
        os.path.join(CFG.save_dir, "lens_observable_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    report = build_quality_report(
        CFG,
        attempts=attempts,
        accepted=accepted,
        reject_counts=reject_counts,
        lens_df=lens_df,
        obs_df=obs_df,
        diag_df=diag_df,
    )
    with open(os.path.join(CFG.save_dir, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_quality_markdown(report, os.path.join(CFG.save_dir, "quality_report.md"))

    print("=" * 90)
    print(f"Saved SIS GW physics-baseline mu0 dataset to: {CFG.save_dir}")
    print(f"Accepted {accepted} events after {attempts} attempts")
    print("Core training files:")
    print("  lens.csv")
    print("  lens_params.csv")
    print("  observable_features.csv")
    print("  observable_features_deployable.csv  (strict observed-input contract)")
    print("  SIS_data_strain_1.npy / SIS_data_strain_2.npy")
    print("Diagnostics:")
    print("  diagnostic_features.csv")
    print("  quality_report.json")
    print("  quality_report.md")
    print("=" * 90)


if __name__ == "__main__":
    main()

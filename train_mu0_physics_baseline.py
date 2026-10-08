#!/usr/bin/env python
# coding: utf-8
"""
Train a mu0 predictor on the SIS GW physical-baseline dataset.

This entry point reuses the compact waveform-plus-observables model and training loop,
with data paths and feature options adapted to this generator:

    SIS_GW_physics_baseline.py

The generator simulates idealized SIS geometric optics rather than a complete survey or detector
pipeline. Its default columns are compatible with the compact waveform-plus-observables model.
"""

from __future__ import annotations
from pathlib import Path as _ReleasePath

import argparse
import os
import sys
from typing import List


# =============================================================================
# User configuration
# Edit this section for routine runs. Command-line arguments override the corresponding
# paths, feature presets and training parameters below.
# =============================================================================

# Data and output paths
DATA_ROOT = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_sis_gw_physics_baseline')
OUT_DIR = str(_ReleasePath(__file__).resolve().parents[0] / 'runs/mu_direct_physics_baseline_waveobs')
# Use "all" to control feature groups independently. Other presets first select
# an ablation subset; the options below can only remove optical groups from that subset.
FEATURE_PRESET = "all"

# Observable-feature options
# These inputs require optical or infrared follow-up and are not supplied by ET alone.
USE_OPTICAL_REDSHIFT = True
USE_OPTICAL_SIGMA_V = True
USE_OPTICAL_IMAGE_GEOMETRY = False
USE_OPTICAL_DISTANCE = False

# Privileged-information (oracle) options
# Keep False for ordinary training. For explicit oracle experiments, enable both
# the selected feature option and ALLOW_ORACLE_FEATURES.
USE_ORACLE_TD = False
USE_ORACLE_BETA = False
USE_ORACLE_R_ABS = False
USE_ORACLE_AMP_RATIO = False
USE_ORACLE_Y = False
ALLOW_ORACLE_FEATURES = False

# Optimal SNR is computed from the noiseless injection and disabled by default; it differs from
# the SNR estimated by a search pipeline from observed data.
USE_OPTIMAL_SNR_FEATURES = False

# Core training parameters
RANDOM_SEED = 42
EPOCHS = 160
BATCH_SIZE = 32
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 5.0e-4
TRAIN_FRACTION = 0.8
TARGET_LEN = 4096
RAW_SCALE = 0.01
TARGET_MODE = "log_mu0_minus1"

# Model architecture parameters
WAVE_NORM_MODE = "pair"
WAVE_WIDTH = 64
D_MODEL = 192
OBS_HIDDEN = 128
ACTIVATION = "relu"

# Loss, EMA, learning-rate scheduler and early-stopping parameters
LOSS_W_MAE = 0.02
LOSS_W_REL = 0.08
LOSS_W_TRANS = 1.0
USE_TAIL_WEIGHT = True
TAIL_ALPHA = 0.35
EMA_DECAY = 0.995
EARLY_STOP_PATIENCE = 14
MIN_EPOCHS_BEFORE_STOP = 50
MIN_LEARNING_RATE = 5.0e-7


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_mu0_from_wave_obs_randomu_a21friendly_realobs as base  # noqa: E402


BASE_ADD_OPTIONAL_OBSERVABLES = base.add_optional_observables


PHYSICS_FEATURE_KIND_NOISY_WHITE = "sis_physics_baseline_noisy_whitened"


# Default feature columns retained for compatibility with earlier models.
LEGACY_WAVE_COLS = [
    "peak_time_diff",
    "env_peak_time_diff",
    "peak_amp_ratio_21",
    "rms_ratio_21",
    "energy_ratio_21",
    "xcorr_lag",
    "norm_xcorr_max",
    "xcorr_lag_win65",
    "norm_xcorr_max_win65",
    "peak_amp_ratio_local_65",
    "rms_ratio_local_65",
    "energy_ratio_local_65",
    "env_area_ratio_21",
    "aligned_l1_residual",
    "aligned_l2_residual",
    "spec_centroid_diff",
    "band_energy_ratio_low_21",
    "band_energy_ratio_mid_21",
    "band_energy_ratio_high_21",
]


TIME_OBS_COLS = [
    "peak_time_diff",
    "env_peak_time_diff",
    "arrival_time_1_sigma",
    "arrival_time_2_sigma",
]


A21_PROXY_COLS = [
    "peak_amp_ratio_21",
    "rms_ratio_21",
    "energy_ratio_21",
    "peak_amp_ratio_local_65",
    "rms_ratio_local_65",
    "energy_ratio_local_65",
    "env_area_ratio_21",
    "win0p25_peak_amp_ratio_21",
    "win0p25_rms_ratio_21",
    "win0p25_energy_ratio_21",
    "win0p25_env_area_ratio_21",
    "win0p5_peak_amp_ratio_21",
    "win0p5_rms_ratio_21",
    "win0p5_energy_ratio_21",
    "win0p5_env_area_ratio_21",
    "win0p9_peak_amp_ratio_21",
    "win0p9_rms_ratio_21",
    "win0p9_energy_ratio_21",
    "win0p9_env_area_ratio_21",
]


WAVE_SHAPE_COLS = [
    "xcorr_lag",
    "norm_xcorr_max",
    "xcorr_lag_win65",
    "norm_xcorr_max_win65",
    "aligned_l1_residual",
    "aligned_l2_residual",
    "spec_centroid_diff",
    "band_energy_ratio_low_21",
    "band_energy_ratio_mid_21",
    "band_energy_ratio_high_21",
]


NOISE_CONTEXT_COLS = [
    "local_noise_rms_1",
    "local_noise_rms_2",
]


OPTICAL_REDSHIFT_COLS = [
    "z_s_observed",
    "z_l_observed",
    "z_l_over_z_s",
    "log1p_z_s_observed",
    "log1p_z_l_observed",
]


OPTICAL_SIGMA_COLS = [
    "sigma_v_observed",
    "log_sigma_v_observed",
]


OPTICAL_SCALE_COLS = [
    "theta_E_observed",
    "image_separation_observed",
]


OPTICAL_DISTANCE_COLS = [
    "source_luminosity_distance",
]


IMAGE_POSITION_COLS = [
    "theta_plus_abs_observed",
    "theta_minus_abs_observed",
    "image_position_asymmetry_observed",
    "lens_center_x_observed",
    "lens_center_y_observed",
    "image_x_0_observed",
    "image_y_0_observed",
    "image_x_1_observed",
    "image_y_1_observed",
]


OPTIMAL_SNR_COLS = [
    "snr_1",
    "snr_2",
    "snr_ratio_21",
    "snr_sum",
    "snr_diff",
]


ORACLE_TD_COLS = ["t_d"]
ORACLE_BETA_COLS = ["beta_true"]
ORACLE_R_ABS_COLS = ["R_abs_21"]
ORACLE_AMP_RATIO_COLS = ["amp_ratio_21_true"]
ORACLE_Y_COLS = ["y_true"]


def unique_cols(cols: List[str]) -> List[str]:
    out = []
    seen = set()
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def resolve_obs_path_physics():
    """Return the observable-feature table path and expected feature type."""
    cfg = base.cfg
    if not (cfg.USE_NOISY_WAVE and cfg.USE_WHITENED_WAVE):
        raise RuntimeError(
            "train_mu0_physics_baseline.py currently supports the default "
            "noisy whitened physics-baseline table only."
        )

    path = base.resolve_data_path(cfg.OBS_CSV_NOISY_WHITE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing physics-baseline observable feature table: {path}. "
            "Generate it with /root/autodl-tmp/tmp/SIS_GW_physics_baseline.py."
        )
    return path, PHYSICS_FEATURE_KIND_NOISY_WHITE


def add_physics_optional_observables(obs_df, lens_params_df):
    """Add enabled optical or oracle columns to the observable-feature table."""
    obs_df = BASE_ADD_OPTIONAL_OBSERVABLES(obs_df, lens_params_df)
    cfg = base.cfg

    oracle_enabled = any(
        (
            cfg.USE_ORACLE_TD,
            cfg.USE_ORACLE_BETA,
            cfg.USE_ORACLE_R_ABS,
            cfg.USE_ORACLE_AMP_RATIO,
            cfg.USE_ORACLE_Y,
        )
    )
    if not oracle_enabled:
        return obs_df

    lens_df = base.pd.read_csv(base.resolve_data_path(cfg.LENS_CSV))
    if len(lens_df) < len(obs_df):
        raise RuntimeError(
            "lens.csv is shorter than the observable table; cannot align oracle columns."
        )
    lens_df = lens_df.iloc[: len(obs_df)].reset_index(drop=True)
    lens_params_df = lens_params_df.iloc[: len(obs_df)].reset_index(drop=True)

    def add_column(output_name, source_df, source_name):
        if output_name in obs_df.columns:
            return
        if source_name not in source_df.columns:
            raise RuntimeError(
                f"Missing source column for oracle feature {output_name}: {source_name}"
            )
        obs_df[output_name] = source_df[source_name].to_numpy()

    if cfg.USE_ORACLE_TD:
        add_column("t_d", lens_df, "t_d")
    if cfg.USE_ORACLE_BETA:
        add_column("beta_true", lens_params_df, "beta")
    if cfg.USE_ORACLE_R_ABS:
        add_column("R_abs_21", lens_df, "R_abs_21")
    if cfg.USE_ORACLE_AMP_RATIO:
        add_column("amp_ratio_21_true", lens_df, "amp_ratio_21")
    if cfg.USE_ORACLE_Y:
        add_column("y_true", lens_df, "y")

    return obs_df


def build_quality_obs_cols(
    preset: str,
    use_optimal_snr: bool = False,
    use_optical_redshift: bool = True,
    use_optical_sigma_v: bool = True,
    use_optical_image_geometry: bool = True,
    use_optical_distance: bool = False,
    use_oracle_td: bool = False,
    use_oracle_beta: bool = False,
    use_oracle_r_abs: bool = False,
    use_oracle_amp_ratio: bool = False,
    use_oracle_y: bool = False,
) -> List[str]:
    """Return observable columns for the selected controlled-experiment preset."""
    if preset == "legacy":
        cols = list(LEGACY_WAVE_COLS)
        cols += OPTICAL_REDSHIFT_COLS + OPTICAL_SIGMA_COLS + OPTICAL_SCALE_COLS
    elif preset == "a21_wave":
        cols = A21_PROXY_COLS + WAVE_SHAPE_COLS + NOISE_CONTEXT_COLS
    elif preset == "time_lens":
        cols = TIME_OBS_COLS + OPTICAL_REDSHIFT_COLS + OPTICAL_SIGMA_COLS + OPTICAL_SCALE_COLS
    elif preset in {"gw_only", "no_optical"}:
        cols = A21_PROXY_COLS + WAVE_SHAPE_COLS + NOISE_CONTEXT_COLS + TIME_OBS_COLS
    elif preset == "no_image":
        cols = (
            A21_PROXY_COLS
            + WAVE_SHAPE_COLS
            + NOISE_CONTEXT_COLS
            + TIME_OBS_COLS
            + OPTICAL_REDSHIFT_COLS
            + OPTICAL_SIGMA_COLS
            + OPTICAL_SCALE_COLS
        )
    elif preset == "no_time":
        cols = (
            A21_PROXY_COLS
            + WAVE_SHAPE_COLS
            + NOISE_CONTEXT_COLS
            + OPTICAL_REDSHIFT_COLS
            + OPTICAL_SIGMA_COLS
            + OPTICAL_SCALE_COLS
            + IMAGE_POSITION_COLS
        )
    elif preset == "all":
        cols = (
            A21_PROXY_COLS
            + WAVE_SHAPE_COLS
            + NOISE_CONTEXT_COLS
            + TIME_OBS_COLS
            + OPTICAL_REDSHIFT_COLS
            + OPTICAL_SIGMA_COLS
            + OPTICAL_SCALE_COLS
            + IMAGE_POSITION_COLS
        )
    else:
        raise ValueError(f"Unsupported preset: {preset}")

    disabled_cols = set()
    if not use_optical_redshift:
        disabled_cols.update(OPTICAL_REDSHIFT_COLS)
    if not use_optical_sigma_v:
        disabled_cols.update(OPTICAL_SIGMA_COLS)
    if not use_optical_image_geometry:
        disabled_cols.update(OPTICAL_SCALE_COLS)
        disabled_cols.update(IMAGE_POSITION_COLS)
    cols = [col for col in cols if col not in disabled_cols]

    if use_optical_distance:
        cols += OPTICAL_DISTANCE_COLS
    if use_optimal_snr:
        cols += OPTIMAL_SNR_COLS
    if use_oracle_td:
        cols += ORACLE_TD_COLS
    if use_oracle_beta:
        cols += ORACLE_BETA_COLS
    if use_oracle_r_abs:
        cols += ORACLE_R_ABS_COLS
    if use_oracle_amp_ratio:
        cols += ORACLE_AMP_RATIO_COLS
    if use_oracle_y:
        cols += ORACLE_Y_COLS
    return unique_cols(cols)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mu0 on the SIS GW physics-baseline data."
    )
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument(
        "--preset",
        type=str,
        default=FEATURE_PRESET,
        choices=[
            "all",
            "gw_only",
            "no_optical",
            "no_image",
            "no_time",
            "a21_wave",
            "time_lens",
            "legacy",
        ],
        help=(
            "Observable feature preset. all uses every safe observed chain; "
            "gw_only/no_optical removes all optical observables; "
            "no_image/no_time are ablations; a21_wave isolates waveform amplitude-ratio proxies; "
            "time_lens isolates time-delay plus lens-scale observables."
        ),
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--target-len", type=int, default=TARGET_LEN)
    parser.add_argument("--raw-scale", type=float, default=RAW_SCALE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--loss-w-trans", type=float, default=LOSS_W_TRANS)
    parser.add_argument("--loss-w-mae", type=float, default=LOSS_W_MAE)
    parser.add_argument("--loss-w-rel", type=float, default=LOSS_W_REL)
    parser.add_argument(
        "--tail-weight",
        action=argparse.BooleanOptionalAction,
        default=USE_TAIL_WEIGHT,
        help="Enable or disable high-mu0 tail weighting in physical-space loss terms.",
    )
    parser.add_argument("--tail-alpha", type=float, default=TAIL_ALPHA)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--min-epochs-before-stop", type=int, default=MIN_EPOCHS_BEFORE_STOP)
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional leaf directory and output-file suffix for automated experiment grids.",
    )
    parser.add_argument(
        "--use-optimal-snr",
        action=argparse.BooleanOptionalAction,
        default=USE_OPTIMAL_SNR_FEATURES,
    )
    parser.add_argument(
        "--allow-oracle",
        action=argparse.BooleanOptionalAction,
        default=ALLOW_ORACLE_FEATURES,
    )
    parser.add_argument(
        "--optical-redshift",
        action=argparse.BooleanOptionalAction,
        default=USE_OPTICAL_REDSHIFT,
    )
    parser.add_argument(
        "--optical-sigma-v",
        action=argparse.BooleanOptionalAction,
        default=USE_OPTICAL_SIGMA_V,
    )
    parser.add_argument(
        "--optical-image-geometry",
        action=argparse.BooleanOptionalAction,
        default=USE_OPTICAL_IMAGE_GEOMETRY,
    )
    parser.add_argument(
        "--optical-distance",
        action=argparse.BooleanOptionalAction,
        default=USE_OPTICAL_DISTANCE,
    )
    return parser.parse_args()


def configure_base(args) -> None:
    cfg = base.cfg

    if args.epochs <= 0:
        raise ValueError("EPOCHS/--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("BATCH_SIZE/--batch-size must be positive.")
    if args.lr <= 0.0:
        raise ValueError("LEARNING_RATE/--lr must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("WEIGHT_DECAY/--weight-decay must be non-negative.")
    if args.target_len <= 0:
        raise ValueError("TARGET_LEN/--target-len must be positive.")
    if args.raw_scale <= 0.0:
        raise ValueError("RAW_SCALE/--raw-scale must be positive.")
    if args.loss_w_trans < 0.0 or args.loss_w_mae < 0.0 or args.loss_w_rel < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    if args.tail_alpha < 0.0:
        raise ValueError("--tail-alpha must be non-negative.")
    if args.ema_decay <= 0.0 or args.ema_decay >= 1.0:
        raise ValueError("--ema-decay must be in (0, 1).")
    if args.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be positive.")
    if args.min_epochs_before_stop < 0:
        raise ValueError("--min-epochs-before-stop must be non-negative.")
    if args.min_lr <= 0.0:
        raise ValueError("--min-lr must be positive.")
    if not 0.0 < TRAIN_FRACTION < 1.0:
        raise ValueError("TRAIN_FRACTION must be between 0 and 1.")

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    run_name = args.run_name or args.preset
    preset_out_dir = os.path.join(out_dir, run_name)

    cfg.DATA_ROOT = data_root
    cfg.LENS_CSV = os.path.join(data_root, "lens.csv")
    cfg.LENS_PARAMS_CSV = os.path.join(data_root, "lens_params.csv")
    cfg.OBS_CSV_NOISY_WHITE = os.path.join(data_root, "observable_features.csv")
    cfg.OBS_CSV_CLEAN_WHITE = os.path.join(data_root, "observable_features_clean.csv")
    cfg.OBS_CSV_NOISY_RAW = os.path.join(data_root, "observable_features_raw.csv")
    cfg.OBS_CSV_CLEAN_RAW = os.path.join(data_root, "observable_features_clean_raw.csv")
    cfg.OUT_DIR = preset_out_dir

    cfg.USE_NOISY_WAVE = True
    cfg.USE_WHITENED_WAVE = True
    cfg.WAVE_NORM_MODE = WAVE_NORM_MODE
    cfg.TARGET_LEN = args.target_len
    cfg.USE_OPTIMAL_SNR_FEATURES = bool(args.use_optimal_snr)
    cfg.ALLOW_ORACLE_FEATURES = bool(args.allow_oracle)

    optical_presets = {"all", "no_image", "no_time", "time_lens", "legacy"}
    preset_uses_optical = args.preset in optical_presets
    cfg.USE_OPTICAL_REDSHIFT = bool(args.optical_redshift) and preset_uses_optical
    cfg.USE_OPTICAL_SIGMA_V = bool(args.optical_sigma_v) and preset_uses_optical
    cfg.USE_OPTICAL_IMAGE_GEOMETRY = bool(args.optical_image_geometry) and preset_uses_optical
    cfg.USE_OPTICAL_DISTANCE = bool(args.optical_distance)

    cfg.USE_ORACLE_TD = USE_ORACLE_TD
    cfg.USE_ORACLE_BETA = USE_ORACLE_BETA
    cfg.USE_ORACLE_R_ABS = USE_ORACLE_R_ABS
    cfg.USE_ORACLE_AMP_RATIO = USE_ORACLE_AMP_RATIO
    cfg.USE_ORACLE_Y = USE_ORACLE_Y

    cfg.SEED = args.seed
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LR = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.SPLIT = {"train": TRAIN_FRACTION, "val": 1.0 - TRAIN_FRACTION}
    cfg.RAW_SCALE = args.raw_scale
    cfg.TARGET_MODE = TARGET_MODE

    cfg.WAVE_WIDTH = WAVE_WIDTH
    cfg.D_MODEL = D_MODEL
    cfg.OBS_HIDDEN = OBS_HIDDEN
    cfg.ACTIVATION = ACTIVATION

    cfg.LOSS_W_TRANS = args.loss_w_trans
    cfg.LOSS_W_MAE = args.loss_w_mae
    cfg.LOSS_W_REL = args.loss_w_rel
    cfg.USE_TAIL_WEIGHT = bool(args.tail_weight)
    cfg.TAIL_ALPHA = args.tail_alpha
    cfg.EMA_DECAY = args.ema_decay
    cfg.EARLY_STOP_PATIENCE = args.early_stop_patience
    cfg.MIN_EPOCHS_BEFORE_STOP = args.min_epochs_before_stop
    cfg.MIN_LR = args.min_lr

    cfg.CKPT_NAME = f"best_mu_direct_physics_baseline_{run_name}.pt"
    cfg.PLOT_NAME = f"mu_direct_physics_baseline_{run_name}_scatter.png"
    cfg.CSV_NAME = f"mu_direct_physics_baseline_{run_name}_val_predictions.csv"
    cfg.HISTORY_CSV_NAME = f"training_history_{run_name}.csv"
    cfg.SUMMARY_JSON_NAME = f"training_summary_{run_name}.json"

    cfg.QUALITY_PRESET = args.preset
    cfg.PHYSICS_PRESET = args.preset
    cfg.RUN_NAME = run_name

    def selected_obs_cols_quality() -> List[str]:
        return build_quality_obs_cols(
            preset=cfg.QUALITY_PRESET,
            use_optimal_snr=cfg.USE_OPTIMAL_SNR_FEATURES,
            use_optical_redshift=cfg.USE_OPTICAL_REDSHIFT,
            use_optical_sigma_v=cfg.USE_OPTICAL_SIGMA_V,
            use_optical_image_geometry=cfg.USE_OPTICAL_IMAGE_GEOMETRY,
            use_optical_distance=cfg.USE_OPTICAL_DISTANCE,
            use_oracle_td=cfg.USE_ORACLE_TD,
            use_oracle_beta=cfg.USE_ORACLE_BETA,
            use_oracle_r_abs=cfg.USE_ORACLE_R_ABS,
            use_oracle_amp_ratio=cfg.USE_ORACLE_AMP_RATIO,
            use_oracle_y=cfg.USE_ORACLE_Y,
        )

    base.resolve_obs_path = resolve_obs_path_physics
    base.selected_obs_cols = selected_obs_cols_quality
    base.add_optional_observables = add_physics_optional_observables


def main():
    args = parse_args()
    configure_base(args)
    print("=" * 90)
    print("SIS GW physics-baseline mu0 training entry")
    print(f"Data root: {base.cfg.DATA_ROOT}")
    print(f"Out dir:   {base.cfg.OUT_DIR}")
    print(f"Preset:    {base.cfg.QUALITY_PRESET}")
    print(f"Epochs:    {base.cfg.EPOCHS}")
    print(f"Batch:     {base.cfg.BATCH_SIZE}")
    print(f"LR:        {base.cfg.LR:.6g}")
    print(f"Seed:      {base.cfg.SEED}")
    print(
        f"Loss:      {base.cfg.LOSS_W_TRANS:g}*trans + {base.cfg.LOSS_W_MAE:g}*MAE + "
        f"{base.cfg.LOSS_W_REL:g}*MAPE"
    )
    print(f"Tail:      {base.cfg.USE_TAIL_WEIGHT} alpha={base.cfg.TAIL_ALPHA:g}")
    print(f"Train/val: {base.cfg.SPLIT['train']:.2f}/{base.cfg.SPLIT['val']:.2f}")
    print(f"Obs cols:  {len(base.selected_obs_cols())}")
    print("Selected columns:")
    for col in base.selected_obs_cols():
        print(f"  - {col}")
    print("=" * 90)
    base.main()


if __name__ == "__main__":
    main()

from pathlib import Path as _ReleasePath
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# config
# ============================================================
@dataclass
class Config:
    # ======================================================
    # Basic runtime settings
    # ======================================================
    # Random seed; use a fixed integer for reproducibility.
    SEED: int = 42

    # Prefer GPU execution; fall back to CPU when CUDA is unavailable.
    USE_CUDA: bool = True

    # ======================================================
    # Data paths
    # ======================================================
    # DATA_ROOT corresponds to SAVE_DIR in SIS_GW_events_randomu_a21friendly_realobs.py.
    DATA_ROOT: str = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_randomu_a21friendly_realobs')
    LENS_CSV: str = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_randomu_a21friendly_realobs/lens.csv')
    LENS_PARAMS_CSV: str = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_randomu_a21friendly_realobs/lens_params.csv')
    OBS_CSV_NOISY_WHITE: str = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_randomu_a21friendly_realobs/observable_features.csv')
    OBS_CSV_CLEAN_WHITE: str = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_randomu_a21friendly_realobs/observable_features_clean.csv')
    OBS_CSV_NOISY_RAW: str = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_randomu_a21friendly_realobs/observable_features_raw.csv')
    OBS_CSV_CLEAN_RAW: str = str(_ReleasePath(__file__).resolve().parents[0] / 'data_generation/data_lens_randomu_a21friendly_realobs/observable_features_clean_raw.csv')

    # Use a separate output directory for each experiment to avoid overwriting checkpoints.
    OUT_DIR: str = str(_ReleasePath(__file__).resolve().parents[0] / 'runs/mu_direct_randomu_a21friendly_realobs_waveobs')

    # ======================================================
    # Waveform input options
    # ======================================================
    # True: use noisy observed waveforms from SIS_data_strain_*.npy.
    # False: use clean SIS_h_strain_*.npy waveforms for idealized or ablation experiments.
    USE_NOISY_WAVE: bool = True

    # True: use whitened waveforms; False: use raw time-domain waveforms.
    # True is the default; raw waveforms have a wider amplitude range and may require a different RAW_SCALE.
    USE_WHITENED_WAVE: bool = True

    # Waveform normalization modes:
    # "pair"  : share the mean and standard deviation between images to preserve relative amplitudes (default).
    # "fixed" : divide by RAW_SCALE without per-sample standardization to retain absolute amplitudes.
    # "none"  : use unnormalized waveforms; intended for debugging because training may be unstable.
    WAVE_NORM_MODE: str = "pair"

    # Fixed amplitude scale for the second waveform channel, tanh(x / RAW_SCALE).
    # This channel retains absolute amplitudes; increase the scale if saturated, or decrease it if values are too small.
    # Example scales for whitened data: 0.003, 0.01, 0.03, 0.1; tune raw data using its RMS.
    RAW_SCALE: float = 0.01

    # ======================================================
    # Optical follow-up input options
    # ======================================================
    # Optional lens observables include redshifts, velocity dispersion, Einstein radius and image separation.
    # These are observables, not truth variables such as y/u/mu/R/amp_ratio_true.

    # Include z_s_observed, z_l_observed and their transforms.
    # Enable redshifts for GW+optical experiments with spectroscopic measurements.
    USE_OPTICAL_REDSHIFT: bool = True

    # Include sigma_v_observed.
    # sigma_v is not measured directly by GW detectors; enable it for simulated lens spectroscopy.
    USE_OPTICAL_SIGMA_V: bool = True

    # Include theta_E_observed / image_separation_observed.
    # Requires identified lens images; SIS geometry strongly constrains the prediction.
    USE_OPTICAL_IMAGE_GEOMETRY: bool = True

    # Include source_luminosity_distance.
    # Enable only when an external distance constraint is explicitly assumed.
    USE_OPTICAL_DISTANCE: bool = False

    # SIS_optimal_SNR_*.npy contains optimal SNR computed from the clean injected response.
    # It may leak target information in the A21-friendly data and is disabled by default.
    USE_OPTIMAL_SNR_FEATURES: bool = False

    # ======================================================
    # Oracle / ablation options
    # ======================================================
    # The following truth inputs are not GW observables and are disabled by default.
    # Enable only to test performance when truth or near-truth inputs are supplied.

    # True time delay t_d; use only for oracle comparisons against peak_time_diff.
    USE_ORACLE_TD: bool = False

    # True source position beta; strongly related to y/mu and contains target information.
    USE_ORACLE_BETA: bool = False

    # True magnification ratio R = abs(mu1) / mu0, also stored as mu_ratio_abs_21/R_abs_21.
    # For SIS, mu0 = 2 / (1 - R), so this almost directly supplies the target.
    USE_ORACLE_R_ABS: bool = False

    # True image amplitude ratio A21 = sqrt(abs(mu1) / mu0) = sqrt(R).
    # For SIS, mu0 = 2 / (1 - A21^2), making this a strong oracle input.
    USE_ORACLE_AMP_RATIO: bool = False

    # True normalized source position y = beta / theta_E.
    # For SIS, mu0 = 1 + 1 / y, so this supplies target information directly.
    USE_ORACLE_Y: bool = False

    # Guard against truth leakage; set True only for explicit oracle experiments.
    ALLOW_ORACLE_FEATURES: bool = False

    # ======================================================
    # Waveform cropping options
    # ======================================================
    # Length after cropping or padding; 4096 samples are approximately one second of whitened data.
    # 2048 samples are faster but shorter; 8192 retain more context at greater cost.
    TARGET_LEN: int = 4096

    # Downsampling stride: 1 keeps all samples; 2/4 reduce cost but lose high-frequency detail.
    STRIDE: int = 1

    # Default target: log(mu0 - 1)
    # Since mu0 = 1 + 1/y, log(mu0-1) = log(1/y).
    # Targets: "log_mu0_minus1" (default), "log_mu0" (weaker compression), or "mu0" (harder direct regression).
    TARGET_MODE: str = "log_mu0_minus1"

    # ======================================================
    # Model capacity
    # ======================================================
    # Activation: "relu" is faster; other values select the smoother but slower SiLU.
    ACTIVATION: str = "relu"

    # Base 1D CNN width: 32 for a small model, 64 by default, or 96/128 for larger models.
    WAVE_WIDTH: int = 64

    # Fusion MLP width; examples: 128, 192, 256.
    D_MODEL: int = 192

    # Observable-feature MLP hidden width; examples: 64, 128, 256.
    OBS_HIDDEN: int = 128

    # Exclude truth inputs: mu_0, mu_1, u_true, log_u_true, y_true, t_d,
    # beta_true, theta_E_true and sigma_v_true.
    # Use noisy-waveform statistics to retain observed amplitude information related to A21.
    # Optimal SNR is disabled by default because it is computed from the clean injection.
    BASE_OBS_COLS: List[str] = field(default_factory=lambda: [
        "peak_time_diff", "env_peak_time_diff",

        "peak_amp_ratio_21", "rms_ratio_21", "energy_ratio_21",

        "xcorr_lag", "norm_xcorr_max",
        "xcorr_lag_win65", "norm_xcorr_max_win65",

        "peak_amp_ratio_local_65", "rms_ratio_local_65", "energy_ratio_local_65",

        "env_area_ratio_21",

        "aligned_l1_residual", "aligned_l2_residual",

        "spec_centroid_diff",
        "band_energy_ratio_low_21",
        "band_energy_ratio_mid_21",
        "band_energy_ratio_high_21",
    ])

    OPTIMAL_SNR_COLS: List[str] = field(default_factory=lambda: [
        "snr_1", "snr_2", "snr_ratio_21", "snr_sum", "snr_diff",
    ])

    # USE_OPTICAL_REDSHIFT=True : append these columns when enabled.
    OPTICAL_REDSHIFT_COLS: List[str] = field(default_factory=lambda: [
        "z_s_observed", "z_l_observed", "z_l_over_z_s",
        "log1p_z_s_observed", "log1p_z_l_observed",
    ])

    # USE_OPTICAL_SIGMA_V=True : append these columns when enabled.
    OPTICAL_SIGMA_V_COLS: List[str] = field(default_factory=lambda: [
        "sigma_v_observed", "log_sigma_v_observed",
    ])

    # USE_OPTICAL_IMAGE_GEOMETRY=True : append these columns when enabled.
    OPTICAL_IMAGE_GEOMETRY_COLS: List[str] = field(default_factory=lambda: [
        "theta_E_observed", "image_separation_observed",
    ])

    # USE_OPTICAL_DISTANCE=True : append these columns when enabled.
    OPTICAL_DISTANCE_COLS: List[str] = field(default_factory=lambda: [
        "source_luminosity_distance",
    ])

    # USE_ORACLE_TD=True : append these columns when enabled.
    ORACLE_TD_COLS: List[str] = field(default_factory=lambda: [
        "t_d",
    ])

    # USE_ORACLE_BETA=True : append these columns when enabled.
    ORACLE_BETA_COLS: List[str] = field(default_factory=lambda: [
        "beta_true",
    ])

    # Append when USE_ORACLE_R_ABS=True; R_abs_21 and mu_ratio_abs_21 are equivalent columns.
    # Use R_abs_21 to match the notation R = abs(mu1) / mu0.
    ORACLE_R_ABS_COLS: List[str] = field(default_factory=lambda: [
        "R_abs_21",
    ])

    # USE_ORACLE_AMP_RATIO=True : append these columns when enabled.
    ORACLE_AMP_RATIO_COLS: List[str] = field(default_factory=lambda: [
        "amp_ratio_21_true",
    ])

    # USE_ORACLE_Y=True : append these columns when enabled.
    ORACLE_Y_COLS: List[str] = field(default_factory=lambda: [
        "y_true",
    ])

    # ======================================================
    # Training hyperparameters
    # ======================================================
    # Maximum epochs with early stopping; examples: 80, 120, 160, 240.
    EPOCHS: int = 160

    # Batch size; reduce to 16 if memory is limited, or try 64 if available.
    BATCH_SIZE: int = 32

    # Initial AdamW learning rate; examples: 3e-5, 1e-4, 3e-4.
    LR: float = 1e-4

    # Weight decay; examples: 1e-5, 1e-4, 5e-4, 1e-3.
    WEIGHT_DECAY: float = 5e-4

    # EMA smoothing coefficient; examples: 0.99, 0.995, 0.999.
    EMA_DECAY: float = 0.995

    # Train/validation fractions: 0.8/0.2 for small datasets, or 0.9/0.1 for larger datasets.
    SPLIT: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})

    # Augmentation is disabled by default because amplitudes and delays carry information.
    # If enabled, use small shifts or noise to preserve physical amplitude relations.
    USE_AUGMENTATION: bool = False

    # Per-sample augmentation probability; examples: 0.1-0.3.
    AUG_PROB: float = 0.25

    # Maximum shared image shift in samples; preserve the relative timing.
    AUG_ROLL_MAX: int = 8

    # Augmentation noise divisor; larger values mean weaker noise. Examples: 50, 100, 200.
    AUG_NOISE_DIV: float = 100.0

    # loss = LOSS_W_TRANS * transformed-space MAE
    #        + LOSS_W_MAE * real-space MAE
    #        + LOSS_W_REL * real-space MAPE.
    # For direct mu regression, consider LOSS_W_REL in 0.02-0.2 to weight relative errors.
    LOSS_W_TRANS: float = 1.0
    LOSS_W_MAE: float = 0.02
    LOSS_W_REL: float = 0.08

    # Optionally weight high-magnification samples, which contribute strongly to RMSE.
    USE_TAIL_WEIGHT: bool = True

    # High-mu sample weighting strength; examples: 0.0, 0.2, 0.35, 0.5.
    TAIL_ALPHA: float = 0.35

    # Early stopping requires MIN_EPOCHS_BEFORE_STOP and consecutive epochs without improvement.
    EARLY_STOP_PATIENCE: int = 14
    MIN_EPOCHS_BEFORE_STOP: int = 50

    # Minimum learning rate for ReduceLROnPlateau.
    MIN_LR: float = 5e-7

    # Output filenames.
    CKPT_NAME: str = "best_mu_direct_realobs_waveobs.pt"
    PLOT_NAME: str = "mu_direct_realobs_waveobs_scatter.png"
    CSV_NAME: str = "mu_direct_realobs_waveobs_val_predictions.csv"
    HISTORY_CSV_NAME: str = "training_history.csv"
    SUMMARY_JSON_NAME: str = "training_summary.json"

    # Numerical stability constant.
    EPS: float = 1e-8


cfg = Config()


# ============================================================
# Utility functions
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_out_dir():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)


def resolve_data_path(path: str) -> str:
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    cand = os.path.join(cfg.DATA_ROOT, base)
    return cand if os.path.exists(cand) else path


def selected_obs_cols() -> List[str]:
    cols = list(cfg.BASE_OBS_COLS)
    if cfg.USE_OPTIMAL_SNR_FEATURES:
        cols += cfg.OPTIMAL_SNR_COLS
    if cfg.USE_OPTICAL_REDSHIFT:
        cols += cfg.OPTICAL_REDSHIFT_COLS
    if cfg.USE_OPTICAL_SIGMA_V:
        cols += cfg.OPTICAL_SIGMA_V_COLS
    if cfg.USE_OPTICAL_IMAGE_GEOMETRY:
        cols += cfg.OPTICAL_IMAGE_GEOMETRY_COLS
    if cfg.USE_OPTICAL_DISTANCE:
        cols += cfg.OPTICAL_DISTANCE_COLS
    if cfg.USE_ORACLE_TD:
        cols += cfg.ORACLE_TD_COLS
    if cfg.USE_ORACLE_BETA:
        cols += cfg.ORACLE_BETA_COLS
    if cfg.USE_ORACLE_R_ABS:
        cols += cfg.ORACLE_R_ABS_COLS
    if cfg.USE_ORACLE_AMP_RATIO:
        cols += cfg.ORACLE_AMP_RATIO_COLS
    if cfg.USE_ORACLE_Y:
        cols += cfg.ORACLE_Y_COLS
    return cols


def validate_no_oracle_leakage(obs_cols: List[str]):
    enabled_oracles = []
    if cfg.USE_ORACLE_TD:
        enabled_oracles.append("USE_ORACLE_TD")
    if cfg.USE_ORACLE_BETA:
        enabled_oracles.append("USE_ORACLE_BETA")
    if cfg.USE_ORACLE_R_ABS:
        enabled_oracles.append("USE_ORACLE_R_ABS")
    if cfg.USE_ORACLE_AMP_RATIO:
        enabled_oracles.append("USE_ORACLE_AMP_RATIO")
    if cfg.USE_ORACLE_Y:
        enabled_oracles.append("USE_ORACLE_Y")

    label_side_cols = {
        "mu_0", "mu_1", "abs_mu_0", "abs_mu_1", "mu_total_abs",
        "mu_ratio_abs_21", "R_abs_21", "amp_ratio_21", "amp_ratio_21_true",
        "u", "log_u", "y", "u_true", "log_u_true", "y_true",
        "beta", "beta_true", "inv_beta", "log_beta", "log_inv_beta",
        "beta_x", "beta_y", "image_x_0", "image_y_0", "image_x_1", "image_y_1",
        "t_d", "theta_E_true", "sigma_v_true",
        "z_l_true", "z_s_true",
    }
    leaked_cols = sorted(set(obs_cols) & label_side_cols)
    privileged_cols = []
    if not cfg.USE_OPTIMAL_SNR_FEATURES:
        privileged_cols = sorted(set(obs_cols) & set(cfg.OPTIMAL_SNR_COLS))

    if cfg.ALLOW_ORACLE_FEATURES:
        return

    if enabled_oracles or leaked_cols or privileged_cols:
        details = []
        if enabled_oracles:
            details.append(f"enabled oracle switches: {enabled_oracles}")
        if leaked_cols:
            details.append(f"label-side input columns: {leaked_cols}")
        if privileged_cols:
            details.append(f"privileged optimal-SNR columns: {privileged_cols}")
        raise RuntimeError(
            "Potential target leakage detected. "
            + "; ".join(details)
            + ". Set ALLOW_ORACLE_FEATURES=True only for explicit oracle ablation."
        )


def resolve_wave_paths():
    if cfg.USE_WHITENED_WAVE:
        prefix = "SIS_data_strain" if cfg.USE_NOISY_WAVE else "SIS_h_strain"
    else:
        prefix = "SIS_data_strain_raw" if cfg.USE_NOISY_WAVE else "SIS_h_strain_raw"

    return (
        resolve_data_path(os.path.join(cfg.DATA_ROOT, f"{prefix}_1.npy")),
        resolve_data_path(os.path.join(cfg.DATA_ROOT, f"{prefix}_2.npy")),
        prefix,
    )


def resolve_obs_path():
    if cfg.USE_NOISY_WAVE and cfg.USE_WHITENED_WAVE:
        path = resolve_data_path(cfg.OBS_CSV_NOISY_WHITE)
        expected_kind = "realobs_a21friendly_noisy_whitened"
    elif (not cfg.USE_NOISY_WAVE) and cfg.USE_WHITENED_WAVE:
        path = resolve_data_path(cfg.OBS_CSV_CLEAN_WHITE)
        expected_kind = "clean_whitened"
    elif cfg.USE_NOISY_WAVE and (not cfg.USE_WHITENED_WAVE):
        path = resolve_data_path(cfg.OBS_CSV_NOISY_RAW)
        expected_kind = "realobs_a21friendly_noisy_raw"
    else:
        path = resolve_data_path(cfg.OBS_CSV_CLEAN_RAW)
        expected_kind = "clean_raw"

    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing observable feature table for {expected_kind}: {path}. "
            "Please regenerate data with data_generation/SIS_GW_events_randomu_a21friendly_realobs.py."
        )

    return path, expected_kind


def add_optional_observables(obs_df: pd.DataFrame, lens_params_df: pd.DataFrame) -> pd.DataFrame:
    obs_df = obs_df.copy()

    def get_col(preferred, fallback=None):
        if preferred in lens_params_df.columns:
            return lens_params_df[preferred].to_numpy(dtype=np.float32)
        if fallback is not None and fallback in lens_params_df.columns:
            return lens_params_df[fallback].to_numpy(dtype=np.float32)
        raise RuntimeError(f"Missing column in lens_params.csv: {preferred}")

    z_s = get_col("z_s_observed", "z_s")
    z_l = get_col("z_l_observed", "z_l")
    sigma_v = get_col("sigma_v_observed", "sigma_v")

    fill_values = {
        "z_s_observed": z_s,
        "z_l_observed": z_l,
        "z_l_over_z_s": z_l / np.maximum(z_s, cfg.EPS),
        "log1p_z_s_observed": np.log1p(np.maximum(z_s, 0.0)),
        "log1p_z_l_observed": np.log1p(np.maximum(z_l, 0.0)),
        "sigma_v_observed": sigma_v,
        "log_sigma_v_observed": np.log(np.maximum(sigma_v, cfg.EPS)),
    }

    if "theta_E_observed" in lens_params_df.columns:
        fill_values["theta_E_observed"] = lens_params_df["theta_E_observed"].to_numpy(dtype=np.float32)
    elif "theta_E_observed(arcsec)" in lens_params_df.columns:
        fill_values["theta_E_observed"] = lens_params_df["theta_E_observed(arcsec)"].to_numpy(dtype=np.float32)
    elif "theta_E(arcsec)" in lens_params_df.columns:
        fill_values["theta_E_observed"] = lens_params_df["theta_E(arcsec)"].to_numpy(dtype=np.float32)

    if "image_separation_observed" in lens_params_df.columns:
        fill_values["image_separation_observed"] = lens_params_df["image_separation_observed"].to_numpy(dtype=np.float32)
    elif "image_separation_observed(arcsec)" in lens_params_df.columns:
        fill_values["image_separation_observed"] = lens_params_df["image_separation_observed(arcsec)"].to_numpy(dtype=np.float32)
    elif "image_separation(arcsec)" in lens_params_df.columns:
        fill_values["image_separation_observed"] = lens_params_df["image_separation(arcsec)"].to_numpy(dtype=np.float32)

    if "source_luminosity_distance" in lens_params_df.columns:
        fill_values["source_luminosity_distance"] = lens_params_df["source_luminosity_distance"].to_numpy(dtype=np.float32)

    for col, values in fill_values.items():
        if col not in obs_df.columns:
            obs_df[col] = values

    return obs_df


def get_device():
    if cfg.USE_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_or_trim(x: np.ndarray, target_len: int, stride: int):
    x = x.astype(np.float32)
    n = x.shape[-1]

    if n >= target_len:
        y = x[-target_len:]
    else:
        y = np.pad(x, (target_len - n, 0), mode="constant")

    if stride > 1:
        y = y[::stride]

    return y


def build_split_indices(n: int):
    train_idx_path = os.environ.get("MU0_TRAIN_INDICES_NPY")
    val_idx_path = os.environ.get("MU0_VAL_INDICES_NPY")
    if train_idx_path or val_idx_path:
        if not (train_idx_path and val_idx_path):
            raise RuntimeError(
                "Both MU0_TRAIN_INDICES_NPY and MU0_VAL_INDICES_NPY must be set "
                "when using an external split."
            )
        idx_tr = np.asarray(np.load(train_idx_path), dtype=np.int64).reshape(-1)
        idx_va = np.asarray(np.load(val_idx_path), dtype=np.int64).reshape(-1)
        if len(idx_tr) == 0 or len(idx_va) == 0:
            raise RuntimeError("External train/validation split is empty.")
        if idx_tr.min() < 0 or idx_va.min() < 0 or idx_tr.max() >= n or idx_va.max() >= n:
            raise RuntimeError(f"External split indices are outside valid range [0, {n}).")
        if len(np.unique(idx_tr)) != len(idx_tr) or len(np.unique(idx_va)) != len(idx_va):
            raise RuntimeError("External split indices contain duplicates.")
        if np.intersect1d(idx_tr, idx_va).size:
            raise RuntimeError("External train and validation indices overlap.")
        return idx_tr, idx_va

    idx = np.arange(n)
    rng = np.random.RandomState(cfg.SEED)
    rng.shuffle(idx)
    n_train = int(n * cfg.SPLIT["train"])
    return idx[:n_train], idx[n_train:]


def target_transform_np(mu0: np.ndarray):
    mu0 = np.asarray(mu0, dtype=np.float32)

    if cfg.TARGET_MODE == "log_mu0_minus1":
        return np.log(np.maximum(mu0 - 1.0, cfg.EPS)).astype(np.float32)

    if cfg.TARGET_MODE == "log_mu0":
        return np.log(np.maximum(mu0, cfg.EPS)).astype(np.float32)

    if cfg.TARGET_MODE == "mu0":
        return mu0.astype(np.float32)

    raise ValueError(f"Unsupported TARGET_MODE: {cfg.TARGET_MODE}")


def target_inverse_torch(pred: torch.Tensor):
    if cfg.TARGET_MODE == "log_mu0_minus1":
        return 1.0 + torch.exp(pred)

    if cfg.TARGET_MODE == "log_mu0":
        return torch.exp(pred)

    if cfg.TARGET_MODE == "mu0":
        return pred

    raise ValueError(f"Unsupported TARGET_MODE: {cfg.TARGET_MODE}")


def target_inverse_np(pred: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64)

    if cfg.TARGET_MODE == "log_mu0_minus1":
        return 1.0 + np.exp(pred)

    if cfg.TARGET_MODE == "log_mu0":
        return np.exp(pred)

    if cfg.TARGET_MODE == "mu0":
        return pred

    raise ValueError(f"Unsupported TARGET_MODE: {cfg.TARGET_MODE}")


# ============================================================
# Normalizer
# ============================================================
class FeatureNormalizer:
    def __init__(self, columns):
        self.columns = list(columns)
        self.mean_ = None
        self.std_ = None

    def fit(self, df: pd.DataFrame):
        arr = df[self.columns].to_numpy(dtype=np.float32)
        self.mean_ = arr.mean(axis=0)
        self.std_ = arr.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform_row(self, row: pd.Series):
        arr = row[self.columns].to_numpy(dtype=np.float32)
        return (arr - self.mean_) / self.std_

    def state_dict(self):
        return {
            "columns": np.array(self.columns, dtype=object),
            "mean": self.mean_,
            "std": self.std_,
        }


# ============================================================
# Dataset
# ============================================================
class MuDirectCompactDataset(Dataset):
    def __init__(self, wave1, wave2, lens_df, obs_df, obs_norm, indices, mode="train"):
        self.wave1 = wave1
        self.wave2 = wave2
        self.lens_df = lens_df
        self.obs_df = obs_df
        self.obs_norm = obs_norm
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def build_wave_pair(self, x1: np.ndarray, x2: np.ndarray):
        x1 = pad_or_trim(x1, cfg.TARGET_LEN, cfg.STRIDE)
        x2 = pad_or_trim(x2, cfg.TARGET_LEN, cfg.STRIDE)

        # Use one scale per image pair to preserve relative amplitudes.
        pair = np.concatenate([x1, x2], axis=0)
        pair_mean = pair.mean()
        pair_std = pair.std() + 1e-8

        def build_one(x):
            if cfg.WAVE_NORM_MODE == "pair":
                shape_ch = (x - pair_mean) / pair_std
            elif cfg.WAVE_NORM_MODE == "fixed":
                shape_ch = x / cfg.RAW_SCALE
            elif cfg.WAVE_NORM_MODE == "none":
                shape_ch = x
            else:
                raise ValueError(f"Unsupported WAVE_NORM_MODE: {cfg.WAVE_NORM_MODE}")

            # Fixed-scale channel retains absolute amplitudes in addition to relative amplitudes.
            amp_ch = np.tanh(x / cfg.RAW_SCALE)
            return np.stack([shape_ch, amp_ch], axis=0).astype(np.float32)

        return build_one(x1), build_one(x2)

    def __getitem__(self, item):
        idx = int(self.indices[item])

        x1 = np.array(self.wave1[idx], copy=True)
        x2 = np.array(self.wave2[idx], copy=True)

        if cfg.USE_AUGMENTATION and self.mode == "train" and np.random.rand() < cfg.AUG_PROB:
            shift = np.random.randint(-cfg.AUG_ROLL_MAX, cfg.AUG_ROLL_MAX + 1)
            x1 = np.roll(x1, shift)
            x2 = np.roll(x2, shift)

            noise1 = np.random.randn(*x1.shape) * (np.std(x1) / cfg.AUG_NOISE_DIV)
            noise2 = np.random.randn(*x2.shape) * (np.std(x2) / cfg.AUG_NOISE_DIV)

            x1 = x1 + noise1
            x2 = x2 + noise2

        w1, w2 = self.build_wave_pair(x1, x2)

        obs_vec = self.obs_norm.transform_row(self.obs_df.iloc[idx]).astype(np.float32)

        mu0 = float(self.lens_df.iloc[idx]["mu_0"])
        mu1 = float(self.lens_df.iloc[idx]["mu_1"])

        y_target = target_transform_np(np.array([mu0], dtype=np.float32))

        return (
            torch.from_numpy(w1),
            torch.from_numpy(w2),
            torch.from_numpy(obs_vec),
            torch.from_numpy(y_target),
            torch.tensor([mu0], dtype=torch.float32),
            torch.tensor([mu1], dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )


# ============================================================
# Model architecture
# ============================================================
def build_activation():
    if cfg.ACTIVATION.lower() == "relu":
        return nn.ReLU(inplace=True)
    return nn.SiLU()


class WaveEncoder(nn.Module):
    def __init__(self, in_ch=2, width=64):
        super().__init__()
        act = build_activation()

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 15, stride=2, padding=7),
            nn.BatchNorm1d(width),
            act,
            nn.MaxPool1d(2),

            nn.Conv1d(width, width * 2, 11, stride=2, padding=5),
            nn.BatchNorm1d(width * 2),
            act,

            nn.Conv1d(width * 2, width * 4, 9, stride=2, padding=4),
            nn.BatchNorm1d(width * 4),
            act,

            nn.Conv1d(width * 4, width * 4, 7, stride=2, padding=3),
            nn.BatchNorm1d(width * 4),
            act,

            nn.AdaptiveAvgPool1d(1),
        )

        self.out_dim = width * 4

    def forward(self, x):
        return self.net(x).flatten(1)


class MuDirectCompactRegressor(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        act = build_activation()

        self.wave_encoder = WaveEncoder(in_ch=2, width=cfg.WAVE_WIDTH)

        self.obs_head = nn.Sequential(
            nn.Linear(obs_dim, cfg.OBS_HIDDEN),
            act,
            nn.Linear(cfg.OBS_HIDDEN, cfg.D_MODEL // 2),
            act,
        )

        wave_dim = self.wave_encoder.out_dim
        fusion_dim = wave_dim * 4 + (cfg.D_MODEL // 2)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, cfg.D_MODEL),
            act,
            nn.Dropout(0.15),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL // 2),
            act,
            nn.Linear(cfg.D_MODEL // 2, 1),
        )

    def forward(self, w1, w2, obs):
        f1 = self.wave_encoder(w1)
        f2 = self.wave_encoder(w2)

        fd = torch.abs(f1 - f2)
        fp = f1 * f2
        fo = self.obs_head(obs)

        fused = torch.cat([f1, f2, fd, fp, fo], dim=1)

        # Output a transformed target, such as log(mu0-1).
        return self.head(fused)


# ============================================================
# loss
# ============================================================
def compute_losses(pred_target, target_transformed, mu0_true):
    """
    pred_target: Transformed target predicted by the model
    target_transformed: True transformed target
    mu0_true: True mu0
    """
    mu0_pred = target_inverse_torch(pred_target)

    # MAE in transformed space for stable training
    trans_mae = torch.abs(pred_target - target_transformed).mean()

    # MAE/MAPE in magnification space to constrain the final mu0
    abs_err = torch.abs(mu0_pred - mu0_true)
    rel_err = abs_err / (torch.abs(mu0_true) + cfg.EPS)

    if cfg.USE_TAIL_WEIGHT:
        # High-mu events are harder and contribute more to RMSE.
        # Use logarithmic weights to avoid excessive emphasis on the extreme tail.
        weight = 1.0 + cfg.TAIL_ALPHA * torch.log1p(torch.abs(mu0_true))
        abs_err = abs_err * weight
        rel_err = rel_err * weight

    mae = abs_err.mean()
    mape = rel_err.mean() * 100.0

    loss = cfg.LOSS_W_TRANS * trans_mae + cfg.LOSS_W_MAE * mae + cfg.LOSS_W_REL * mape

    return loss, mae, mape


# ============================================================
# Evaluation
# ============================================================
def compute_metrics(pred, true, name):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mape = np.mean(np.abs((pred - true) / (np.abs(true) + 1e-8))) * 100.0

    corr = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else np.nan

    return {
        "name": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "Corr": float(corr),
    }


def print_metric(m):
    print(
        f"[{m['name']}] "
        f"MAE={m['MAE']:.6f} | "
        f"RMSE={m['RMSE']:.6f} | "
        f"MAPE={m['MAPE']:.2f}% | "
        f"Corr={m['Corr']:.4f}"
    )


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_mu0_pred = []
    all_mu1_pred = []
    all_mu0_true = []
    all_mu1_true = []
    all_idx = []

    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0

    for w1, w2, obs, y_trans, mu0_true, mu1_true, idx in tqdm(loader, desc="Validation", leave=False):
        w1 = w1.to(device)
        w2 = w2.to(device)
        obs = obs.to(device)
        y_trans = y_trans.to(device)
        mu0_true = mu0_true.to(device)

        pred_trans = model(w1, w2, obs)
        loss, mae, mape = compute_losses(pred_trans, y_trans, mu0_true)

        mu0_pred = target_inverse_torch(pred_trans)

        # SIS relation: mu0 + mu1 = 2
        mu1_pred = 2.0 - mu0_pred

        total_loss += loss.item()
        total_mae += mae.item()
        total_mape += mape.item()

        all_mu0_pred.append(mu0_pred.cpu().numpy().reshape(-1))
        all_mu1_pred.append(mu1_pred.cpu().numpy().reshape(-1))
        all_mu0_true.append(mu0_true.cpu().numpy().reshape(-1))
        all_mu1_true.append(mu1_true.numpy().reshape(-1))
        all_idx.append(idx.numpy().reshape(-1))

    mu0_pred = np.concatenate(all_mu0_pred)
    mu1_pred = np.concatenate(all_mu1_pred)
    mu0_true = np.concatenate(all_mu0_true)
    mu1_true = np.concatenate(all_mu1_true)
    idx_all = np.concatenate(all_idx)

    n_batches = len(loader)

    return {
        "val_loss": total_loss / n_batches,
        "val_mae_loss": total_mae / n_batches,
        "val_mape_loss": total_mape / n_batches,

        "mu0_pred": mu0_pred,
        "mu1_pred": mu1_pred,
        "mu0_true": mu0_true,
        "mu1_true": mu1_true,
        "idx": idx_all,

        "mu0_metrics": compute_metrics(mu0_pred, mu0_true, "mu0 direct"),
        "mu1_metrics": compute_metrics(mu1_pred, mu1_true, "mu1 from SIS"),
    }


def update_ema_model(ema_model, model, decay):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(v * decay + msd[k].detach() * (1.0 - decay))


def train_one_epoch(model, ema_model, loader, opt, device, desc):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0

    for w1, w2, obs, y_trans, mu0_true, _, _ in tqdm(loader, desc=desc, leave=False):
        w1 = w1.to(device)
        w2 = w2.to(device)
        obs = obs.to(device)
        y_trans = y_trans.to(device)
        mu0_true = mu0_true.to(device)

        opt.zero_grad()

        pred_trans = model(w1, w2, obs)
        loss, mae, mape = compute_losses(pred_trans, y_trans, mu0_true)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        update_ema_model(ema_model, model, cfg.EMA_DECAY)

        total_loss += loss.item()
        total_mae += mae.item()
        total_mape += mape.item()

    n_batches = len(loader)

    return (
        total_loss / n_batches,
        total_mae / n_batches,
        total_mape / n_batches,
    )


# ============================================================
# Plotting
# ============================================================
def plot_results(eval_out, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[Warn] matplotlib unavailable; skipping scatter plot export: {exc}")
        return

    mu0_true = eval_out["mu0_true"]
    mu0_pred = eval_out["mu0_pred"]
    mu1_true = eval_out["mu1_true"]
    mu1_pred = eval_out["mu1_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pairs = [
        (mu0_true, mu0_pred, r"$\mu_0$"),
        (mu1_true, mu1_pred, r"$\mu_1$"),
    ]

    for ax, (true, pred, name) in zip(axes, pairs):
        ax.scatter(true, pred, s=8, alpha=0.55)

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Pred {name}")
        ax.set_title(f"Direct prediction: {name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_val_csv(eval_out, save_path, obs_df=None):
    df = pd.DataFrame({
        "event_id": eval_out["idx"],
        "mu0_true": eval_out["mu0_true"],
        "mu0_pred": eval_out["mu0_pred"],
        "mu1_true": eval_out["mu1_true"],
        "mu1_pred": eval_out["mu1_pred"],
    })

    if obs_df is not None:
        event_idx = df["event_id"].to_numpy(dtype=np.int64)
        for col in [
            "z_s_observed", "z_l_observed", "z_l_over_z_s",
            "sigma_v_observed", "theta_E_observed",
            "image_separation_observed", "R_abs_21",
            "u_true", "log_u_true", "y_true",
        ]:
            if col in obs_df.columns:
                df[col] = obs_df.iloc[event_idx][col].to_numpy()

    df["mu0_abs_err"] = np.abs(df["mu0_pred"] - df["mu0_true"])
    df["mu1_abs_err"] = np.abs(df["mu1_pred"] - df["mu1_true"])

    df["mu0_rel_err_percent"] = df["mu0_abs_err"] / (np.abs(df["mu0_true"]) + 1e-8) * 100.0
    df["mu1_rel_err_percent"] = df["mu1_abs_err"] / (np.abs(df["mu1_true"]) + 1e-8) * 100.0

    df.to_csv(save_path, index=False, encoding="utf-8-sig")


# ============================================================
# Main workflow
# ============================================================
def main():
    seed_everything(cfg.SEED)
    ensure_out_dir()

    device = get_device()

    print("=" * 90)
    print("Train Direct Mu Compact Regressor - A21-friendly realistic wave+observables")
    print(f"Device:      {device}")
    print(f"Target mode: {cfg.TARGET_MODE}")
    print(f"Out dir:     {cfg.OUT_DIR}")
    print(f"Data root:   {cfg.DATA_ROOT}")

    wave1_path, wave2_path, wave_prefix = resolve_wave_paths()
    obs_path, expected_feature_kind = resolve_obs_path()
    obs_cols = selected_obs_cols()
    validate_no_oracle_leakage(obs_cols)

    print(f"Wave source: {wave_prefix}")
    print(f"Obs table:   {os.path.basename(obs_path)}")
    print(f"Obs kind:    {expected_feature_kind}")
    print(f"Noisy wave:  {cfg.USE_NOISY_WAVE}")
    print(f"Whitened:    {cfg.USE_WHITENED_WAVE}")
    print(f"Wave norm:   {cfg.WAVE_NORM_MODE}")
    print(f"RAW_SCALE:   {cfg.RAW_SCALE}")
    print(f"Use optimal SNR features: {cfg.USE_OPTIMAL_SNR_FEATURES}")
    print(f"Optical redshift:       {cfg.USE_OPTICAL_REDSHIFT}")
    print(f"Optical sigma_v:        {cfg.USE_OPTICAL_SIGMA_V}")
    print(f"Optical image geometry: {cfg.USE_OPTICAL_IMAGE_GEOMETRY}")
    print(f"Optical distance:       {cfg.USE_OPTICAL_DISTANCE}")
    print(f"Oracle t_d:             {cfg.USE_ORACLE_TD}")
    print(f"Oracle beta_true:       {cfg.USE_ORACLE_BETA}")
    print(f"Oracle R_abs_21:        {cfg.USE_ORACLE_R_ABS}")
    print(f"Oracle amp_ratio_21:    {cfg.USE_ORACLE_AMP_RATIO}")
    print(f"Oracle y_true:          {cfg.USE_ORACLE_Y}")
    print(f"Loss weights:          trans={cfg.LOSS_W_TRANS:g} mae={cfg.LOSS_W_MAE:g} mape={cfg.LOSS_W_REL:g}")
    print(f"Tail weighting:        {cfg.USE_TAIL_WEIGHT} alpha={cfg.TAIL_ALPHA:g}")
    print(f"Obs dim:     {len(obs_cols)}")
    print("=" * 90)

    wave1 = np.load(wave1_path, mmap_mode="r")
    wave2 = np.load(wave2_path, mmap_mode="r")
    lens_df = pd.read_csv(resolve_data_path(cfg.LENS_CSV))
    lens_params_df = pd.read_csv(resolve_data_path(cfg.LENS_PARAMS_CSV))
    obs_df = pd.read_csv(obs_path)

    valid_len = min(len(wave1), len(wave2), len(lens_df), len(lens_params_df), len(obs_df))

    lens_df = lens_df.iloc[:valid_len].copy().reset_index(drop=True)
    lens_params_df = lens_params_df.iloc[:valid_len].copy().reset_index(drop=True)
    obs_df = obs_df.iloc[:valid_len].copy().reset_index(drop=True)
    obs_df = add_optional_observables(obs_df, lens_params_df)

    if "feature_data_kind" in obs_df.columns:
        kinds = set(obs_df["feature_data_kind"].dropna().astype(str).unique())
        if kinds and kinds != {expected_feature_kind}:
            raise RuntimeError(
                f"Observable feature table kind mismatch: expected "
                f"{expected_feature_kind}, got {sorted(kinds)}"
            )

    for c in ["mu_0", "mu_1"]:
        if c not in lens_df.columns:
            raise RuntimeError(f"Missing column in lens.csv: {c}")

    for c in obs_cols:
        if c not in obs_df.columns:
            raise RuntimeError(f"Missing column in {obs_path}: {c}")

    idx_tr, idx_va = build_split_indices(valid_len)

    obs_norm = FeatureNormalizer(obs_cols).fit(obs_df.iloc[idx_tr])

    tr_ds = MuDirectCompactDataset(
        wave1=wave1,
        wave2=wave2,
        lens_df=lens_df,
        obs_df=obs_df,
        obs_norm=obs_norm,
        indices=idx_tr,
        mode="train",
    )

    va_ds = MuDirectCompactDataset(
        wave1=wave1,
        wave2=wave2,
        lens_df=lens_df,
        obs_df=obs_df,
        obs_norm=obs_norm,
        indices=idx_va,
        mode="val",
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    va_loader = DataLoader(
        va_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = MuDirectCompactRegressor(obs_dim=len(obs_cols)).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    opt = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.7,
        patience=3,
        min_lr=cfg.MIN_LR,
    )

    best_mape = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    best_mu0_metrics = None
    best_mu1_metrics = None
    wait = 0
    history = []

    for ep in range(1, cfg.EPOCHS + 1):
        tr_loss, tr_mae, tr_mape = train_one_epoch(
            model=model,
            ema_model=ema_model,
            loader=tr_loader,
            opt=opt,
            device=device,
            desc=f"Epoch {ep}",
        )

        eval_out = evaluate_model(ema_model, va_loader, device)

        mu0_m = eval_out["mu0_metrics"]
        mu1_m = eval_out["mu1_metrics"]

        print(
            f"[Epoch {ep:03d}] "
            f"TrainLoss={tr_loss:.4f} TrainMAE={tr_mae:.6f} TrainMAPE={tr_mape:.2f}% | "
            f"ValLoss={eval_out['val_loss']:.4f} | "
            f"mu0_MAE={mu0_m['MAE']:.6f} mu0_MAPE={mu0_m['MAPE']:.2f}% mu0_Corr={mu0_m['Corr']:.4f} | "
            f"mu1_MAE={mu1_m['MAE']:.6f} mu1_MAPE={mu1_m['MAPE']:.2f}% mu1_Corr={mu1_m['Corr']:.4f} | "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        current_lr = float(opt.param_groups[0]["lr"])
        history.append(
            {
                "epoch": ep,
                "train_loss": float(tr_loss),
                "train_mae": float(tr_mae),
                "train_mape": float(tr_mape),
                "val_loss": float(eval_out["val_loss"]),
                "mu0_mae": float(mu0_m["MAE"]),
                "mu0_rmse": float(mu0_m["RMSE"]),
                "mu0_mape": float(mu0_m["MAPE"]),
                "mu0_corr": float(mu0_m["Corr"]),
                "mu1_mae": float(mu1_m["MAE"]),
                "mu1_rmse": float(mu1_m["RMSE"]),
                "mu1_mape": float(mu1_m["MAPE"]),
                "mu1_corr": float(mu1_m["Corr"]),
                "lr": current_lr,
            }
        )

        # Monitor mu0 MAPE first, with MAE as a tie-breaker.
        improved = False
        if mu0_m["MAPE"] < best_mape:
            best_mape = mu0_m["MAPE"]
            best_mae = mu0_m["MAE"]
            improved = True
        elif abs(mu0_m["MAPE"] - best_mape) < 1e-6 and mu0_m["MAE"] < best_mae:
            best_mae = mu0_m["MAE"]
            improved = True

        if improved:
            wait = 0
            best_epoch = ep
            best_mu0_metrics = dict(mu0_m)
            best_mu1_metrics = dict(mu1_m)

            ckpt_path = os.path.join(cfg.OUT_DIR, cfg.CKPT_NAME)
            torch.save(
                {
                    "model_state": copy.deepcopy(ema_model.state_dict()),
                    "obs_norm": obs_norm.state_dict(),
                    "target_mode": cfg.TARGET_MODE,
                    "obs_cols": obs_cols,
                    "wave_prefix": wave_prefix,
                    "config": cfg.__dict__,
                    "best_mu0_metrics": mu0_m,
                    "best_mu1_metrics": mu1_m,
                },
                ckpt_path,
            )

            plot_path = os.path.join(cfg.OUT_DIR, cfg.PLOT_NAME)
            plot_results(eval_out, plot_path)

            csv_path = os.path.join(cfg.OUT_DIR, cfg.CSV_NAME)
            save_val_csv(eval_out, csv_path, obs_df=obs_df)

            print(f"[Save] Best model saved to: {ckpt_path}")
            print_metric(mu0_m)
            print_metric(mu1_m)

        else:
            wait += 1
            if ep >= cfg.MIN_EPOCHS_BEFORE_STOP and wait >= cfg.EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

        scheduler.step(mu0_m["MAPE"])

    history_path = os.path.join(cfg.OUT_DIR, cfg.HISTORY_CSV_NAME)
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

    summary = {
        "best_epoch": best_epoch,
        "best_mu0_metrics": best_mu0_metrics,
        "best_mu1_metrics": best_mu1_metrics,
        "final_epoch": history[-1]["epoch"] if history else 0,
        "final_lr": history[-1]["lr"] if history else None,
        "config": cfg.__dict__,
        "obs_cols": obs_cols,
        "wave_prefix": wave_prefix,
    }
    summary_path = os.path.join(cfg.OUT_DIR, cfg.SUMMARY_JSON_NAME)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Save] Training history saved to: {history_path}")
    print(f"[Save] Training summary saved to: {summary_path}")
    print("Training finished.")


if __name__ == "__main__":
    main()

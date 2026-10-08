#!/usr/bin/env python
# coding: utf-8
"""Shared helpers for the third-round reviewer experiments.

All third-round scripts write machine-readable CSV/JSON outputs and print a
real tqdm progress bar.  The module deliberately keeps the experiments
separate from the original benchmark code so that a partial server run can be
resumed without changing the SIS baseline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional on minimal environments
    tqdm = None


GEOMETRY_COLS = [
    "obs__z_s_observed",
    "obs__z_l_observed",
    "obs__z_l_over_z_s",
    "obs__log1p_z_s_observed",
    "obs__log1p_z_l_observed",
    "obs__sigma_v_observed",
    "obs__log_sigma_v_observed",
    "obs__theta_E_observed",
    "obs__image_separation_observed",
    "obs__theta_plus_abs_observed",
    "obs__theta_minus_abs_observed",
    "obs__image_position_asymmetry_observed",
    "obs__lens_center_x_observed",
    "obs__lens_center_y_observed",
    "obs__image_x_0_observed",
    "obs__image_y_0_observed",
    "obs__image_x_1_observed",
    "obs__image_y_1_observed",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: object, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=str)


def progress_write(message: str) -> None:
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)


def progress(iterable: Iterable, *, total: int | None = None, desc: str, unit: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def metric_dict(true: Iterable[float], pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(true, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "mape_pct": np.nan}
    err = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-8)) * 100.0),
    }


def y_to_mu0(y: Iterable[float]) -> np.ndarray:
    return 1.0 + 1.0 / np.clip(np.abs(np.asarray(y, dtype=float)), 1e-4, 1.0 - 1e-4)


def first_existing(row: pd.Series, names: Sequence[str]) -> str:
    for name in names:
        if name in row.index:
            return name
    raise KeyError(f"None of {names} exists in row columns")


def safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    den = float(den)
    if not np.isfinite(den) or abs(den) < eps:
        den = eps if den >= 0 else -eps
    return float(num / den)


def load_sis_table(data_root: str | Path) -> pd.DataFrame:
    """Load the original SIS CSV tables into the flat feature layout."""

    root = Path(data_root)
    lens = pd.read_csv(root / "lens.csv").reset_index(drop=True)
    params = pd.read_csv(root / "lens_params.csv").reset_index(drop=True)
    obs = pd.read_csv(root / "observable_features.csv").reset_index(drop=True)
    n = min(len(lens), len(params), len(obs))
    if n <= 0:
        raise RuntimeError(f"No rows found under {root}")
    lens, params, obs = lens.iloc[:n], params.iloc[:n], obs.iloc[:n]

    out = pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "mu0_true": pd.to_numeric(lens["mu_0"], errors="coerce"),
        "mu1_true": pd.to_numeric(lens.get("mu_1", np.nan), errors="coerce"),
        "y_true": pd.to_numeric(lens.get("y", np.nan), errors="coerce"),
        "t_d_true": pd.to_numeric(lens.get("t_d", np.nan), errors="coerce"),
    })
    out["source_luminosity_distance"] = pd.to_numeric(
        params.get("source_luminosity_distance", np.nan), errors="coerce"
    )
    if "source_luminosity_distance" in obs.columns:
        out["source_luminosity_distance"] = pd.to_numeric(
            obs["source_luminosity_distance"], errors="coerce"
        ).to_numpy()

    def copy_param(source: str, target: str) -> None:
        if source in params:
            out[target] = pd.to_numeric(params[source], errors="coerce")
        elif source in obs:
            out[target] = pd.to_numeric(obs[source], errors="coerce")
        else:
            out[target] = np.nan

    aliases = {
        "z_s_observed": ["z_s_observed", "z_s"],
        "z_l_observed": ["z_l_observed", "z_l"],
        "z_l_over_z_s": ["z_l_over_z_s"],
        "log1p_z_s_observed": ["log1p_z_s_observed"],
        "log1p_z_l_observed": ["log1p_z_l_observed"],
        "sigma_v_observed": ["sigma_v_observed", "sigma_v"],
        "source_luminosity_distance": ["source_luminosity_distance"],
        "log_sigma_v_observed": ["log_sigma_v_observed"],
        "theta_E_observed": ["theta_E_observed", "theta_E_observed(arcsec)"],
        "image_separation_observed": ["image_separation_observed", "image_separation_observed(arcsec)"],
        "theta_plus_abs_observed": ["theta_plus_abs_observed"],
        "theta_minus_abs_observed": ["theta_minus_abs_observed"],
        "image_position_asymmetry_observed": ["image_position_asymmetry_observed"],
        "lens_center_x_observed": ["lens_center_x_observed"],
        "lens_center_y_observed": ["lens_center_y_observed"],
        "image_x_0_observed": ["image_x_0_observed"],
        "image_y_0_observed": ["image_y_0_observed"],
        "image_x_1_observed": ["image_x_1_observed"],
        "image_y_1_observed": ["image_y_1_observed"],
    }
    for target, candidates in aliases.items():
        chosen = next((c for c in candidates if c in obs.columns or c in params.columns), None)
        if chosen is None:
            out[f"obs__{target}"] = np.nan
        elif chosen in obs.columns:
            out[f"obs__{target}"] = pd.to_numeric(obs[chosen], errors="coerce").to_numpy()
        else:
            out[f"obs__{target}"] = pd.to_numeric(params[chosen], errors="coerce").to_numpy()

    # The original observable table has all derived columns, but these fallbacks
    # make the third-round scripts usable on older SIS exports too.
    if out["obs__z_l_over_z_s"].isna().all():
        out["obs__z_l_over_z_s"] = out["obs__z_l_observed"] / np.maximum(out["obs__z_s_observed"], 1e-8)
    if out["obs__log1p_z_s_observed"].isna().all():
        out["obs__log1p_z_s_observed"] = np.log1p(np.maximum(out["obs__z_s_observed"], 0.0))
    if out["obs__log1p_z_l_observed"].isna().all():
        out["obs__log1p_z_l_observed"] = np.log1p(np.maximum(out["obs__z_l_observed"], 0.0))
    if out["obs__log_sigma_v_observed"].isna().all():
        out["obs__log_sigma_v_observed"] = np.log(np.maximum(out["obs__sigma_v_observed"], 1e-8))
    return out


def feature_columns(df: pd.DataFrame, requested: Sequence[str] = GEOMETRY_COLS) -> list[str]:
    return [c for c in requested if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]


def fit_histgb(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], seed: int = 42,
               max_iter: int = 80, verbose: int = 0) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    cols = feature_columns(train, features)
    if not cols:
        raise RuntimeError("No usable geometry columns were found.")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=0.055,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=seed,
            verbose=verbose,
        ),
    )
    tau = np.log(np.maximum(pd.to_numeric(train["mu0_true"], errors="coerce").to_numpy() - 1.0, 1e-8))
    model.fit(train[cols].apply(pd.to_numeric, errors="coerce"), tau)
    pred = 1.0 + np.exp(model.predict(test[cols].apply(pd.to_numeric, errors="coerce")))
    return np.asarray(pred, dtype=float)

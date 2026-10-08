#!/usr/bin/env python
# coding: utf-8
"""Shared utilities for SIS gravitational-wave lensing experiments.

This module orchestrates experiments, records results and plots outputs without changing the generator or model logic.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data_generation" / "data_lens_sis_gw_physics_baseline"
DEFAULT_EXP_ROOT = ROOT / "experiments" / "outputs"

_PHYSICS_DATA_REQUIRED_FILES = (
    "lens.csv",
    "lens_params.csv",
    "observable_features.csv",
    "SIS_data_strain_1.npy",
    "SIS_data_strain_2.npy",
    "quality_report.json",
)


def ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_latest_expanded_dataset() -> Path:
    """Return the newest complete, successfully generated expansion dataset.

    Expansion runs are timestamped, so a fixed data_lens_*_expanded_5000 path
    is neither created by the generator nor reproducible across runs.  The
    caller persists the resolved path in its run configuration.
    """
    root = DEFAULT_EXP_ROOT / "expanded_data_generation"
    candidates: list[Path] = []
    if root.exists():
        for path in root.glob("*/data_lens_sis_gw_physics_baseline_*"):
            if not path.is_dir() or not all((path / name).is_file() for name in _PHYSICS_DATA_REQUIRED_FILES):
                continue
            try:
                with (path / "quality_report.json").open("r", encoding="utf-8") as f:
                    report = json.load(f)
                if int(report.get("accepted", 0)) <= 0:
                    continue
            except (OSError, ValueError, TypeError):
                continue
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Expanded SIS dataset not found. First run `experiments/expanded_data_generation.py`, "
            "or specify the data directory with --data-root."
        )
    return max(candidates, key=lambda path: (path / "quality_report.json").stat().st_mtime)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


@contextlib.contextmanager
def tee_output(log_path: str | Path):
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        tee_out = Tee(sys.stdout, f)
        tee_err = Tee(sys.stderr, f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            yield


def set_deterministic(seed: int, deterministic: bool = True) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def metric_dict(true: Iterable[float], pred: Iterable[float], prefix: str = "mu0") -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ok = np.isfinite(true) & np.isfinite(pred)
    true = true[ok]
    pred = pred[ok]
    if len(true) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mape": np.nan,
            f"{prefix}_bias": np.nan,
            f"{prefix}_pearson": np.nan,
            f"{prefix}_spearman": np.nan,
        }
    err = pred - true
    pearson = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else np.nan
    spearman = float(pd.Series(true).rank().corr(pd.Series(pred).rank())) if len(true) > 1 else np.nan
    return {
        f"{prefix}_n": int(len(true)),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_mape": float(np.mean(np.abs(err) / (np.abs(true) + 1e-8)) * 100.0),
        f"{prefix}_bias": float(np.mean(err)),
        f"{prefix}_pearson": pearson,
        f"{prefix}_spearman": spearman,
    }


def summarize_prediction_csv(csv_path: str | Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    out: dict[str, float] = {"prediction_csv": str(csv_path), "n": int(len(df))}
    if {"mu0_true", "mu0_pred"}.issubset(df.columns):
        out.update(metric_dict(df["mu0_true"], df["mu0_pred"], "mu0"))
    if {"mu1_true", "mu1_pred"}.issubset(df.columns):
        out.update(metric_dict(df["mu1_true"], df["mu1_pred"], "mu1"))
    return out


def sci_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.45,
            "axes.prop_cycle": plt.cycler(color=["#8DBCE8", "#F1AAA6", "#F4D67D", "#A9D0B0"]),
        }
    )


def plot_prediction_scatter_sci(
    csv_path: str | Path,
    out_path: str | Path,
    title: str = "",
    true_col: str = "mu0_true",
    pred_col: str = "mu0_pred",
) -> None:
    sci_style()
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    true = df[true_col].to_numpy(dtype=np.float64)
    pred = df[pred_col].to_numpy(dtype=np.float64)
    ok = np.isfinite(true) & np.isfinite(pred)
    true = true[ok]
    pred = pred[ok]
    m = metric_dict(true, pred, "x")

    lo = float(np.nanmin([true.min(), pred.min()]))
    hi = float(np.nanmax([true.max(), pred.max()]))
    pad = 0.04 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.scatter(true, pred, s=10, alpha=0.58, edgecolors="none", color="#8DBCE8")
    ax.plot([lo, hi], [lo, hi], color="#E88983", linestyle="--", linewidth=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"True $\mu_0$")
    ax.set_ylabel(r"Predicted $\mu_0$")
    ax.set_title(title or "Prediction")
    text = (
        f"MAE={m['x_mae']:.3f}\n"
        f"RMSE={m['x_rmse']:.3f}\n"
        f"MAPE={m['x_mape']:.2f}%\n"
        f"r={m['x_pearson']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="#999999", linewidth=0.6, alpha=0.86),
    )
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_residual_sci(
    csv_path: str | Path,
    out_path: str | Path,
    title: str = "",
    true_col: str = "mu0_true",
    pred_col: str = "mu0_pred",
) -> None:
    sci_style()
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    true = df[true_col].to_numpy(dtype=np.float64)
    pred = df[pred_col].to_numpy(dtype=np.float64)
    resid = pred - true
    ok = np.isfinite(true) & np.isfinite(resid)
    true = true[ok]
    resid = resid[ok]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.scatter(true, resid, s=9, alpha=0.55, edgecolors="none", color="#F1AAA6")
    ax.set_xlabel(r"True $\mu_0$")
    ax.set_ylabel(r"Residual $\hat{\mu}_0-\mu_0$")
    ax.set_title(title or "Residuals")
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def bin_metrics(
    df: pd.DataFrame,
    bin_col: str = "mu0_true",
    true_col: str = "mu0_true",
    pred_col: str = "mu0_pred",
    bins: list[float] | None = None,
    quantiles: int | None = None,
) -> pd.DataFrame:
    work = df.copy()
    if bins is not None:
        work["_bin"] = pd.cut(work[bin_col], bins=bins, include_lowest=True)
    elif quantiles is not None:
        work["_bin"] = pd.qcut(work[bin_col], q=quantiles, duplicates="drop")
    else:
        work["_bin"] = pd.qcut(work[bin_col], q=6, duplicates="drop")

    rows = []
    for b, sub in work.groupby("_bin", observed=True):
        m = metric_dict(sub[true_col], sub[pred_col], "mu0")
        rows.append(
            {
                "bin": str(b),
                "bin_left": float(b.left) if hasattr(b, "left") else np.nan,
                "bin_right": float(b.right) if hasattr(b, "right") else np.nan,
                **m,
            }
        )
    return pd.DataFrame(rows)


def plot_bin_metric_sci(
    bin_df: pd.DataFrame,
    out_path: str | Path,
    metric: str = "mu0_mae",
    title: str = "",
    ylabel: str | None = None,
) -> None:
    sci_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    x = np.arange(len(bin_df))
    ax.bar(x, bin_df[metric].to_numpy(dtype=np.float64), color="#F4D67D", width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_df["bin"].astype(str), rotation=30, ha="right")
    metric_labels = {
        "mu0_mae": r"MAE of $\mu_0$",
        "mu0_rmse": r"RMSE of $\mu_0$",
        "mu0_mape": r"MAPE of $\mu_0$ (%)",
    }
    ax.set_ylabel(ylabel or metric_labels.get(metric, metric))
    ax.set_title(title or metric)
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_bars(
    summary_csv: str | Path,
    out_path: str | Path,
    label_col: str = "experiment",
    metric_cols: tuple[str, ...] = ("mu0_mae", "mu0_rmse", "mu0_mape"),
    title: str = "",
) -> None:
    sci_style()
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    labels = df[label_col].astype(str).to_list() if label_col in df.columns else [str(i) for i in range(len(df))]
    n = len(labels)
    x = np.arange(n)
    width = min(0.22, 0.8 / max(1, len(metric_cols)))
    colors = ["#8DBCE8", "#F1AAA6", "#F4D67D", "#A9D0B0"]

    fig, ax = plt.subplots(figsize=(max(5.5, 0.7 * n), 3.8))
    for i, col in enumerate(metric_cols):
        if col not in df.columns:
            continue
        metric_labels = {
            "mu0_mae": r"MAE of $\mu_0$",
            "mu0_rmse": r"RMSE of $\mu_0$",
            "mu0_mape": r"MAPE of $\mu_0$ (%)",
        }
        ax.bar(
            x + (i - (len(metric_cols) - 1) / 2) * width,
            df[col],
            width=width,
            label=metric_labels.get(col, col),
            color=colors[i % len(colors)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title or "Experiment metrics")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_dataset_distributions(data_root: str | Path, out_dir: str | Path, prefix: str = "dataset") -> pd.DataFrame:
    sci_style()
    import matplotlib.pyplot as plt

    data_root = Path(data_root)
    out_dir = ensure_dir(out_dir)
    lens = pd.read_csv(data_root / "lens.csv")
    obs = pd.read_csv(data_root / "observable_features.csv")
    params = pd.read_csv(data_root / "lens_params.csv")
    df = lens.copy()
    for col in ["snr_pair", "snr_1", "snr_2", "window_snr_like_1", "window_snr_like_2"]:
        if col in obs.columns:
            df[col] = obs[col].to_numpy()[: len(df)]
    for col in ["z_l_observed", "z_s_observed", "sigma_v_observed", "theta_E_observed"]:
        if col in params.columns and col not in df.columns:
            df[col] = params[col].to_numpy()[: len(df)]

    cols = [c for c in ["mu_0", "y", "t_d", "snr_pair", "sigma_v_observed", "z_s_observed"] if c in df.columns]
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append(
            {
                "column": c,
                "n": int(s.size),
                "min": float(s.min()),
                "p05": float(s.quantile(0.05)),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{prefix}_distribution_statistics.csv", index=False, encoding="utf-8-sig")

    ncols = 3 if len(cols) >= 6 else 2
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.65 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    colors = ["#8DBCE8", "#F1AAA6", "#F4D67D", "#A9D0B0", "#8DBCE8", "#F1AAA6"]
    for ax, c, color in zip(axes, cols, colors):
        values = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if c == "t_d":
            values = np.log10(np.maximum(values, 1e-12))
            xlabel = r"$\log_{10}(\Delta t / \mathrm{s})$"
        else:
            xlabel = {
                "mu_0": r"$\mu_0$",
                "y": r"Source offset $y$",
                "snr_pair": "Network SNR",
                "sigma_v_observed": r"Observed $\sigma_v$ (km s$^{-1}$)",
                "z_s_observed": "Observed source redshift",
            }.get(c, c)
        ax.hist(values, bins=36, color=color, alpha=0.88, edgecolor="white", linewidth=0.35)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_distribution_plot.png")
    fig.savefig(out_dir / f"{prefix}_distribution_plot.pdf")
    plt.close(fig)
    return summary


def run_subprocess(command: list[str], log_path: str | Path, cwd: str | Path | None = None) -> dict[str, Any]:
    start = time.time()
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(command) + "\n\n")
        f.flush()
        # Keep a durable log while forwarding tqdm/progress output to the
        # terminal.  Long-running waveform generation must not look idle.
        proc = subprocess.Popen(
            command,
            cwd=str(cwd or ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = proc.wait()
    return {
        "command": command,
        "returncode": int(returncode),
        "duration_sec": float(time.time() - start),
        "log_path": str(log_path),
    }


def build_kfold_indices(n: int, k: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((train, val))
    return out


@dataclass
class TrainRunConfig:
    data_root: str
    out_dir: str
    preset: str = "all"
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1.0e-4
    target_len: int = 4096
    raw_scale: float = 0.01
    seed: int = 42
    use_optimal_snr: bool = False
    allow_oracle: bool = False
    use_optical_redshift: bool = True
    use_optical_sigma_v: bool = True
    use_optical_image_geometry: bool = False
    use_optical_distance: bool = False
    use_oracle_td: bool = False
    use_oracle_beta: bool = False
    use_oracle_r_abs: bool = False
    use_oracle_amp_ratio: bool = False
    use_oracle_y: bool = False
    use_tail_weight: bool = True
    target_mode: str | None = None
    deterministic: bool = False


def run_physics_training(
    config: TrainRunConfig,
    overrides: dict[str, Any] | None = None,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    ensure_repo_on_path()
    out_dir = ensure_dir(config.out_dir)
    log_path = out_dir / "training_log.txt"
    set_deterministic(config.seed, deterministic=config.deterministic)

    start = time.time()
    with tee_output(log_path):
        base = importlib.import_module("train_mu0_from_wave_obs_randomu_a21friendly_realobs")
        entry = importlib.import_module("train_mu0_physics_baseline")
        base = importlib.reload(base)
        entry = importlib.reload(entry)

        entry.USE_OPTICAL_REDSHIFT = bool(config.use_optical_redshift)
        entry.USE_OPTICAL_SIGMA_V = bool(config.use_optical_sigma_v)
        entry.USE_OPTICAL_IMAGE_GEOMETRY = bool(config.use_optical_image_geometry)
        entry.USE_OPTICAL_DISTANCE = bool(config.use_optical_distance)
        entry.USE_ORACLE_TD = bool(config.use_oracle_td)
        entry.USE_ORACLE_BETA = bool(config.use_oracle_beta)
        entry.USE_ORACLE_R_ABS = bool(config.use_oracle_r_abs)
        entry.USE_ORACLE_AMP_RATIO = bool(config.use_oracle_amp_ratio)
        entry.USE_ORACLE_Y = bool(config.use_oracle_y)

        args = SimpleNamespace(
            data_root=str(Path(config.data_root).resolve()),
            out_dir=str(out_dir.parent),
            preset=config.preset,
            epochs=int(config.epochs),
            batch_size=int(config.batch_size),
            lr=float(config.lr),
            target_len=int(config.target_len),
            raw_scale=float(config.raw_scale),
            use_optimal_snr=bool(config.use_optimal_snr),
            allow_oracle=bool(
                config.allow_oracle
                or config.use_oracle_td
                or config.use_oracle_beta
                or config.use_oracle_r_abs
                or config.use_oracle_amp_ratio
                or config.use_oracle_y
            ),
        )
        entry.configure_base(args)
        base.cfg.OUT_DIR = str(out_dir)
        base.cfg.SEED = int(config.seed)
        base.cfg.CKPT_NAME = "best_model.pt"
        base.cfg.PLOT_NAME = "SCI_training_scatter.png"
        base.cfg.CSV_NAME = "val_predictions.csv"
        base.cfg.USE_TAIL_WEIGHT = bool(config.use_tail_weight)
        if config.target_mode is not None:
            base.cfg.TARGET_MODE = str(config.target_mode)

        if overrides:
            for key, value in overrides.items():
                if not hasattr(base.cfg, key):
                    raise AttributeError(f"base.cfg has no attribute {key}")
                setattr(base.cfg, key, value)

        if split_indices is not None:
            train_idx = np.asarray(split_indices[0], dtype=np.int64)
            val_idx = np.asarray(split_indices[1], dtype=np.int64)

            def fixed_split(n: int):
                train_max = int(train_idx.max()) if len(train_idx) else -1
                val_max = int(val_idx.max()) if len(val_idx) else -1
                if train_max >= n or val_max >= n:
                    raise RuntimeError("Provided split indices exceed dataset length.")
                return train_idx, val_idx

            base.build_split_indices = fixed_split

        manifest = {
            "run_label": run_label or Path(config.out_dir).name,
            "config": asdict(config),
            "overrides": overrides or {},
            "split": {
                "custom": split_indices is not None,
                "train_n": int(len(split_indices[0])) if split_indices is not None else None,
                "val_n": int(len(split_indices[1])) if split_indices is not None else None,
            },
        }
        save_json(manifest, out_dir / "run_config.json")
        base.main()

    pred_csv = out_dir / "val_predictions.csv"
    ckpt_path = out_dir / "best_model.pt"
    summary: dict[str, Any] = {
        "run_label": run_label or out_dir.name,
        "out_dir": str(out_dir),
        "data_root": str(Path(config.data_root).resolve()),
        "preset": config.preset,
        "seed": int(config.seed),
        "epochs": int(config.epochs),
        "duration_sec": float(time.time() - start),
        "prediction_csv": str(pred_csv) if pred_csv.exists() else "",
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else "",
        "log_path": str(log_path),
    }
    if pred_csv.exists():
        summary.update(summarize_prediction_csv(pred_csv))
        plot_prediction_scatter_sci(pred_csv, out_dir / "SCI_prediction_scatter.png", title="Validation prediction")
        plot_residual_sci(pred_csv, out_dir / "SCI_residuals.png", title="Validation residuals")
        df = pd.read_csv(pred_csv)
        if {"mu0_true", "mu0_pred"}.issubset(df.columns) and len(df) >= 8:
            bins = bin_metrics(df, quantiles=min(6, max(2, len(df) // 10)))
            bins.to_csv(out_dir / "mu0binned_metrics.csv", index=False, encoding="utf-8-sig")
            plot_bin_metric_sci(bins, out_dir / "SCI_mu0_binned_MAE.png", "mu0_mae", title=summary["run_label"])
    save_json(summary, out_dir / "training_summary.json")
    gc.collect()
    return summary


def dataset_length(data_root: str | Path) -> int:
    lens_path = Path(data_root) / "lens.csv"
    if not lens_path.exists():
        raise FileNotFoundError(lens_path)
    return int(len(pd.read_csv(lens_path)))


def load_joined_dataset(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    lens = pd.read_csv(data_root / "lens.csv").reset_index(drop=True)
    obs = pd.read_csv(data_root / "observable_features.csv").reset_index(drop=True)
    params = pd.read_csv(data_root / "lens_params.csv").reset_index(drop=True)
    n = min(len(lens), len(obs), len(params))
    lens = lens.iloc[:n].copy()
    obs = obs.iloc[:n].copy()
    params = params.iloc[:n].copy()
    joined = pd.concat(
        [
            lens.add_prefix("lens__"),
            obs.drop(columns="event_id", errors="ignore").add_prefix("obs__"),
            params.drop(columns="event_id", errors="ignore").add_prefix("params__"),
        ],
        axis=1,
        copy=False,
    )
    return joined.assign(row_index=np.arange(n, dtype=np.int64))


def slice_npy_first_axis(src: Path, dst: Path, indices: np.ndarray, chunk: int = 512) -> None:
    arr = np.load(src, mmap_mode="r")
    max_idx = int(indices.max()) if len(indices) else -1
    if arr.ndim == 0 or max_idx >= arr.shape[0]:
        return
    ensure_dir(dst.parent)
    pieces = []
    for start in range(0, len(indices), chunk):
        pieces.append(np.asarray(arr[indices[start : start + chunk]]))
    np.save(dst, np.concatenate(pieces, axis=0) if pieces else np.asarray(arr[:0]))


def slice_dataset(
    data_root: str | Path,
    out_root: str | Path,
    indices: Iterable[int],
    reset_event_id: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Path:
    data_root = Path(data_root)
    out_root = ensure_dir(out_root)
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)
    n_total = dataset_length(data_root)
    if len(indices) == 0:
        raise ValueError("Cannot create an empty subset dataset.")
    if int(indices.max()) >= n_total:
        raise ValueError("Subset index exceeds dataset length.")

    for name in ["lens.csv", "lens_params.csv", "observable_features.csv", "source_truth.csv", "source_samples.csv", "diagnostic_features.csv", "lens_observable_features.csv"]:
        src = data_root / name
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if len(df) <= int(indices.max()):
            continue
        sub = df.iloc[indices].copy().reset_index(drop=True)
        if reset_event_id and "event_id" in sub.columns:
            sub["event_id"] = np.arange(len(sub), dtype=np.int64)
        sub.to_csv(out_root / name, index=False, encoding="utf-8-sig")

    pd.DataFrame({"lensed_index": np.arange(len(indices), dtype=np.int64)}).to_csv(
        out_root / "lensed_index.csv", index=False
    )
    pd.DataFrame({"original_index": indices}).to_csv(out_root / "subset_original_indices.csv", index=False)

    for src in data_root.glob("*.npy"):
        slice_npy_first_axis(src, out_root / src.name, indices)

    meta = {
        "source_data_root": str(data_root),
        "subset_size": int(len(indices)),
        "reset_event_id": bool(reset_event_id),
        "created_at": now_stamp(),
        "metadata": metadata or {},
    }
    save_json(meta, out_root / "subset_metadata.json")
    try:
        plot_dataset_distributions(out_root, out_root / "figures", prefix="special_subset")
    except Exception as exc:
        save_json({"plot_error": repr(exc)}, out_root / "plot_error.json")
    return out_root


def load_trusted_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu"):
    """Load local experiment checkpoints across PyTorch 2.6+ defaults.

    The checkpoints used here are produced by this project and store config,
    normalizer state and model weights. PyTorch 2.6 defaults torch.load to
    weights_only=True, which rejects those full dictionaries.
    """
    import torch

    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def checkpoint_weight_digest(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    h = hashlib.sha256()
    total_params = 0
    tensor_count = 0
    for key in sorted(state.keys()):
        value = state[key]
        if not hasattr(value, "detach"):
            continue
        arr = value.detach().cpu().contiguous().numpy()
        h.update(key.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
        total_params += int(arr.size)
        tensor_count += 1
    return {
        "checkpoint": str(checkpoint_path),
        "sha256": h.hexdigest(),
        "tensor_count": int(tensor_count),
        "total_params": int(total_params),
    }


def compare_checkpoint_weights(path_a: str | Path, path_b: str | Path) -> dict[str, Any]:
    ckpt_a = load_trusted_checkpoint(path_a, map_location="cpu")
    ckpt_b = load_trusted_checkpoint(path_b, map_location="cpu")
    state_a = ckpt_a.get("model_state", ckpt_a)
    state_b = ckpt_b.get("model_state", ckpt_b)
    keys = sorted(set(state_a.keys()) & set(state_b.keys()))
    max_abs = 0.0
    sum_abs = 0.0
    n = 0
    compared = 0
    for key in keys:
        a = state_a[key]
        b = state_b[key]
        if not (hasattr(a, "detach") and hasattr(b, "detach")):
            continue
        da = a.detach().cpu().float()
        db = b.detach().cpu().float()
        if da.shape != db.shape:
            continue
        diff = (da - db).abs()
        max_abs = max(max_abs, float(diff.max().item()) if diff.numel() else 0.0)
        sum_abs += float(diff.sum().item())
        n += int(diff.numel())
        compared += 1
    return {
        "checkpoint_a": str(path_a),
        "checkpoint_b": str(path_b),
        "compared_tensors": int(compared),
        "compared_params": int(n),
        "max_abs_diff": float(max_abs),
        "mean_abs_diff": float(sum_abs / max(n, 1)),
        "exact_sha_match": checkpoint_weight_digest(path_a)["sha256"] == checkpoint_weight_digest(path_b)["sha256"],
    }


def evaluate_checkpoint_on_data(
    checkpoint_path: str | Path,
    data_root: str | Path,
    out_dir: str | Path,
    batch_size: int = 64,
    run_label: str = "external_eval",
) -> dict[str, Any]:
    ensure_repo_on_path()
    import torch
    from torch.utils.data import DataLoader

    out_dir = ensure_dir(out_dir)
    base = importlib.import_module("train_mu0_from_wave_obs_randomu_a21friendly_realobs")
    entry = importlib.import_module("train_mu0_physics_baseline")
    base = importlib.reload(base)
    entry = importlib.reload(entry)

    ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    cfg_dict = dict(ckpt.get("config", {}))
    preset = cfg_dict.get("QUALITY_PRESET", cfg_dict.get("PHYSICS_PRESET", "all"))
    args = SimpleNamespace(
        data_root=str(Path(data_root).resolve()),
        out_dir=str(out_dir.parent),
        preset=preset,
        epochs=1,
        batch_size=int(batch_size),
        lr=1e-4,
        target_len=int(cfg_dict.get("TARGET_LEN", 4096)),
        raw_scale=float(cfg_dict.get("RAW_SCALE", 0.01)),
        use_optimal_snr=bool(cfg_dict.get("USE_OPTIMAL_SNR_FEATURES", False)),
        allow_oracle=bool(cfg_dict.get("ALLOW_ORACLE_FEATURES", False)),
    )
    entry.configure_base(args)
    for key, value in cfg_dict.items():
        if hasattr(base.cfg, key):
            setattr(base.cfg, key, value)
    base.cfg.DATA_ROOT = str(Path(data_root).resolve())
    base.cfg.LENS_CSV = str(Path(data_root) / "lens.csv")
    base.cfg.LENS_PARAMS_CSV = str(Path(data_root) / "lens_params.csv")
    base.cfg.OBS_CSV_NOISY_WHITE = str(Path(data_root) / "observable_features.csv")
    base.cfg.OUT_DIR = str(out_dir)
    base.cfg.BATCH_SIZE = int(batch_size)

    device = base.get_device()
    obs_cols = list(ckpt["obs_cols"])
    wave_prefix = ckpt.get("wave_prefix", "SIS_data_strain")
    wave1 = np.load(Path(data_root) / f"{wave_prefix}_1.npy", mmap_mode="r")
    wave2 = np.load(Path(data_root) / f"{wave_prefix}_2.npy", mmap_mode="r")
    lens_df = pd.read_csv(Path(data_root) / "lens.csv")
    lens_params_df = pd.read_csv(Path(data_root) / "lens_params.csv")
    obs_df = pd.read_csv(Path(data_root) / "observable_features.csv")
    n = min(len(wave1), len(wave2), len(lens_df), len(lens_params_df), len(obs_df))
    lens_df = lens_df.iloc[:n].copy().reset_index(drop=True)
    lens_params_df = lens_params_df.iloc[:n].copy().reset_index(drop=True)
    obs_df = obs_df.iloc[:n].copy().reset_index(drop=True)
    obs_df = entry.add_physics_optional_observables(obs_df, lens_params_df)

    for col in obs_cols:
        if col not in obs_df.columns:
            raise RuntimeError(f"Checkpoint expects missing observable column: {col}")

    norm_state = ckpt["obs_norm"]
    obs_norm = base.FeatureNormalizer(obs_cols)
    obs_norm.mean_ = np.asarray(norm_state["mean"], dtype=np.float32)
    obs_norm.std_ = np.asarray(norm_state["std"], dtype=np.float32)
    obs_norm.columns = list(norm_state["columns"])

    ds = base.MuDirectCompactDataset(
        wave1=wave1,
        wave2=wave2,
        lens_df=lens_df,
        obs_df=obs_df,
        obs_norm=obs_norm,
        indices=np.arange(n, dtype=np.int64),
        mode="val",
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=True)
    model = base.MuDirectCompactRegressor(obs_dim=len(obs_cols)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    eval_out = base.evaluate_model(model, loader, device)

    csv_path = out_dir / "external_predictions.csv"
    base.save_val_csv(eval_out, csv_path, obs_df=obs_df)
    plot_error = None
    try:
        plot_prediction_scatter_sci(csv_path, out_dir / "SCI_external_prediction_scatter.png", title="External validation prediction")
        plot_residual_sci(csv_path, out_dir / "SCI_external_residuals.png", title="External validation residuals")
    except ModuleNotFoundError as exc:
        plot_error = repr(exc)
        save_json({"plot_error": plot_error}, out_dir / "external_evaluation_plot_error.json")
    summary = {"run_label": run_label, "checkpoint": str(checkpoint_path), "data_root": str(data_root), **summarize_prediction_csv(csv_path)}
    if plot_error is not None:
        summary["plot_error"] = plot_error
    save_json(summary, out_dir / "external_evaluation_summary.json")
    return summary

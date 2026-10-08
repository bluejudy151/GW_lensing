#!/usr/bin/env python
# coding: utf-8
"""Utilities for the additional SIS mu0 validation experiments.

The helpers in this module are intentionally lightweight: they read existing
CSV outputs, compute analytic SIS anchors, train small tabular baselines, and
write publication-oriented summary tables/figures.  They do not run waveform
generation or deep training unless a wrapper script explicitly launches those
jobs.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data_generation" / "data_lens_sis_gw_physics_baseline"
DEFAULT_OUT_ROOT = EXPERIMENT_DIR / "outputs"

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

MACARON = [
    "#8DBCE8",
    "#F1AAA6",
    "#F4D67D",
    "#A9D0B0",
    "#CDB4DB",
    "#A0CED9",
    "#FFDAC1",
    "#B8E0D2",
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def progress_iter(iterable, total: int | None = None, desc: str = "", unit: str = "it"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def progress_write(message: str) -> None:
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)


def sci_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.45,
            "axes.prop_cycle": plt.cycler(color=MACARON),
        }
    )


def metric_dict(true: Iterable[float], pred: Iterable[float], prefix: str = "") -> dict[str, float]:
    true_arr = np.asarray(true, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    ok = np.isfinite(true_arr) & np.isfinite(pred_arr)
    true_arr = true_arr[ok]
    pred_arr = pred_arr[ok]
    key = (lambda name: f"{prefix}_{name}" if prefix else name)
    if true_arr.size == 0:
        return {
            key("n"): 0,
            key("mae"): np.nan,
            key("rmse"): np.nan,
            key("mape"): np.nan,
            key("bias"): np.nan,
            key("median_abs_err"): np.nan,
            key("p90_abs_err"): np.nan,
            key("pearson"): np.nan,
            key("spearman"): np.nan,
        }

    err = pred_arr - true_arr
    abs_err = np.abs(err)
    pearson = float(np.corrcoef(true_arr, pred_arr)[0, 1]) if true_arr.size > 1 else np.nan
    spearman = (
        float(pd.Series(true_arr).rank().corr(pd.Series(pred_arr).rank()))
        if true_arr.size > 1
        else np.nan
    )
    return {
        key("n"): int(true_arr.size),
        key("mae"): float(np.mean(abs_err)),
        key("rmse"): float(np.sqrt(np.mean(err**2))),
        key("mape"): float(np.mean(abs_err / (np.abs(true_arr) + 1e-8)) * 100.0),
        key("bias"): float(np.mean(err)),
        key("median_abs_err"): float(np.median(abs_err)),
        key("p90_abs_err"): float(np.quantile(abs_err, 0.90)),
        key("pearson"): pearson,
        key("spearman"): spearman,
    }


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_joined_physics_table(data_root: str | Path) -> pd.DataFrame:
    """Load lens, observed, parameter, and diagnostic tables into one frame."""

    data_root = Path(data_root)
    lens = pd.read_csv(data_root / "lens.csv").reset_index(drop=True)
    params = pd.read_csv(data_root / "lens_params.csv").reset_index(drop=True)
    obs = pd.read_csv(data_root / "observable_features.csv").reset_index(drop=True)
    diag = read_optional_csv(data_root / "diagnostic_features.csv").reset_index(drop=True)

    n = min(len(lens), len(params), len(obs), len(diag) if len(diag) else len(lens))
    if n <= 0:
        raise RuntimeError(f"No usable rows under {data_root}")

    lens = lens.iloc[:n].copy()
    params = params.iloc[:n].copy()
    obs = obs.iloc[:n].copy()
    diag = diag.iloc[:n].copy() if len(diag) else pd.DataFrame(index=np.arange(n))

    # Generated releases use event_id in observable/diagnostic tables.  Older
    # lens/parameter tables may omit it, in which case their row order is the
    # generator's declared event order.  Never silently combine tables with
    # conflicting identifiers.
    if "event_id" in obs.columns:
        event_id = pd.to_numeric(obs["event_id"], errors="raise").astype(np.int64).to_numpy()
        if len(np.unique(event_id)) != len(event_id):
            raise RuntimeError("observable_features.csv contains duplicate event_id values")
        for name, frame in (("lens_params", params), ("diagnostic_features", diag)):
            if "event_id" in frame.columns:
                other = pd.to_numeric(frame["event_id"], errors="raise").astype(np.int64).to_numpy()
                if not np.array_equal(event_id, other):
                    raise RuntimeError(f"event_id order mismatch between observable and {name}")
    else:
        event_id = np.arange(n, dtype=np.int64)

    base = pd.DataFrame({"event_id": event_id})
    base["mu0_true"] = pd.to_numeric(lens["mu_0"], errors="coerce").to_numpy()
    base["mu1_true"] = pd.to_numeric(lens.get("mu_1", np.nan), errors="coerce").to_numpy()
    base["y_true"] = pd.to_numeric(lens.get("y", np.nan), errors="coerce").to_numpy()
    base["t_d_true"] = pd.to_numeric(lens.get("t_d", np.nan), errors="coerce").to_numpy()

    parts = [base]
    for prefix, frame in (("lens", lens), ("params", params), ("obs", obs), ("diag", diag)):
        sub = frame.drop(columns=["event_id"], errors="ignore").copy()
        sub.columns = [f"{prefix}__{col}" for col in sub.columns]
        parts.append(sub.reset_index(drop=True))

    return pd.concat(parts, axis=1, copy=False)


def align_with_prediction_csv(joined: pd.DataFrame, prediction_csv: str | Path | None) -> pd.DataFrame:
    """Align the full table to a validation prediction CSV when one is supplied."""

    if not prediction_csv:
        return joined.copy().reset_index(drop=True)

    prediction_csv = Path(prediction_csv)
    pred = pd.read_csv(prediction_csv)
    id_col = "event_id" if "event_id" in pred.columns else "idx" if "idx" in pred.columns else None
    if id_col is None:
        raise RuntimeError(f"{prediction_csv} does not contain event_id/idx for alignment.")

    idx = pd.to_numeric(pred[id_col], errors="coerce").astype("Int64")
    if idx.isna().any():
        raise RuntimeError(f"{prediction_csv} contains non-integer event ids.")
    idx_np = idx.to_numpy(dtype=np.int64)
    if idx_np.min(initial=0) < 0 or idx_np.max(initial=0) >= len(joined):
        raise RuntimeError("Prediction event ids exceed the dataset length.")

    aligned = joined.iloc[idx_np].copy().reset_index(drop=True)
    for col in pred.columns:
        if col == id_col:
            continue
        aligned[col] = pred[col].to_numpy()
    aligned["prediction_event_id"] = idx_np
    aligned["prediction_csv"] = str(prediction_csv)
    return aligned


def first_existing(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def to_float_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def y_to_mu0(y: Iterable[float], eps: float = 1e-4) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    y_arr = np.clip(np.abs(y_arr), eps, 1.0 - eps)
    return 1.0 + 1.0 / y_arr


def ratio_to_mu0(ratio: Iterable[float], ratio_is_amplitude: bool, eps: float = 1e-4) -> np.ndarray:
    r = np.asarray(ratio, dtype=np.float64)
    r = np.abs(r)
    if ratio_is_amplitude:
        r = r**2
    r = np.clip(r, eps, 1.0 - eps)
    y = (1.0 - r) / (1.0 + r)
    y = np.clip(y, eps, 1.0 - eps)
    return 1.0 + 1.0 / y


def add_anchor_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Add analytic/semi-analytic SIS anchor predictions.

    Returns the modified frame, method columns, and display labels.
    """

    out = df.copy()
    labels: dict[str, str] = {}
    method_cols: list[str] = []

    def register(name: str, values: Iterable[float], label: str) -> None:
        arr = np.array(values, dtype=np.float64, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        out[name] = arr
        method_cols.append(name)
        labels[name] = label

    if "mu0_pred" in out.columns:
        register("method__neural_current", out["mu0_pred"], "current neural model")

    diag_img = first_existing(
        out,
        [
            "diag__mu0_from_observed_image_asymmetry",
            "params__mu0_from_image_asymmetry_observed",
        ],
    )
    if diag_img:
        register("method__anchor_img_diag", to_float_series(out, diag_img), "image-asymmetry anchor")

    asym_col = first_existing(
        out,
        [
            "obs__image_position_asymmetry_observed",
            "params__image_position_asymmetry_observed",
        ],
    )
    if asym_col:
        register("method__anchor_img_asym", y_to_mu0(to_float_series(out, asym_col)), "observed asymmetry")

    theta_plus = first_existing(out, ["obs__theta_plus_abs_observed", "params__theta_plus_abs_observed"])
    theta_minus = first_existing(out, ["obs__theta_minus_abs_observed", "params__theta_minus_abs_observed"])
    if theta_plus and theta_minus:
        plus = to_float_series(out, theta_plus).to_numpy()
        minus = to_float_series(out, theta_minus).to_numpy()
        y_theta = (plus - minus) / (plus + minus + 1e-12)
        register("method__anchor_theta_distance", y_to_mu0(y_theta), "theta-distance anchor")

    for col, label, amp_like in [
        ("obs__peak_amp_ratio_21", "peak amplitude ratio", True),
        ("obs__rms_ratio_21", "RMS amplitude ratio", True),
        ("obs__energy_ratio_21", "energy ratio", False),
        ("obs__env_area_ratio_21", "envelope area ratio", True),
        ("obs__win0p5_peak_amp_ratio_21", "0.5s peak ratio", True),
        ("obs__win0p5_rms_ratio_21", "0.5s RMS ratio", True),
    ]:
        if col in out.columns:
            register(f"method__anchor_{col.replace('obs__', '')}", ratio_to_mu0(out[col], amp_like), label)

    diag_td = first_existing(out, ["diag__mu0_from_observed_t_d_using_true_K"])
    if diag_td:
        register("method__anchor_td_trueK", to_float_series(out, diag_td), "time-delay anchor with true K")

    if "y_true" in out.columns:
        register("method__oracle_y", y_to_mu0(out["y_true"]), "oracle y")

    if "lens__R_abs_21" in out.columns:
        register("method__oracle_R_abs", ratio_to_mu0(out["lens__R_abs_21"], False), "oracle |R21|")

    geometry_for_hybrid = [
        c
        for c in [
            "method__anchor_img_diag",
            "method__anchor_img_asym",
            "method__anchor_theta_distance",
        ]
        if c in method_cols
    ]
    if geometry_for_hybrid:
        tau_stack = []
        for col in geometry_for_hybrid:
            vals = np.asarray(out[col], dtype=np.float64)
            tau_stack.append(np.log(np.maximum(vals - 1.0, 1e-8)))
        tau = np.nanmedian(np.vstack(tau_stack), axis=0)
        register("method__hybrid_geometry_median", 1.0 + np.exp(tau), "geometry-median anchor")

    return out, method_cols, labels


def subset_masks(df: pd.DataFrame, tail_quantile: float = 0.90) -> dict[str, np.ndarray]:
    mu = pd.to_numeric(df["mu0_true"], errors="coerce").to_numpy(dtype=np.float64)
    masks: dict[str, np.ndarray] = {"overall": np.isfinite(mu)}
    if np.isfinite(mu).any():
        threshold = float(np.nanquantile(mu, tail_quantile))
        masks[f"tail_top_{int(round((1.0 - tail_quantile) * 100))}pct"] = mu >= threshold
    for cutoff in [3.0, 5.0, 10.0]:
        masks[f"mu0_ge_{cutoff:g}"] = mu >= cutoff
    if "y_true" in df.columns:
        y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(y).any():
            y_thr = float(np.nanquantile(y, 0.10))
            masks["small_y_bottom_10pct"] = y <= y_thr
    return masks


def summarize_methods(
    df: pd.DataFrame,
    method_cols: Sequence[str],
    labels: dict[str, str] | None = None,
    tail_quantile: float = 0.90,
    min_subset_n: int = 3,
) -> pd.DataFrame:
    labels = labels or {}
    rows: list[dict[str, Any]] = []
    masks = subset_masks(df, tail_quantile=tail_quantile)
    true = pd.to_numeric(df["mu0_true"], errors="coerce").to_numpy(dtype=np.float64)

    for subset, mask in masks.items():
        if int(np.sum(mask)) < min_subset_n:
            continue
        for col in method_cols:
            if col not in df.columns:
                continue
            pred = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            metrics = metric_dict(true[mask], pred[mask])
            rows.append(
                {
                    "subset": subset,
                    "method": col,
                    "label": labels.get(col, col.replace("method__", "")),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def write_latex_three_line_table(
    summary: pd.DataFrame,
    path: str | Path,
    caption: str,
    label: str,
    subset: str = "overall",
    top_n: int | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    work = summary.loc[summary["subset"] == subset].copy()
    work = work.sort_values(["mae", "rmse"], ascending=True)
    if top_n:
        work = work.head(top_n)

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabularx}{\linewidth}{@{}L c c c c c@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{MAE} & \textbf{RMSE} & \textbf{MAPE (\%)} & \textbf{Bias} & \textbf{Pearson} \\",
        r"\midrule",
    ]
    for _, row in work.iterrows():
        lines.append(
            "{} & {:.3f} & {:.3f} & {:.2f} & {:.3f} & {:.3f} \\\\".format(
                str(row["label"]).replace("_", r"\_"),
                float(row["mae"]),
                float(row["rmse"]),
                float(row["mape"]),
                float(row["bias"]),
                float(row["pearson"]) if np.isfinite(row["pearson"]) else float("nan"),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_bar(
    summary: pd.DataFrame,
    out_path: str | Path,
    subset: str = "overall",
    metric: str = "mae",
    title: str = "",
    max_methods: int = 12,
) -> None:
    sci_style()
    import matplotlib.pyplot as plt

    work = summary.loc[summary["subset"] == subset].copy()
    work = work.sort_values(metric, ascending=True).head(max_methods)
    if work.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6.2, 0.58 * len(work)), 3.7))
    x = np.arange(len(work))
    ax.bar(x, work[metric], color=[MACARON[i % len(MACARON)] for i in range(len(work))], width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(work["label"], rotation=35, ha="right")
    ylabels = {
        "mae": r"MAE of $\mu_0$",
        "rmse": r"RMSE of $\mu_0$",
        "mape": r"MAPE of $\mu_0$ (%)",
        "bias": r"Bias of $\hat{\mu}_0-\mu_0$",
    }
    ax.set_ylabel(ylabels.get(metric, metric))
    ax.set_title(title or f"{subset}: {metric}")
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_true_pred_panels(
    df: pd.DataFrame,
    methods: Sequence[str],
    labels: dict[str, str],
    out_path: str | Path,
    title: str = "Anchor and model comparison",
) -> None:
    sci_style()
    import matplotlib.pyplot as plt

    methods = [m for m in methods if m in df.columns][:6]
    if not methods:
        return

    ncols = 3 if len(methods) > 2 else len(methods)
    nrows = int(math.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.35 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    true = pd.to_numeric(df["mu0_true"], errors="coerce").to_numpy(dtype=np.float64)
    finite_true = true[np.isfinite(true)]
    lo = float(np.nanmin(finite_true)) if finite_true.size else 1.0
    hi = float(np.nanmax(finite_true)) if finite_true.size else 2.0
    pad = 0.04 * max(hi - lo, 1e-6)
    lo_plot = max(1.0, lo - pad)
    hi_plot = hi + pad

    for ax, method, color in zip(axes, methods, MACARON):
        pred = pd.to_numeric(df[method], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(true) & np.isfinite(pred)
        ax.scatter(true[ok], pred[ok], s=13, alpha=0.58, color=color, edgecolors="none")
        ax.plot([lo_plot, hi_plot], [lo_plot, hi_plot], color="#444444", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo_plot, hi_plot)
        ax.set_ylim(lo_plot, hi_plot)
        ax.set_xlabel(r"True $\mu_0$")
        ax.set_ylabel(r"Predicted $\mu_0$")
        ax.set_title(labels.get(method, method.replace("method__", "")))

    for ax in axes[len(methods) :]:
        ax.axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution(
    df: pd.DataFrame,
    methods: Sequence[str],
    labels: dict[str, str],
    out_path: str | Path,
    title: str = "Residual distribution",
) -> None:
    sci_style()
    import matplotlib.pyplot as plt

    true = pd.to_numeric(df["mu0_true"], errors="coerce").to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, method in enumerate(methods[:8]):
        if method not in df.columns:
            continue
        pred = pd.to_numeric(df[method], errors="coerce").to_numpy(dtype=np.float64)
        resid = pred - true
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            continue
        ax.hist(
            resid,
            bins=40,
            histtype="step",
            linewidth=1.3,
            color=MACARON[i % len(MACARON)],
            label=labels.get(method, method.replace("method__", "")),
            density=True,
        )
    ax.axvline(0.0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"Residual $\hat{\mu}_0-\mu_0$")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


WAVE_PROXY_COLS = [
    "obs__peak_time_diff",
    "obs__env_peak_time_diff",
    "obs__arrival_time_1_sigma",
    "obs__arrival_time_2_sigma",
    "obs__peak_amp_ratio_21",
    "obs__rms_ratio_21",
    "obs__energy_ratio_21",
    "obs__peak_amp_ratio_local_65",
    "obs__rms_ratio_local_65",
    "obs__energy_ratio_local_65",
    "obs__env_area_ratio_21",
    "obs__xcorr_lag",
    "obs__norm_xcorr_max",
    "obs__xcorr_lag_win65",
    "obs__norm_xcorr_max_win65",
    "obs__aligned_l1_residual",
    "obs__aligned_l2_residual",
    "obs__spec_centroid_diff",
    "obs__band_energy_ratio_low_21",
    "obs__band_energy_ratio_mid_21",
    "obs__band_energy_ratio_high_21",
    "obs__local_noise_rms_1",
    "obs__local_noise_rms_2",
    "obs__window_snr_like_1",
    "obs__window_snr_like_2",
]

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

# Columns that are generated from labels, privileged simulator truth, or
# injected quantities.  They may be retained in joined diagnostic tables but
# must never enter a deployable observed-input feature matrix.
FORBIDDEN_FEATURE_TOKENS = (
    "mu_", "y_true", "u_true", "beta_true", "t_d_true", "A21_true",
    "R_abs_21", "mu_ratio_abs_21", "amp_ratio_21_true", "theta_E_true",
    "sigma_v_true", "z_l_true", "z_s_true", "source_luminosity_distance",
    "optimal_snr", "injected_snr",
)


def audit_feature_columns(columns: Sequence[str], allow_privileged: bool = False) -> list[str]:
    """Fail fast on target-derived or privileged columns.

    This guard is intentionally conservative.  A caller must explicitly opt
    in for diagnostic/oracle experiments instead of accidentally training on
    simulator truth.
    """
    cols = list(columns)
    if allow_privileged:
        return cols
    bad = [c for c in cols if any(tok.lower() in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ValueError("Forbidden privileged/target-derived features: " + ", ".join(bad))
    return cols


def numeric_feature_cols(df: pd.DataFrame, feature_set: str = "all_safe") -> list[str]:
    if feature_set == "wave_proxy":
        candidates = WAVE_PROXY_COLS
    elif feature_set == "geometry":
        candidates = GEOMETRY_COLS
    elif feature_set == "time_lens":
        candidates = [
            "obs__peak_time_diff",
            "obs__env_peak_time_diff",
            "obs__arrival_time_1_sigma",
            "obs__arrival_time_2_sigma",
            "obs__z_s_observed",
            "obs__z_l_observed",
            "obs__z_l_over_z_s",
            "obs__sigma_v_observed",
            "obs__log_sigma_v_observed",
            "obs__theta_E_observed",
            "obs__image_separation_observed",
        ]
    elif feature_set == "all_safe":
        candidates = WAVE_PROXY_COLS + GEOMETRY_COLS
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    cols: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            cols.append(col)
    cols = list(dict.fromkeys(cols))
    return audit_feature_columns(cols)


@dataclass
class SklearnRun:
    name: str
    model: str
    feature_set: str = "all_safe"
    residual_anchor: str | None = None
    random_state: int = 42


def make_sklearn_model(model_name: str, random_state: int = 42):
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on server env
        raise RuntimeError("scikit-learn is required for this experiment.") from exc

    if model_name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if model_name == "histgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.055,
                max_leaf_nodes=31,
                max_bins=128,
                l2_regularization=0.02,
                verbose=1,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=12,
                random_state=random_state,
            ),
        )
    if model_name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=320,
                min_samples_leaf=3,
                verbose=1,
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if model_name == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=320,
                min_samples_leaf=2,
                verbose=1,
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if model_name == "mlp":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=8e-4,
                max_iter=420,
                early_stopping=True,
                verbose=True,
                random_state=random_state,
            ),
        )
    raise ValueError(f"Unsupported sklearn model: {model_name}")


def build_train_val_indices(
    n: int,
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_event_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if validation_event_ids is not None:
        val_idx = np.asarray(sorted(set(int(i) for i in validation_event_ids)), dtype=np.int64)
        all_idx = np.arange(n, dtype=np.int64)
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        return train_idx, val_idx

    idx = np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_fraction))
    return np.sort(idx[:n_train]), np.sort(idx[n_train:])


def fit_predict_tau_model(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    run: SklearnRun,
) -> tuple[np.ndarray, list[str]]:
    features = numeric_feature_cols(df, run.feature_set)
    if not features:
        raise RuntimeError(f"No numeric features for {run.feature_set}")

    X_train = df.iloc[train_idx][features].apply(pd.to_numeric, errors="coerce")
    X_val = df.iloc[val_idx][features].apply(pd.to_numeric, errors="coerce")
    tau_true = np.log(np.maximum(pd.to_numeric(df["mu0_true"], errors="coerce").to_numpy() - 1.0, 1e-8))

    if run.residual_anchor:
        if run.residual_anchor not in df.columns:
            raise RuntimeError(f"Missing residual anchor: {run.residual_anchor}")
        anchor_tau = np.log(
            np.maximum(pd.to_numeric(df[run.residual_anchor], errors="coerce").to_numpy() - 1.0, 1e-8)
        )
        y_train = tau_true[train_idx] - anchor_tau[train_idx]
    else:
        anchor_tau = np.zeros(len(df), dtype=np.float64)
        y_train = tau_true[train_idx]

    model = make_sklearn_model(run.model, random_state=run.random_state)
    model.fit(X_train, y_train)
    pred_component = model.predict(X_val)
    tau_pred = pred_component + anchor_tau[val_idx]
    return 1.0 + np.exp(tau_pred), features

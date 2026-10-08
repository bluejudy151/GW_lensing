# Conditional GW lensing magnification inference

Core code for data generation, conditional magnification prediction and the
main scientific comparisons. All filenames and documentation use English.
Simulated data, saved predictions, plots and manuscript files are excluded.

## Contents

| Path | Purpose |
| --- | --- |
| SIS_GW_physics_baseline.py | Generate the selected SIS waveform and observable catalogue |
| core_experiments/controlled_inputs.py | Primary HistGB comparison of five observed input sets |
| core_experiments/common.py | Input contracts, identity partitions, HistGB fitting and calibration |
| core_experiments/evaluate_selected_tail.py | Evaluate an independent selected SIS sample at small source offset |
| core_experiments/matched_transfer.py and nfw_scale_control.py | Population matching and NFW angular normalization controls |
| core_experiments/astrometry.py, diagnostics.py and targeted_tail.py | Astrometric sensitivity, Jacobian diagnostics and targeted tails |
| experiments/extended_experiments/ | Four-family catalogue generation, waveform features and regression evaluation |
| scripts/sis_bayesian_primary.py | Coordinate-likelihood Bayesian comparison |
| scripts/review_dependency_tail.py | Calibration-unit and SIS training/noise controls |
| train_mu0_physics_baseline.py | Optional waveform-plus-observables neural baseline training |
| weights/sis_geometry_neural_baseline.pt | Existing neural-baseline checkpoint; not the primary HistGB estimator |

Supporting modules and focused scientific checks are included only where needed
by these workflows. No publication packaging or manuscript editing scripts are
included.

## Environment

Use Python 3.12. requirements-experiments.txt records the original package
versions rather than a freshly verified installation lockfile. NumPy, pandas,
SciPy, scikit-learn and threadpoolctl are needed for HistGB experiments.
Astropy and lenstronomy are needed for lens generation; Bilby, LALSuite and
GWpy for waveform generation; PyTorch for the optional neural baseline.

Run commands from the repository root. Each entry point provides --help.
Generate data locally or supply external input directories explicitly.

```bash
python SIS_GW_physics_baseline.py --save-dir reproduced/sis_2500 --n-events 2500 --max-attempts 25000 --seed 238
python core_experiments/run_experiments.py --help
python core_experiments/evaluate_selected_tail.py --help
python experiments/extended_experiments/generate_universal_latent_benchmark.py --help
python experiments/extended_experiments/evaluate_universal_identifiability.py --help
python scripts/sis_bayesian_primary.py --help
python train_mu0_physics_baseline.py --help
```

The four-family generator uses a generated SIS catalogue as its empirical
source prior. Core experiment scripts consume generated catalogues and saved
partition records; they are not data-free inference examples. Specify output
directories separately from inputs. A small smoke run checks functionality,
not the manuscript's numerical results.

## Checkpoint

The included checkpoint is copied from the strict-observed-geometry neural
baseline run in the original release. It stores model_state, obs_norm,
target_mode, obs_cols, waveform selection, configuration and validation metrics.
Use the matching neural architecture and normalization when loading it; it is
not a standalone geometry-only HistGB model. Embedded historical paths may
refer to the original training environment and must be overridden locally.

No saved primary HistGB model was found in the original release. Its training
code is included; retrain it on the generated catalogue and the specified
identity partitions. This checkpoint does not replace that model or reproduce
its reported MAE.

FILES.txt lists packaged files and SHA256SUMS records their checksums. See
NOTICE.md for the existing attribution and licence status.

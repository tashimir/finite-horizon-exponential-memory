# Finite-horizon exponential memory

Data, figures and numerical programs for Pedro M. M. de Castro, *Finite-horizon phase transitions in optimal exponential memory for Euclidean connections*.

The article studies how initialization changes the optimal constant memory parameter when the horizon grows and the edge-cost power approaches one. The numerical material compares asymptotic predictions with finite objectives, objective derivatives and paired trajectory costs for uniform-ball inputs.

## Start here

- [Numerical methods and reproduction guide](SupplementaryMaterial1.pdf) gives the moment derivation, truncation bounds and interpretation of the computations.
- The `code/`, `data/` and `figures/` directories contain the reproduction programs, tabular data and six figures supplied in Supplementary Material 2.
- [Trajectory costs for the twelve-scenario validation](data/calibration_experiment/calibration_trajectory_costs.npz) contains the complete arrays for that experiment, supplied in Supplementary Material 3.
- [Reproduction commands](REPRODUCING.md) distinguish rendering supplied arrays from regenerating their numerical values.
- [Data dictionary](DATA_DICTIONARY.md) explains the scientific quantities and file families.

This collection contains the files associated with the article. Individual CSV files can be inspected directly. The expanded paired experiment supplies block moments sufficient to reconstruct every reported estimate and confidence interval; its full trajectory arrays are retained by the author.

## Figures

| Scientific figure | PDF | Data |
|---|---|---|
| Geometric construction | [Figure](figures/geometric_construction.pdf) | [Coordinates and retained edges](data/geometric_construction) |
| Joint-window phase diagram | [Figure](figures/joint_window_phase_diagram.pdf) | Analytical diagram derived from the article's theorems |
| Finite-horizon phase transitions | [Figure](figures/finite_horizon_phase_transitions.pdf) | [Optimizer locations and size-pair diagnostics](data/finite_horizon_phase_transitions) |
| Local objective geometry | [Figure](figures/local_objective_geometry.pdf) | [Values, derivatives and resolution diagnostics](data/local_objective_geometry) |
| Measured calibration gains | [Figure](figures/measured_calibration_gains.pdf) | [Paired estimates and block moments](data/measured_calibration_gains) |
| Stationary calibration comparison | [Figure](figures/stationary_calibration_comparison.pdf) | [Complete-objective comparisons](data/stationary_calibration_comparison) |

[FIGURE_CATALOG.csv](FIGURE_CATALOG.csv) gives the same map in machine-readable form.

## Reproduction and scope

Install the Python dependencies with `python -m pip install -r requirements.txt`. Rendering also requires LaTeX with Latin Modern and the standard Matplotlib LaTeX dependencies. Regenerating the expanded paired trajectories requires a GNU/Linux C++17 compiler with OpenMP. The detailed commands and their outputs are in [REPRODUCING.md](REPRODUCING.md).

The finite-horizon phase and local-objective figures can be rendered directly from their supplied numerical arrays. Their original multidimensional radial-Poisson solver is available from the author on reasonable request and is not included here. The moment-based stationary comparison and the paired Monte Carlo calculations have their own supplied implementations. Grid diagnostics, analytical series-tail bounds and sampling intervals describe different sources of uncertainty and retain their separate meanings.

Positive reported savings favor the finite-balance rule. Negative values are cost increases and are retained. Pointwise confidence intervals in the paired experiment describe sampling uncertainty; deterministic moment calculations carry the numerical qualifications given in the methods supplement. A separate validation program compares the moment implementation with exact formulas and 100-digit reference calculations through 78 recorded checks on selected cases.

Tables and figure annotations report ordinary numerical results with at most three decimal places. A positive percentage smaller than 0.001% is described by that upper threshold when its precise magnitude is unnecessary for interpretation. Logarithmic axes retain powers of ten to show the scale of small effects. The supplied data, curve coordinates and interval endpoints retain their computational precision. The one-dimensional benchmark table reports percentage deviations of the normalized update and excess cost from their limiting value one; both columns are derived from `data/analytical_benchmarks/exact_1d_finite.csv`.

[CITATION.cff](CITATION.cff) supplies citation metadata. Cite the associated article and identify the repository version or commit used. Correspondence: [pmmc@cin.ufpe.br](mailto:pmmc@cin.ufpe.br).

Copyright 2026 Pedro M. M. de Castro. The terms in [COPYRIGHT.md](COPYRIGHT.md) apply.

The journal supplements consist of the methods PDF (Supplementary Material 1), the code and data archive (Supplementary Material 2), and the complete twelve-scenario trajectory arrays (Supplementary Material 3). Extract both archives into the same directory. The methods PDF is also included in the code and data archive.

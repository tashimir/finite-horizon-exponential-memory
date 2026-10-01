# Data dictionary

This collection accompanies *Finite-horizon phase transitions in optimal exponential memory for Euclidean connections*, by Pedro M. M. de Castro. [README.md](README.md) locates the figures and downloads; [REPRODUCING.md](REPRODUCING.md) gives executable commands and the scope of each reproduction.

## Model and units

The input ball has radius one, the dimension is d and the number of insertions is N. The memory is gamma, the update fraction is delta=1-gamma, the edge-cost power is alpha and epsilon=alpha-1. A trajectory starts at x_0=p_0. The retained-edge factor is gamma^alpha+delta^alpha, so reported total costs include both retained edges at every insertion. The scaled power is r=(alpha-1)log(N)/lambda_star(d).

The notation `delta_N` and `Delta_N` distinguishes the analytical stationary and critical scales defined in the article. A numerical optimizer and an asymptotic scale have different meanings. Empty reference fields represent quantities unavailable for that scenario.

## Construction and asymptotic diagnostics

| Directory | Contents and interpretation |
|---|---|
| `data/geometric_construction` | Coordinates of the input and memory sites, retained edges and a local insertion example underlying the geometric illustration. |
| `data/finite_horizon_phase_transitions` | Four convergence arrays, raw evaluations, size-pair exponents, correction comparisons and discretization parameters. `log2N` identifies the horizon; `log2N_right` identifies the larger member of a size pair; `lambda_ratio` is r. The common figure legend denotes 4N for the size-pair exponents and N for the other quantities. |
| `data/local_objective_geometry` | Profiles at three evaluation resolutions, errors by derivative order and comparisons between resolutions. The normalized coordinate z sets delta=Delta_N z. The C0, C1 and C2 errors are maxima on the stored grid and do not certify continuous suprema. |
| `data/dimension_constants` | Initialization constants, thresholds, critical coefficients and raw quadrature evaluations supporting the article's dimension table. The full numerical dimension set is retained. |
| `data/analytical_benchmarks` | Exact one-dimensional cost checks, independently evaluated constants, scalar critical-window checks, symbolic identities and a three-case Monte Carlo cost check. |

In `exact_1d_finite.csv`, `delta` is the numerical minimizer of the exact d=1, alpha=1 objective; `sqrtN_delta` and `six_scaled_excess` are the two normalized quantities reported in the benchmark table. `absolute_slope` and `curvature_positive` record local numerical checks. In the scalar model, the paths `centered`, `below_weak` and `above_weak` have epsilon=[lambda_star(2)+s]/log N with s equal to 0, -1/sqrt(log N) and +1/sqrt(log N). The Lambert quantity is w. These numerical checks retain the hypotheses stated in the article.

## Paired calibration data

The expanded experiment in `data/measured_calibration_gains` has 259 scenarios: 31 scaled powers and six fixed powers at seven horizons. Each horizon uses 32,768 independent input sequences shared across 74 policies. Different horizon streams are independent. Write A for finite-rule trajectory cost, B for analytical-stationary-scale cost and D=A-B.

| Field | Definition |
|---|---|
| `N`, `family`, `case`, `r`, `alpha` | Scenario identifiers, path family, scaled power and cost power. |
| `delta_finite`, `delta_stationary_scale` | Scalar-balance parameter and analytical stationary scale fixed before simulation. |
| `finite_column`, `stationary_column` | The two policy columns in the trajectory computation. |
| `mean_finite`, `mean_stationary_scale`, `mean_difference` | Sample means of A, B and D. |
| `variance_difference`, `variance_stationary_scale` | Unbiased sample variances of D and B. |
| `covariance_difference_stationary_scale` | Unbiased sample covariance of D and B. |
| `saving_estimate_pct` | -100 mean(D)/mean(B), measured in percentage points. |
| `saving_standard_error_pct` | 100 sd[D-(mean(D)/mean(B))B]/[sqrt(m) mean(B)]. |
| `saving_lower95_pct`, `saving_upper95_pct` | Estimate minus and plus 1.96 standard errors. |
| `saved_cost_per_input` | -mean(D)/N. |

Positive savings favor the finite rule; negative values are retained as cost increases. All intervals are pointwise paired normal delta-method intervals. Each scenario has 256 disjoint blocks of 128 trajectories, giving 66,304 rows in the block-moment table. Pooling includes both within-block and between-block contributions. The supplied moments reconstruct all estimates and intervals; the full expanded trajectory arrays are retained by the author.

The `policies_N*.tsv` files have no header. Their columns are alpha and delta, in policy order. The design JSON specifies the random generator and seed mapping. The `observed_zero_crossing_brackets.json` file records adjacent sampled values with opposite estimated signs. These brackets describe the sampled grid and do not certify continuous zero boundaries. The observed-loss envelope has no simultaneous confidence interpretation.

## Twelve-scenario validation

`data/calibration_experiment` contains a separate experiment at N=256,1024,4096 and r=0,0.5,1,2. The archived numerical finite-horizon reference is available at the last two horizons; reference fields are empty at N=256. The complete NPZ arrays have shape (policies, trajectories). The arrays `policy_r_N...` and `policy_name_N...` identify the rows of `costs_N...`.

For the reference cost R, `reference_excess_estimate_pct` is 100[mean(A)/mean(R)-1], with a paired ratio standard error. Reference parameters are supplied inputs; the included simulation program regenerates trajectory costs at those parameters.

## Deterministic stationary comparison

`data/stationary_calibration_comparison` contains the complete-objective comparison used in the stationary-calibration figure. The finite-balance rule, analytical stationary scale and numerically minimized stationary choice are evaluated in the same finite objective. A saving against comparator B is 100[1-F(balance)/F(B)]. These deterministic values have numerical error diagnostics and do not carry Monte Carlo confidence intervals.

| Field | Definition |
|---|---|
| `N`, `log2_N`, `family`, `r`, `alpha` | Horizon, its base-two logarithm, path family, scaled power and cost power. |
| `delta_finite`, `delta_stationary_scale`, `delta_stationary` | Finite-balance choice, analytical stationary scale and numerically minimized stationary choice. |
| `order` | Selected order of the potential-series calculation. |
| `H`, `H_truncation_bound` | Initialization constant used in the balance and its series-tail diagnostic. |
| `excess_finite`, `excess_stationary`, `excess_scale` | Finite mean costs after subtracting c=2/(2+alpha). The corresponding total cost is N(c+excess). |
| `saving_vs_stationary_pct`, `saving_vs_scale_pct` | Percentage saving of the finite-balance choice against the named comparator. |
| `stationary_gain_truncation_bound_pct`, `scale_gain_truncation_bound_pct` | Gain-error bounds propagated from omitted potential-series terms. |
| `stationary_gain_error_estimate_pct`, `scale_gain_error_estimate_pct` | The larger of the observed order change and the series-tail diagnostic. |
| `stationary_sign_resolved`, `scale_sign_resolved` | Whether that diagnostic is smaller than one tenth of the absolute computed gain. These flags do not enclose floating-point or search error. |
| `gain_error_scope` | Scope statement accompanying the gain-error diagnostics. |

The crossing table contains `comparator`, `r_lower`, `r_upper`, the gains at both endpoints, their numerical diagnostics and the number of bisection steps. `stationary_calibration_crossing_evaluations.csv` retains every additional sampled evaluation. A bracket records opposite numerically resolved signs; no uniqueness theorem for a continuous crossing is inferred from it. The design JSON records the grid, approximation orders, numerical method and arithmetic precision.

The radial-moment solver distinguishes an analytical bound on omitted potential-series terms from comparisons between truncation orders. Those bounds do not enclose every floating-point error, and the parameter search does not certify a global optimum over the continuous interval. [Supplementary Material 1](SupplementaryMaterial1.pdf) gives the method and its scope.

## Exact and high-precision validation records

Four files in `data/stationary_calibration_comparison` document separate checks of the moment implementation against exact formulas and independent high-precision calculations. The twelve selected calibration cases supplement formula checks to give 78 comparisons; their scope is the cases and tolerances explicitly recorded. [REPRODUCING.md](REPRODUCING.md) gives the command that regenerates the three output files from the case design.

| File | Contents and interpretation |
|---|---|
| `numerical_validation_cases.csv` | The twelve-case input design. `case_id` identifies a case; `N`, `alpha` and `order` specify its horizon, power and calculation order. The optional `r`, `comparator` and `endpoint` descriptors locate selected sign-change endpoints. Empty descriptors are unnecessary for executing a case: the calculation uses its `N`, `alpha` and `order`. |
| `numerical_validation.csv` | One row per comparison. `quantity` names the checked quantity; `N`, `alpha`, `delta` and `order` give its parameters when applicable. `calculated_value` is the implementation output, `reference_value` is the independent or exact reference, and `absolute_difference` is their absolute difference. `passed` records whether that difference is at most `tolerance`. `reference_method` states how the reference was obtained. |
| `high_precision_reference_cases.csv` | Reference parameters and gain estimates for the twelve selected calibration cases. The three `delta_...` fields and two `saving_..._pct` fields have the meanings given above. Each `..._averaged_tail_moment` is the finite-horizon average of the radial moment of degree 2(order+1), used to bound the omitted potential-series terms for the named policy. |
| `numerical_validation.json` | Decimal precision of the reference calculations, checksum of the solver tested, NumPy extended-precision accuracy, number of checks, aggregate outcome and the scope of the comparisons. |

The reference calculations use 100 decimal digits. Their agreement with the implementation provides evidence on the recorded cases; numerical interval certification and a proof of global continuous minimization remain outside the scope of these records.

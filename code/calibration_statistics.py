# Reconstruct paired cost estimates and confidence intervals from trajectory moments.
import argparse
import math
from pathlib import Path


def summarize(n, mean_b, mean_d, var_b, var_d, cov, residual_variance=None):
    ratio = mean_d / mean_b
    variance = (math.fsum([var_d, ratio * ratio * var_b, -2 * ratio * cov])
                if residual_variance is None else residual_variance)
    if variance < 0:
        raise ValueError('Negative paired residual variance')
    se = 100 * math.sqrt(variance / n) / mean_b
    estimate = -100 * ratio
    return dict(trajectories=n, mean_finite=mean_b + mean_d,
                mean_stationary_scale=mean_b, mean_difference=mean_d,
                variance_difference=var_d, variance_stationary_scale=var_b,
                covariance_difference_stationary_scale=cov,
                saving_estimate_pct=estimate, saving_standard_error_pct=se,
                saving_lower95_pct=estimate - 1.96 * se,
                saving_upper95_pct=estimate + 1.96 * se)


def trajectory_moments(a, b):
    n = len(a)
    if n < 2 or len(b) != n:
        raise ValueError('At least two paired trajectories are required')
    d = [x - y for x, y in zip(a, b)]
    mb, md = math.fsum(b) / n, math.fsum(d) / n
    bc, dc = [y - mb for y in b], [y - md for y in d]
    vb = math.fsum(y * y for y in bc) / (n - 1)
    vd = math.fsum(y * y for y in dc) / (n - 1)
    cov = math.fsum(x * y for x, y in zip(bc, dc)) / (n - 1)
    residual = [x - (md / mb) * y for x, y in zip(dc, bc)]
    return summarize(n, mb, md, vb, vd, cov, math.fsum(y * y for y in residual) / (n - 1))


def pool_moments(blocks):
    counts = blocks['count'].to_numpy(dtype=int)
    n = int(counts.sum())
    if (counts < 2).any() or n < 2:
        raise ValueError('Invalid block counts')
    mb = math.fsum(counts * blocks.mean_stationary_scale) / n
    md = math.fsum(counts * blocks.mean_difference) / n
    bc = blocks.mean_stationary_scale.to_numpy() - mb
    dc = blocks.mean_difference.to_numpy() - md
    vb = math.fsum((counts - 1) * blocks.variance_stationary_scale + counts * bc * bc) / (n - 1)
    vd = math.fsum((counts - 1) * blocks.variance_difference + counts * dc * dc) / (n - 1)
    cov = math.fsum((counts - 1) * blocks.covariance_difference_stationary_scale + counts * bc * dc) / (n - 1)
    return summarize(n, mb, md, vb, vd, cov)


def reconstruct(data):
    import numpy as np
    import pandas as pd
    scenarios = pd.read_csv(data / 'scenarios.csv', dtype={'case': str})
    blocks = pd.read_csv(data / 'expanded_calibration_block_moments.csv', dtype={'case': str})
    grouped = blocks.groupby(['N', 'family', 'case'], sort=False)
    rows = []
    for scenario in scenarios.to_dict('records'):
        group = grouped.get_group((scenario['N'], scenario['family'], scenario['case'])).sort_values('start')
        expected = np.r_[0, np.cumsum(group['count'].to_numpy())[:-1]]
        if not np.array_equal(group.start.to_numpy(), expected):
            raise ValueError('Gap or overlap in trajectory blocks')
        summary = pool_moments(group)
        rows.append(dict(scenario, **summary,
                         saved_cost_per_input=-summary['mean_difference'] / scenario['N'],
                         N_delta_finite=scenario['N'] * scenario['delta_finite'],
                         N_delta_stationary_scale=scenario['N'] * scenario['delta_stationary_scale']))
    return pd.DataFrame(rows)


def main():
    import numpy as np
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Reconstruct paired cost estimates and confidence intervals from trajectory moments.')
    parser.add_argument('--data', type=Path, default=root / 'data/measured_calibration_gains')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = reconstruct(args.data)
    published = pd.read_csv(args.data / 'expanded_calibration_results.csv')
    fields = ['saving_estimate_pct', 'saving_standard_error_pct', 'saving_lower95_pct', 'saving_upper95_pct']
    if not np.allclose(results[fields], published[fields], rtol=1e-10, atol=1e-12):
        raise ValueError('Published estimates and reconstructed moments disagree')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    discrepancy = np.max(np.abs(results[fields].to_numpy() - published[fields].to_numpy()))
    print(f'Reconstructed {len(results)} scenarios; maximum discrepancy {discrepancy:.3g} percentage points.')


if __name__ == '__main__':
    main()

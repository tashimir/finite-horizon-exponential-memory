# Compare finite, stationary-scale and reference policies on paired disk trajectories.
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from scipy.optimize import brentq
from analytical_benchmarks import H_numeric

HERE = Path(__file__).resolve().parent
HORIZONS = (256, 1024, 4096)
R_VALUES = (0.0, 0.5, 1.0, 2.0)
TRAJECTORIES = 32768
SEED = 2026091202


def scalar_rule(N, alpha):
    epsilon = alpha - 1
    c = 2 / (2 + alpha)
    q = (14 - alpha) / 16
    H = H_numeric(2, alpha)
    stationary = 0.0 if epsilon == 0 else np.exp(np.log(q) / epsilon)
    if epsilon == 0:
        finite = np.sqrt(H / (2 * N * c * (1 - q)))
    else:
        def balance(delta):
            return N * alpha * c * (delta**epsilon - q) * delta**2 - H / 2
        finite = brentq(balance, stationary, 1.0, xtol=1e-15)
    assert 0 < finite < 1 and 0 <= stationary < 1
    return finite, stationary, H


def relative_comparison(A, B, saving=True):

    ratio = A.mean() / B.mean()
    se = (A - ratio * B).std(ddof=1) / (np.sqrt(len(A)) * B.mean())
    estimate = 1 - ratio if saving else ratio - 1
    return dict(estimate_pct=100*estimate, standard_error_pct=100*se,
                lower95_pct=100*(estimate-1.96*se), upper95_pct=100*(estimate+1.96*se))


def main():
    parser = argparse.ArgumentParser(description='Compare finite, stationary-scale and reference policies on paired disk trajectories.')
    parser.add_argument('--references', type=Path, default=HERE/'calibration_reference_parameters.csv')
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.references.open(encoding='utf-8', newline='') as stream:
        references = {(int(r['N']), float(r['r'])): r for r in csv.DictReader(stream)}
    lambda_star = 2*np.log(16/13)
    rows, arrays = [], {}
    for N in HORIZONS:
        policies, params = [], []
        for r in R_VALUES:
            alpha = 1+r*lambda_star/np.log(N)
            finite, stationary, H = scalar_rule(N, alpha)
            p = dict(N=N, r=r, alpha=alpha, H=H, delta_finite=finite,
                     delta_stationary_scale=stationary)
            indices = {}
            for name, delta in [('finite_balance', finite), ('stationary_scale', stationary)]:
                indices[name] = len(policies)
                policies.append((r, name, delta, alpha))
            reference = references.get((N, r))
            if reference is not None:
                assert abs(float(reference['alpha'])-alpha) < 1e-13
                p['delta_reference'] = float(reference['delta'])
                p['stored_reference_cost'] = float(reference['stored_objective'])
                indices['numerical_reference'] = len(policies)
                policies.append((r, 'numerical_reference', p['delta_reference'], alpha))
            p['indices'] = indices
            params.append(p)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, N]))
        def points():
            u = rng.random((TRAJECTORIES, 2))
            radius, angle = np.sqrt(u[:, 0]), 2*np.pi*u[:, 1]
            return np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
        delta = np.array([p[2] for p in policies])[:, None]
        alpha = np.array([p[3] for p in policies])[:, None]
        factor = delta**alpha + (1-delta)**alpha
        state = np.broadcast_to(points(), (len(policies), TRAJECTORIES, 2)).copy()
        costs = np.zeros((len(policies), TRAJECTORIES))
        for step in range(N):
            difference = points()[None, :, :] - state
            costs += factor*np.sum(difference*difference, axis=2)**(alpha/2)
            state += delta[:, :, None]*difference
        arrays[f'costs_N{N}'] = costs
        arrays[f'policy_r_N{N}'] = np.array([p[0] for p in policies])
        arrays[f'policy_name_N{N}'] = np.array([p[1] for p in policies])
        for p in params:
            idx = p.pop('indices')
            A, B = costs[idx['finite_balance']], costs[idx['stationary_scale']]
            row = {**p, 'trajectories': TRAJECTORIES, 'seed': SEED,
                   'mean_finite': float(A.mean()), 'mean_stationary_scale': float(B.mean())}
            for key, val in relative_comparison(A, B).items():
                row['saving_'+key] = val
            if 'numerical_reference' in idx:
                C = costs[idx['numerical_reference']]
                row['mean_reference'] = float(C.mean())
                for key, val in relative_comparison(A, C, saving=False).items():
                    row['reference_excess_'+key] = val
            rows.append(row)
            print(json.dumps(row), flush=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/'calibration_results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output/'calibration_trajectory_costs.npz', **arrays)
    design = dict(d=2, distribution='Uniform unit disk', initialization='x_0=p_0',
                  horizons=HORIZONS, r_values=R_VALUES, trajectories=TRAJECTORIES, seed=SEED,
                  alpha_rule='1+r*lambda_star(2)/log(N)', common_inputs_within_horizon=True,
                  independent_horizon_streams=True, policy_selection='Fixed before simulation',
                  uncertainty='Pointwise 95% paired delta-method intervals; no simultaneous coverage claim',
                  baseline='Analytical stationary scale, not the exact stationary optimizer',
                  reference='Stored numerical parameter, simulated again on the common trajectories',
                  unavailable_reference_horizons=[256])
    (args.output/'calibration_experiment_design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')
    print('Completed all prespecified scenarios.', flush=True)


if __name__ == '__main__':
    main()

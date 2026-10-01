# Generate experiment grids, policy parameters and reproducible validation inputs.
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq


def main():
    parser = argparse.ArgumentParser(description='Generate experiment grids, policy parameters and reproducible validation inputs.')
    parser.add_argument('--benchmarks', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmarks))
    from analytical_benchmarks import H_numeric
    import mpmath as mp
    target = args.root/'inputs'
    target.mkdir(parents=True, exist_ok=True)
    horizons = [2**k for k in range(8, 21, 2)]
    r_values = [0,.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.95,.975,1,1.025,1.05,
                1.1,1.15,1.2,1.3,1.4,1.45,1.5,1.55,1.6,1.75,2,2.25,2.5,2.75,3]
    fixed_alpha = [1.01,1.03,1.05,1.075,1.10,1.20]
    lambda_star = 2*np.log(16/13)
    rows, residuals = [], []
    for N in horizons:
        parameters = []
        cases = [('scaled_r', str(r), r, 1+r*lambda_star/np.log(N)) for r in r_values]
        cases += [('fixed_alpha', str(a), (a-1)*np.log(N)/lambda_star, a) for a in fixed_alpha]
        for case_index, (family, label, r, alpha) in enumerate(cases):
            eps = alpha-1
            H = H_numeric(2, alpha)
            c, q = 2/(2+alpha), (14-alpha)/16
            logq = np.log(q)
            stationary = 0.0 if eps == 0 else np.exp(logq/eps)
            if eps == 0:
                finite = np.sqrt(H/(2*N*c*(1-q)))
            else:
                lower = logq/eps
                def balance(v):
                    gap = eps*(v-lower)
                    if gap <= 0:
                        return -np.inf
                    return 2*v+logq+np.log(np.expm1(gap))-np.log(H/(2*N*alpha*c))
                v = brentq(balance, np.nextafter(lower, np.inf), 0.0, xtol=1e-13, rtol=1e-14)
                finite = np.exp(v)
            assert 0 < finite < 1 and 0 <= stationary < finite

            a, dd, hh = mp.mpf(float(alpha)), mp.mpf(float(finite)), mp.mpf(float(H))
            residual = abs((N*a*(2/(2+a))*(dd**(a-1)-(14-a)/16)*dd**2-hh/2)/(hh/2))
            residuals.append(float(residual))
            rows.append(dict(N=N, family=family, case=label, case_index=case_index,
                r=float(r), alpha=float(alpha), H=H, delta_finite=float(finite),
                delta_stationary_scale=float(stationary), finite_column=2*case_index,
                stationary_column=2*case_index+1))
            parameters.extend([(alpha, finite), (alpha, stationary)])
        (target/f'policies_N{N}.tsv').write_text(''.join(f'{a:.17g}\t{d:.17g}\n' for a,d in parameters), encoding='utf-8', newline='\n')
    assert max(residuals) < 1e-8, max(residuals)
    with (target/'scenarios.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    mask = (1<<64)-1
    def mix(z):
        z = (z + 0x9e3779b97f4a7c15) & mask
        z = ((z ^ (z >> 30))*0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27))*0x94d049bb133111eb) & mask
        return z ^ (z >> 31)
    master, trajectories = 2026091203, 32768
    all_seeds = [mix((mix((mix(master)+N)&mask)+i)&mask) for N in horizons for i in range(trajectories)]
    assert len(all_seeds) == len(set(all_seeds))
    design = dict(d=2, distribution='Uniform unit disk', initialization='x_0=p_0',
        horizons=horizons, r_values=r_values, fixed_alpha_values=fixed_alpha,
        scenarios=len(rows), policies_per_horizon=74, trajectories=trajectories,
        master_seed=master, rng='C++17 std::mt19937_64 with top-53-bit uniforms',
        trajectory_seed='mix64(mix64(mix64(master_seed)+N)+trajectory_index); uint64 arithmetic',
        independent_horizon_streams=True, common_inputs_across_policies=True,
        seed_collision_check='All 229376 trajectory seeds are distinct',
        seed_vector_sha256=hashlib.sha256(np.asarray(all_seeds, dtype='<u8').tobytes()).hexdigest(),
        policy_selection='Fixed before simulation; no fitting on evaluation samples',
        baseline='Analytical stationary scale, without an exact stationary optimality claim',
        fixed_alpha_interpretation='Finite-sample sensitivity; no extension of the joint-window theorem',
        uncertainty='Pointwise 95% paired delta-method intervals over independent complete trajectories',
        accumulation='Compensated summation of costs; float64',
        raw_format='Little-endian float64, row-major (trajectory, policy), columns in scenarios.csv',
        primary_measure='100*(1-mean_cost_finite/mean_cost_stationary_scale)',
        additional_measures=['Absolute saved cost per input', 'N*delta_stationary_scale', 'N*delta_finite'],
        no_simultaneous_coverage_claim=True, balance_max_relative_residual=max(residuals))
    (target/'design.json').write_text(json.dumps(design, indent=2)+'\n', encoding='utf-8', newline='\n')

    rng = np.random.default_rng(3019202609)
    u = rng.random((19, 258, 2))
    radius, angle = np.sqrt(u[:,:,0]), 2*np.pi*u[:,:,1]
    points = np.stack((radius*np.cos(angle), radius*np.sin(angle)), axis=-1)
    points.astype('<f8').tofile(target/'validation_points.bin')
    validation_policies = [(1,0),(1,1),(1,.03),(1.1,1e-10),(1.1,.01),(1.2,.07),(2,.2),(2,1)]
    (target/'validation_policies.tsv').write_text(''.join(f'{a:.17g}\t{d:.17g}\n' for a,d in validation_policies), encoding='utf-8', newline='\n')
    expected = np.zeros((len(points),len(validation_policies)))
    for j,(a,d) in enumerate(validation_policies):
        state = points[:,0,:].copy()
        increments=[]
        for step in range(1,258):
            previous=state.copy(); incoming=points[:,step,:]
            state = previous+d*(incoming-previous)
            edge1=np.linalg.norm(state-previous,axis=1)**a
            edge2=np.linalg.norm(incoming-state,axis=1)**a
            increments.append(edge1+edge2)
        expected[:,j]=np.sum(increments,axis=0)
    expected.astype('<f8').tofile(target/'validation_expected.bin')
    print(json.dumps(dict(scenarios=len(rows), policies=74, max_relative_balance_residual=max(residuals), seed_check='PASS')))


if __name__ == '__main__':
    main()

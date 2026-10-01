# Compare finite-horizon memory calibration with analytical and numerical stationary choices using disk moments.
from pathlib import Path
from functools import lru_cache
import argparse
import csv
import json
import math
import os
import platform
import time
import mpmath as mp
import numpy as np
from scipy.optimize import brentq, minimize_scalar


REAL = np.longdouble
LAMBDA_STAR = 2 * math.log(16 / 13)


@lru_cache(maxsize=256)
def triangular_factors(order):
    rows, cols = np.tril_indices(order + 1)
    factors = np.array([math.comb(int(k), int(j))**2 / REAL(k-j+1)
                        for k, j in zip(rows, cols)], dtype=REAL)
    return rows, cols, factors


@lru_cache(maxsize=8192)
def potential(alpha, order):
    a = REAL(alpha)
    values = np.empty(order + 1, dtype=REAL)
    values[0] = 2 / (2 + a)
    for k in range(order):
        values[k+1] = values[k] * (k-a/2) * (k-1-a/2) / (k+1)**2
    with mp.workdps(65):
        am = mp.mpf(float(alpha))
        term = 2 / (2 + am)
        total = term
        for k in range(order):
            term *= (k-am/2) * (k-1-am/2) / (k+1)**2
            total += term
        endpoint = 2*mp.gamma(2+am)/((2+am)*mp.gamma(1+am/2)*mp.gamma(2+am/2))
        tail = REAL(str(max(mp.mpf(0), total-endpoint)))
    return values, tail


def transition_increment(delta, order):
    delta = REAL(delta)
    if not 0 <= delta <= 1:
        raise ValueError('The update fraction must lie in [0,1].')
    rows, cols, factors = triangular_factors(order)
    matrix = np.zeros((order+1, order+1), dtype=REAL)
    if delta == 1:
        matrix[:, 0] = 1 / np.arange(1, order+2, dtype=REAL)
        matrix[np.diag_indices(order+1)] -= 1
        return matrix
    logarithm = np.log1p(-delta)
    lower = rows > cols
    rr, cc = rows[lower], cols[lower]
    matrix[rr, cc] = factors[lower] * np.exp(2*cc*logarithm) * delta**(2*(rr-cc))
    matrix[np.diag_indices(order+1)] = np.expm1(2*np.arange(order+1, dtype=REAL)*logarithm)
    return matrix


def stationary_moments(delta, order, derivative=False):
    values = np.zeros(order+1, dtype=REAL)
    slopes = np.zeros(order+1, dtype=REAL)
    values[0] = 1
    if delta == 0:
        return (values, slopes) if derivative else values
    delta = REAL(delta)
    if delta == 1:
        if derivative:
            raise ValueError('The derivative is evaluated only in the open interval.')
        return 1/np.arange(1,order+2,dtype=REAL)
    logarithm = np.log1p(-delta)
    rows, cols, factors = triangular_factors(order)
    lower = rows > cols
    rr, cc = rows[lower], cols[lower]
    denominator = -np.expm1(2*np.arange(order+1,dtype=REAL)*logarithm)/delta
    weights = np.zeros((order+1,order+1),dtype=REAL)
    weights[rr,cc] = factors[lower]*np.exp(2*cc*logarithm)*delta**(2*(rr-cc)-1)/denominator[rr]
    for k in range(1, order+1):
        values[k] = np.dot(weights[k,:k],values[:k])
        if derivative:
            indices = np.arange(k,dtype=REAL)
            logarithmic_denominator_slope = 2*k*np.exp((2*k-1)*logarithm)/denominator[k]
            slope_factor = 2*(k-indices)-2*indices*delta/(1-delta)-logarithmic_denominator_slope
            slopes[k] = np.dot(weights[k,:k],slopes[:k]+slope_factor*values[:k])
    return (values,slopes) if derivative else values


def finite_moments(delta, horizon, order):
    state = 1 / np.arange(1, order+2, dtype=REAL)
    if delta == 0:
        return state
    increment = transition_increment(delta, order)
    average = np.eye(order+1, dtype=REAL)
    total = np.zeros(order+1, dtype=REAL)
    remaining, length = int(horizon), 1
    while remaining:
        if remaining & 1:
            total += length * (average @ state)
            state += increment @ state
        remaining >>= 1
        if remaining:
            average += (increment @ average) / 2
            increment = 2*increment + increment @ increment
            length *= 2
    result = total / horizon
    result[0] = 1
    tolerance = 10000 * np.finfo(REAL).eps
    if np.min(result) < -tolerance or np.max(np.diff(result)) > tolerance:
        raise ArithmeticError('Radial moments violate positivity or monotonicity.')
    return result


def mean_excess(alpha, delta, horizon=None, order=64):
    coefficients, tail = potential(float(alpha), order)
    delta = REAL(delta)
    moments = (stationary_moments(delta, order+1) if horizon is None else
               finite_moments(delta, horizon, order+1))
    hm1 = (REAL(0) if delta in (0, 1) else
           np.expm1(REAL(alpha)*np.log1p(-delta)) + delta**REAL(alpha))
    excess = coefficients[0]*hm1 + (1+hm1)*np.dot(coefficients[1:], moments[1:order+1])
    bound = (1+hm1)*tail*moments[order+1]
    return excess, max(REAL(0), bound)


@lru_cache(maxsize=8192)
def stationary_minimizer(alpha, order=64):
    if alpha == 1:
        return REAL(0)
    scale_log = math.log((14-alpha)/16)/(alpha-1)
    lower = max(-740.0, 2*scale_log-8)
    grid = np.unique(np.r_[np.linspace(lower, -1e-8, 100),
                           np.linspace(max(lower, scale_log-3), min(-1e-8, scale_log+3), 45)])
    def objective(location):
        return float(mean_excess(alpha, np.exp(REAL(location)), order=order)[0])
    def derivative(location):
        delta = np.exp(REAL(location))
        moments, slopes = stationary_moments(delta,order,derivative=True)
        coefficients, _ = potential(alpha,order)
        p = np.dot(coefficients,moments)
        h = np.exp(REAL(alpha)*np.log1p(-delta))+delta**REAL(alpha)
        dh = REAL(alpha)*(delta**REAL(alpha-1)-np.exp(REAL(alpha-1)*np.log1p(-delta)))
        return float(dh*p+h*np.dot(coefficients,slopes/delta))
    values = np.array([objective(x) for x in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for i in range(1, len(grid)-1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            answer = minimize_scalar(objective, bounds=(grid[i-1], grid[i+1]),
                                     method='bounded', options={'xatol': 2e-13})
            left, right = float(grid[i-1]), float(grid[i+1])
            if derivative(left) < 0 < derivative(right):
                location = brentq(derivative,left,right,xtol=5e-14,rtol=1e-14)
                candidates.append((objective(location),location))
            else:
                candidates.append((answer.fun, answer.x))
    value, location = min(candidates)
    return REAL(0) if value >= 0 else np.exp(REAL(location))


@lru_cache(maxsize=4096)
def initialization_constant(alpha, order=512):
    coefficients, tail = potential(float(alpha), order)
    k = np.arange(1, order+1, dtype=REAL)
    return np.sum(coefficients[1:]/(k*(k+1))), tail/((order+1)*(order+2))


def policy_parameters(alpha, horizon):
    c, q = 2/(2+alpha), (14-alpha)/16
    H, H_bound = initialization_constant(float(alpha))
    epsilon = alpha-1
    if epsilon == 0:
        return REAL(np.sqrt(H/(2*horizon*c*(1-q)))), REAL(0), H, H_bound
    lower = math.log(q)/epsilon
    target = np.log(H/(2*horizon*alpha*c))
    def equation(location):
        difference = REAL(epsilon)*(REAL(location)-REAL(lower))
        if difference <= 0:
            return -math.inf
        return float(2*REAL(location)+math.log(q)+np.log(np.expm1(difference))-target)
    location = brentq(equation, np.nextafter(lower, math.inf), 0.0, xtol=5e-14, rtol=1e-14)
    return np.exp(REAL(location)), np.exp(REAL(lower)), H, H_bound


def evaluate(alpha, horizon, order):
    finite, scale, H, H_bound = policy_parameters(alpha, horizon)
    stationary = stationary_minimizer(float(alpha), order)
    ef, bf = mean_excess(alpha, finite, horizon, order)
    es, bs = mean_excess(alpha, stationary, horizon, order)
    ea, ba = mean_excess(alpha, scale, horizon, order)
    c = REAL(2)/(2+REAL(alpha))
    gs, ga = 100*(es-ef)/(c+es), 100*(ea-ef)/(c+ea)
    gs_bound = 100*((bf+bs)/(c+es-bs)+abs(es-ef)*bs/((c+es)*(c+es-bs)))
    ga_bound = 100*((bf+ba)/(c+ea-ba)+abs(ea-ef)*ba/((c+ea)*(c+ea-ba)))
    return dict(order=order, delta_finite=finite, delta_stationary_scale=scale,
                delta_stationary=stationary, H=H, H_truncation_bound=H_bound,
                excess_finite=ef, excess_stationary=es, excess_scale=ea,
                saving_vs_stationary_pct=gs, saving_vs_scale_pct=ga,
                stationary_gain_truncation_bound_pct=gs_bound,
                scale_gain_truncation_bound_pct=ga_bound)


def scenario(alpha, horizon, orders=(40, 64, 96)):
    first = evaluate(alpha, horizon, orders[0])
    previous = first
    for order in orders[1:]:
        selected = evaluate(alpha, horizon, order)
        errors = {}
        for name, tail in [('stationary', 'stationary'), ('scale', 'scale')]:
            gain = selected[f'saving_vs_{name}_pct']
            change = abs(gain-previous[f'saving_vs_{name}_pct'])
            errors[name] = max(change, selected[f'{tail}_gain_truncation_bound_pct'])
        previous = selected
        if all(err <= min(REAL('1e-7'), abs(selected[f'saving_vs_{name}_pct'])/10)
               for name, err in errors.items()):
            break
    for name, err in errors.items():
        selected[f'{name}_gain_error_estimate_pct'] = err
        selected[f'{name}_sign_resolved'] = bool(err < abs(selected[f'saving_vs_{name}_pct'])/10)
    selected['gain_error_scope'] = 'series tail and order change; rounding and minimizer search are not enclosed'
    return selected


def grid(pilot=False):
    if pilot:
        for power in [8, 20, 32, 40]:
            for r in [.1, .5, 1, 1.5, 3]:
                yield dict(N=2**power, log2_N=power, family='joint_window', r=r,
                           alpha=1+r*LAMBDA_STAR/math.log(2**power))
        for power in [8, 40]:
            for alpha in [1.2, 1.5, 1.8]:
                yield dict(N=2**power, log2_N=power, family='fixed_power',
                           r=(alpha-1)*math.log(2**power)/LAMBDA_STAR, alpha=alpha)
    else:
        r_values = sorted(set(np.round(np.r_[np.arange(0, 3.001, .05), np.arange(1.20, 1.701, .01)], 5)))
        for power in range(8, 41, 2):
            for r in r_values:
                yield dict(N=2**power, log2_N=power, family='joint_window', r=float(r),
                           alpha=1+float(r)*LAMBDA_STAR/math.log(2**power))
            for alpha in [1.01, 1.02, 1.03, 1.05, 1.075, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]:
                yield dict(N=2**power, log2_N=power, family='fixed_power',
                           r=(alpha-1)*math.log(2**power)/LAMBDA_STAR, alpha=alpha)


def serialized(value):
    if isinstance(value, np.floating):
        return np.format_float_scientific(value, precision=20, unique=False)
    return value


def write_csv(path, rows):
    temporary = path.with_suffix('.partial')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--orders', nargs='+', type=int, default=[40, 64, 96])
    args = parser.parse_args()
    if len(args.orders) < 2:
        parser.error('At least two approximation orders are required.')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'stationary_calibration_comparison.csv'
    rows = []
    if target.exists():
        with target.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
    cases = list(grid(args.pilot))
    keys = {(int(row['N']), row['family'], float(row['alpha'])) for row in rows}
    start = time.monotonic()
    for case in cases:
        if (case['N'], case['family'], case['alpha']) in keys:
            continue
        values = scenario(case['alpha'], case['N'], tuple(args.orders))
        row = {key: serialized(value) for key, value in (case | values).items()}
        rows.append(row)
        write_csv(target, rows)
        print(json.dumps({'completed': len(rows), 'total': len(cases), 'elapsed_seconds': time.monotonic()-start,
                          'N': case['N'], 'family': case['family'], 'r': case['r']}), flush=True)
    design = dict(distribution='Independent uniform unit-disk inputs', initialization='x_0=p_0',
                  scenarios=len(cases), orders=args.orders, pilot=args.pilot,
                  method='Radial moments with transition increments and binary block averaging',
                  mean_cost_difference='Potential baseline subtracted before comparing costs',
                  numerical_error='Series tail and order comparison; no certified floating-point enclosure',
                  python=platform.python_version(), numpy=np.__version__,
                  longdouble_precision=int(np.finfo(REAL).precision))
    (args.output/'stationary_calibration_comparison_design.json').write_text(
        json.dumps(design, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()

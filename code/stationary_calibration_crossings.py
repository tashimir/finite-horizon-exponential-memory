# Refine sampled gain-zero crossings for the finite-horizon stationary calibration comparison.
from pathlib import Path
import argparse
import csv
import math
import numpy as np
from stationary_calibration_comparison import LAMBDA_STAR, scenario, serialized, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=float, default=1e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.data.open(encoding='utf-8', newline='') as stream:
        data = list(csv.DictReader(stream))
    rows, evaluations = [], []
    for horizon in sorted({int(row['N']) for row in data}):
        path = sorted([row for row in data if int(row['N']) == horizon and row['family'] == 'joint_window'],
                      key=lambda item: float(item['r']))
        for comparator in ['scale', 'stationary']:
            name = f'saving_vs_{comparator}_pct'
            error_name = f'{comparator}_gain_error_estimate_pct'
            crossings = [(a,b) for a,b in zip(path,path[1:]) if float(a[name]) > 0 and float(b[name]) < 0]
            if len(crossings) != 1:
                raise ValueError(f'Expected one sampled sign change, found {len(crossings)} at N={horizon}.')
            lower, upper = (dict(row) for row in crossings[0])
            iterations = 0
            while float(upper['r'])-float(lower['r']) > args.width:
                midpoint = (float(lower['r'])+float(upper['r']))/2
                alpha = 1+midpoint*LAMBDA_STAR/math.log(horizon)
                result = scenario(alpha,horizon,orders=(64,96,128))
                value = {'N': horizon, 'log2_N': int(math.log2(horizon)), 'family':'crossing_refinement',
                         'r': midpoint, 'alpha':alpha} | result
                value = {key: serialized(item) for key,item in value.items()}
                evaluations.append(value)
                error, gain = float(value[error_name]), float(value[name])
                if error >= abs(gain)/10:
                    break
                if gain > 0:
                    lower = value
                else:
                    upper = value
                iterations += 1
            row = dict(N=horizon, log2_N=int(math.log2(horizon)), comparator=comparator,
                       r_lower=float(lower['r']), r_upper=float(upper['r']),
                       gain_lower_pct=lower[name], gain_upper_pct=upper[name],
                       lower_error_pct=lower[error_name], upper_error_pct=upper[error_name],
                       bisection_steps=iterations,
                       numerical_scope='series/order checks; floating-point and search errors assessed separately')
            if not (float(row['gain_lower_pct']) > 10*float(row['lower_error_pct']) and
                    -float(row['gain_upper_pct']) > 10*float(row['upper_error_pct'])):
                raise ArithmeticError('The sampled crossing endpoints have unresolved signs.')
            rows.append(row)
            write_csv(args.output/'stationary_calibration_crossings.csv',rows)
            if evaluations:
                write_csv(args.output/'stationary_calibration_crossing_evaluations.csv',evaluations)
            print(f"N={horizon} comparator={comparator} r=[{row['r_lower']:.8f},{row['r_upper']:.8f}]",flush=True)


if __name__ == '__main__':
    main()

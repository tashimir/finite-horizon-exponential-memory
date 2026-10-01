# Regenerate the expanded paired-trajectory experiment and its summary statistics.
import argparse
from array import array
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from calibration_statistics import trajectory_moments


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Regenerate the expanded paired-trajectory experiment and its summary statistics.')
    parser.add_argument('--data', type=Path, default=root / 'data/measured_calibration_gains')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--horizons', type=int, nargs='+')
    parser.add_argument('--trajectories', type=int, default=32768)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    if min(args.trajectories, args.block_size) < 2 or args.threads < 1:
        parser.error('Invalid trajectory, block, or thread count')
    if args.trajectories % args.block_size == 1:
        parser.error('The final block must contain at least two trajectories')
    design = json.loads((args.data / 'expanded_calibration_design.json').read_text())
    horizons = args.horizons or design['horizons']
    if len(set(horizons)) != len(horizons) or not set(horizons).issubset(design['horizons']):
        parser.error('Horizons must be distinct members of the published design')
    if args.trajectories > design['trajectories']:
        parser.error('Use at most the published number of trajectories')
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('disk_calibration.cpp')
    binary = args.output.resolve() / 'disk_calibration'
    flags = ['-std=c++17', '-O3', '-fopenmp', '-ffp-contract=off']
    subprocess.run([args.compiler, *flags, str(source), '-o', str(binary)], check=True)
    with (args.data / 'scenarios.csv').open(newline='') as stream:
        scenarios = list(csv.DictReader(stream))
    for case in scenarios:
        for key in ['N', 'case_index', 'finite_column', 'stationary_column']:
            case[key] = int(case[key])
        for key in ['r', 'alpha', 'H', 'delta_finite', 'delta_stationary_scale']:
            case[key] = float(case[key])
    rows, block_rows = [], []
    for n in horizons:
        folder = args.output / f'N{n}'
        folder.mkdir(exist_ok=True)
        parameters = args.data / f'policies_N{n}.tsv'
        policy_count = len(parameters.read_text().splitlines())
        costs = array('d')
        for start in range(0, args.trajectories, args.block_size):
            count = min(args.block_size, args.trajectories - start)
            payload = folder / f'costs_{start}_{count}.bin'
            receipt = payload.with_suffix('.json')
            identity = dict(N=n, start=start, count=count, seed=design['master_seed'],
                            kernel_sha256=digest(source), parameters_sha256=digest(parameters),
                            binary_sha256=digest(binary), compiler_flags=flags)
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if any(saved[k] != v for k, v in identity.items()) or digest(payload) != saved['sha256']:
                    raise ValueError('Stored trajectory block does not match the requested computation')
            else:
                temporary = payload.with_suffix('.tmp')
                subprocess.run([str(binary), '--N', str(n), '--start', str(start), '--count', str(count),
                                '--seed', str(design['master_seed']), '--threads', str(args.threads),
                                '--parameters', str(parameters), '--output', str(temporary)], check=True)
                if temporary.stat().st_size != count * policy_count * 8:
                    raise ValueError('Unexpected trajectory block dimensions')
                temporary.replace(payload)
                receipt.write_text(json.dumps(dict(identity, sha256=digest(payload)), indent=2) + '\n')
            chunk = array('d')
            chunk.frombytes(payload.read_bytes())
            if not all(math.isfinite(value) and value >= 0 for value in chunk):
                raise ValueError('Invalid trajectory costs')
            costs.extend(chunk)
            for case in (case for case in scenarios if case['N'] == n):
                summary = trajectory_moments(chunk[case['finite_column']::policy_count],
                                             chunk[case['stationary_column']::policy_count])
                block_rows.append(dict(N=n, family=case['family'], case=case['case'],
                                       start=start, count=count, **summary))
        for case in (case for case in scenarios if case['N'] == n):
            summary = trajectory_moments(costs[case['finite_column']::policy_count],
                                         costs[case['stationary_column']::policy_count])
            rows.append(dict(case, **summary, saved_cost_per_input=-summary['mean_difference'] / n,
                             N_delta_finite=n * case['delta_finite'],
                             N_delta_stationary_scale=n * case['delta_stationary_scale']))
        print(f'Computed N={n}, {args.trajectories} paired trajectories.', flush=True)
    for name, table in [('expanded_calibration_results.csv', rows),
                        ('expanded_calibration_block_moments.csv', block_rows)]:
        with (args.output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


if __name__ == '__main__':
    main()

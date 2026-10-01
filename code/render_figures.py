# Reproduce the three numerical figures from the distributed datasets.
from pathlib import Path
import argparse
import subprocess
import sys
import pandas as pd
import finite_horizon_plot as phase
import local_objective_plot as profile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description='Reproduce the three numerical figures from the distributed datasets.')
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced_figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.concat([pd.read_csv(p) for p in sorted(
        (ROOT/'data/finite_horizon_phase_transitions').glob('convergence_phase_regimes_shard_*.csv'))], ignore_index=True)
    data = data.sort_values(['lambda_ratio','log2N'])
    phase.validate_data(data)
    phase.make_figure(data, args.output)
    raw = pd.concat([pd.read_csv(p) for p in sorted(
        (ROOT/'data/local_objective_geometry').glob('object*profiles_shard_*.csv'))], ignore_index=True)
    raw = raw.rename(columns={'evaluation_resolution':'resolution'})
    enriched = pd.concat([profile.enrich_group(g) for _,g in
        raw.groupby(['resolution','lambda_ratio','log2N'])], ignore_index=True)
    norms = pd.read_csv(ROOT/'data/local_objective_geometry/local_objective_geometry_C2_errors.csv')
    profile.make_figure(enriched, norms, args.output)
    subprocess.run([sys.executable, str(ROOT/'code/plot_calibration_gains.py'),
        '--data', str(ROOT/'data/measured_calibration_gains/expanded_calibration_results.csv'),
        '--output', str(args.output)], check=True)
    print('Rendered the three numerical figures.')

if __name__ == '__main__':
    main()

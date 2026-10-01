# Estimate retained-edge costs at three fixed parameters using independent trajectories.
from pathlib import Path
import csv
import json
import numpy as np

HERE = Path(__file__).resolve().parent

def main():
    with (HERE / 'monte_carlo_cases.csv').open(encoding='utf-8', newline='') as f:
        cases = list(csv.DictReader(f))
    paths, seed, N = 32768, 20260912, 1024
    rng = np.random.default_rng(seed)
    gamma = np.array([float(c['gamma']) for c in cases])[:, None]
    delta = 1 - gamma
    alpha = np.array([float(c['alpha']) for c in cases])[:, None]
    h = gamma**alpha + delta**alpha
    def points():
        u = rng.random((paths, 2))
        radius, theta = np.sqrt(u[:, 0]), 2*np.pi*u[:, 1]
        return np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))
    state = np.broadcast_to(points(), (len(cases), paths, 2)).copy()
    cost = np.zeros((len(cases), paths))
    for _ in range(N):
        difference = points()[None, :, :] - state
        cost += h * np.sum(difference*difference, axis=2)**(alpha/2)
        state += delta[:, :, None] * difference
    rows = []
    for j, case in enumerate(cases):
        mean = float(cost[j].mean())
        se = float(cost[j].std(ddof=1)/np.sqrt(paths))
        stored = float(case['stored_objective'])
        rows.append(dict(r=float(case['r']), N=N, trajectories=paths, seed=seed,
                         mean=mean, standard_error=se, stored_objective=stored,
                         standardized_difference=(mean-stored)/se))
    result = dict(scope='Costs at fixed parameters; no claim of certified minimization.', results=rows)
    (HERE/'finite_cost_monte_carlo.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

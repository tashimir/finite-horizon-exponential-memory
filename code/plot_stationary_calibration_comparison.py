# Plot comparator-dependent gain crossings and fixed-power calibration losses.
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

NAME = 'stationary_calibration_comparison'
NAVY = '#004F7A'
BLUE = '#007D9A'
ORANGE = '#A8470B'
PURPLE = '#795386'
COMPARATORS = {
    'scale': 'scale',
    'analytical': 'scale',
    'analytical_scale': 'scale',
    'stationary_scale': 'scale',
    'analytical_stationary_scale': 'scale',
    'stationary': 'stationary',
    'numerical': 'stationary',
    'numerical_stationary': 'stationary',
    'numerical_stationary_choice': 'stationary',
}


def configure():
    if shutil.which('latex') is None:
        raise RuntimeError('A LaTeX executable and the lmodern package are required.')
    mpl.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{lmodern}',
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 9.0,
        'axes.labelsize': 9.0,
        'axes.titlesize': 9.0,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'legend.fontsize': 8.3,
        'lines.linewidth': 1.55,
        'lines.markersize': 3.0,
        'axes.linewidth': .7,
        'xtick.major.width': .7,
        'ytick.major.width': .7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.bbox': None,
        'savefig.pad_inches': 0.0,
    })


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing columns: {", ".join(missing)}.')
    values = frame[list(columns)].select_dtypes(include='number').to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f'{label} contains nonfinite numerical values.')


def validate(data, brackets):
    require_columns(data, ['N', 'family', 'alpha', 'saving_vs_stationary_pct',
                           'saving_vs_scale_pct'], 'Cost data')
    require_columns(brackets, ['N', 'log2_N', 'comparator', 'r_lower', 'r_upper',
                               'gain_lower_pct', 'gain_upper_pct',
                               'lower_error_pct', 'upper_error_pct'], 'Crossing data')
    if len(data) == 0 or len(brackets) == 0:
        raise ValueError('Both input tables must contain observations.')
    if (data.N <= 0).any() or (brackets.N <= 0).any():
        raise ValueError('Every horizon must be positive.')
    if not np.allclose(brackets.log2_N, np.log2(brackets.N), rtol=0, atol=1e-10):
        raise ValueError('The crossing coordinates do not agree with log2(N).')
    canonical = brackets.comparator.astype(str).str.strip().str.lower().map(COMPARATORS)
    if canonical.isna().any():
        unknown = brackets.loc[canonical.isna(), 'comparator'].unique().tolist()
        raise ValueError(f'Unknown comparator values: {unknown}.')
    brackets = brackets.copy()
    brackets['comparator'] = canonical
    if brackets[['N', 'comparator']].duplicated().any():
        raise ValueError('Each comparator must have one crossing interval per horizon.')
    if not brackets.groupby('N').comparator.nunique().eq(2).all():
        raise ValueError('Both stationary comparators are required at every horizon.')
    if not (brackets.r_lower < brackets.r_upper).all():
        raise ValueError('Every crossing interval must have positive width.')
    if (brackets[['lower_error_pct', 'upper_error_pct']] < 0).any().any():
        raise ValueError('Numerical error estimates must be nonnegative.')
    resolved = ((brackets.gain_lower_pct > brackets.lower_error_pct) &
                (brackets.gain_upper_pct < -brackets.upper_error_pct))
    if not resolved.all():
        raise ValueError('Crossing endpoints must have resolved opposite signs under the supplied error estimates.')
    horizons = sorted(brackets.N.unique())
    if horizons != sorted(data.N.unique()):
        raise ValueError('The cost and crossing tables must cover the same horizons.')
    fixed = data[data.family.isin(['fixed_power', 'fixed_alpha'])].copy()
    for alpha in [1.20, 1.50]:
        subset = fixed[np.isclose(fixed.alpha, alpha, rtol=0, atol=1e-12)]
        if subset.N.duplicated().any() or sorted(subset.N.unique()) != horizons:
            raise ValueError(f'Fixed alpha={alpha:.2f} needs one observation at every horizon.')
    return fixed, brackets


def style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.set_axisbelow(True)
    axis.tick_params(direction='out', length=2.8, width=.7, pad=2)
    axis.grid(axis='y', color='#dfdfdf', linewidth=.45)


def horizon_ticks(x):
    first, last = int(np.ceil(min(x))), int(np.floor(max(x)))
    step = 8 if last - first >= 24 else 4 if last - first >= 12 else 2
    ticks = list(range(first, last + 1, step))
    if ticks[-1] != last:
        ticks.append(last)
    return ticks


def crossings(axis, brackets):
    groups = {name: group.sort_values('N') for name, group in brackets.groupby('comparator')}
    analytic, numeric = groups['scale'], groups['stationary']
    x = analytic.log2_N.to_numpy()
    low = analytic.r_upper.to_numpy()
    high = numeric.r_lower.to_numpy()
    axis.fill_between(x, low, high, where=high > low, interpolate=True,
                      facecolor='#DBCBDD', alpha=.75, edgecolor='none', zorder=1)
    for name, color, linestyle, marker in [
            ('stationary', NAVY, '-', 'o'), ('scale', ORANGE, '--', 's')]:
        group = groups[name]
        midpoint = (group.r_lower + group.r_upper).to_numpy() / 2
        axis.fill_between(x, group.r_lower, group.r_upper, color=color,
                          alpha=.18, linewidth=0, zorder=2)
        axis.plot(x, midpoint, color=color, ls=linestyle, marker=marker,
                  markevery=max(1, len(x) // 8), markersize=2.8,
                  markerfacecolor='white', markeredgewidth=.75, lw=1.65, zorder=3)
    ymin = float(brackets.r_lower.min()) - .055
    ymax = float(brackets.r_upper.max()) + .095
    xmin, xmax = float(x.min()), float(x.max())
    span = xmax - xmin
    axis.set(xlim=(xmin - .5, xmax + .8), ylim=(ymin, ymax),
             xticks=horizon_ticks(x), xlabel=r'Horizon $\log_2 N$',
             ylabel=r'Scaled power $r$')
    axis.set_title('(a) Where savings change sign', loc='left', pad=6)
    middle = len(x) // 2
    numeric_y = float((numeric.r_lower.iloc[middle] + numeric.r_upper.iloc[middle]) / 2)
    analytic_y = float((analytic.r_lower.iloc[middle] + analytic.r_upper.iloc[middle]) / 2)
    axis.text(xmin + .43 * span, numeric_y + .035,
              'Numerical stationary\nchoice', color=NAVY, fontsize=8.0,
              ha='center', va='bottom', linespacing=1.08)
    axis.text(xmin + .66 * span, analytic_y - .028,
              'Analytical stationary\nscale', color=ORANGE, fontsize=8.0,
              ha='center', va='top', linespacing=1.08)
    location = min(2, len(x) - 1)
    band_y = (float(low[location]) + float(high[location])) / 2
    if high[location] > low[location]:
        axis.text(x[location], band_y, 'Different\nsigns',
                  color=PURPLE, fontsize=7.6, ha='center', va='center',
                  linespacing=1.05)
    return {
        'horizons': [int(n) for n in analytic.N],
        'maximum_crossing_width': float((brackets.r_upper - brackets.r_lower).max()),
        'minimum_crossing_width': float((brackets.r_upper - brackets.r_lower).min()),
        'maximum_comparator_separation': float(np.max(high - low)),
    }


def loss_series(fixed, alpha, comparator):
    subset = fixed[np.isclose(fixed.alpha, alpha, rtol=0, atol=1e-12)].sort_values('N')
    gain_key = f'saving_vs_{comparator}_pct'
    loss = -subset[gain_key].to_numpy()
    error_key = f'{comparator}_gain_error_estimate_pct'
    error = subset[error_key].to_numpy() if error_key in subset else np.zeros(len(subset))
    if not np.all(np.isfinite(loss)) or np.any(loss <= error):
        raise ValueError(f'Every plotted loss for alpha={alpha:.2f}, {comparator} must exceed its supplied error estimate.')
    return np.log2(subset.N.to_numpy()), loss


def fixed_losses(axis, fixed):
    series = [
        (1.50, 'stationary', NAVY, '-', '^'),
        (1.20, 'stationary', BLUE, '-', 'o'),
        (1.20, 'scale', ORANGE, '--', 's'),
    ]
    results = []
    for alpha, comparator, color, linestyle, marker in series:
        x, loss = loss_series(fixed, alpha, comparator)
        axis.plot(x, loss, color=color, ls=linestyle, lw=1.7,
                  marker=marker, markevery=max(1, len(x) // 8), markersize=2.8,
                  markerfacecolor='white', markeredgewidth=.75)
        results.append((alpha, comparator, color, x, loss))
    lowest = min(float(np.min(result[4])) for result in results)
    highest = max(float(np.max(result[4])) for result in results)
    ymin = 10.0 ** np.floor(np.log10(lowest) - .2)
    ymax = 10.0 ** np.ceil(np.log10(highest) + .35)
    axis.set_yscale('log')
    axis.set(xlim=(x.min() - .5, x.max() + .8), ylim=(ymin, ymax),
             xticks=horizon_ticks(x), xlabel=r'Horizon $\log_2 N$',
             ylabel=r'Cost increase $-S_B$ (\%)')
    exponents = list(range(int(np.ceil(np.log10(ymin))), int(np.floor(np.log10(ymax))) + 1))
    ticks = [10.0 ** k for k in exponents if k % 3 == 0]
    if ymin < 10 < ymax:
        ticks.append(10.0)
    ticks = sorted(set(ticks))
    axis.set_yticks(ticks, labels=[r'$10^{' + str(int(round(np.log10(t)))) + '}$'
                                   if t not in [1.0, 10.0] else r'$' + str(int(t)) + '$'
                                   for t in ticks])
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_title('(b) Losses at fixed powers', loc='left', pad=6)
    span = float(x.max() - x.min())
    for alpha, comparator, color, x, loss in results[:2]:
        location = len(x) // 2
        displacement = 2.3 if alpha == 1.50 else .34
        axis.text(float(x.min()) + .56 * span, float(loss[location]) * displacement,
                  rf'Numerical, $\alpha={alpha:.2f}$',
                  color=color, fontsize=7.9, ha='center', va='center')
    alpha, comparator, color, x, loss = results[2]
    location = len(x) // 2
    xposition = float(x.min()) + .09 * span
    yposition = float(loss[location]) * .025
    axis.annotate('Analytical scale,\n' + r'$\alpha=1.20$', xy=(float(x[location]), float(loss[location])),
                  xytext=(xposition, yposition), color=ORANGE, fontsize=7.9,
                  ha='left', va='center', linespacing=1.1,
                  arrowprops={'arrowstyle': '-', 'color': ORANGE, 'lw': .8})
    return [{'alpha': alpha, 'comparator': comparator,
             'first_horizon': int(round(2 ** x[0])), 'last_horizon': int(round(2 ** x[-1])),
             'first_increase_pct': float(loss[0]), 'last_increase_pct': float(loss[-1])}
            for alpha, comparator, color, x, loss in results]


def main():
    parser = argparse.ArgumentParser(description='Plot comparator-dependent gain crossings and fixed-power calibration losses.')
    parser.add_argument('--data', type=Path, required=True,
                        help='CSV containing finite-cost comparisons.')
    parser.add_argument('--brackets', type=Path, required=True,
                        help='CSV containing opposite-sign endpoints for each comparator and horizon.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Directory for the figure PDF and PNG.')
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    brackets = pd.read_csv(args.brackets)
    fixed, brackets = validate(data, brackets)
    configure()
    fig = plt.figure(figsize=(6.06, 3.20))
    left = fig.add_axes([.095, .165, .382, .745])
    right = fig.add_axes([.625, .165, .357, .745])
    for axis in [left, right]:
        style(axis)
    summary = crossings(left, brackets)
    summary['fixed_power_curves'] = fixed_losses(right, fixed)
    fig.canvas.draw()
    args.output.mkdir(parents=True, exist_ok=True)
    title = 'Finite-horizon calibration against stationary choices'
    fig.savefig(args.output / f'{NAME}.pdf',
                metadata={'Title': title, 'CreationDate': None, 'ModDate': None})
    fig.savefig(args.output / f'{NAME}.png', dpi=300, metadata={'Title': title})
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

# Render normalized local-objective profiles and their derivative errors.
import argparse
import json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from article_visual_style import FIGURE_WIDTH, OKABE_ITO, configure_plotting, panel_label, sha256, style_axis

def enrich_group(group):
    group = group.sort_values('z').copy()
    z = group['z'].to_numpy(dtype=float)
    values = group['G_N'].to_numpy(dtype=float)
    approximation = np.polynomial.Chebyshev.fit(z, values, deg=18, domain=[0.7, 1.4])
    group['G_smoothed'] = approximation(z)
    group['G_prime'] = approximation.deriv(1)(z)
    group['G_second'] = approximation.deriv(2)(z)
    w = float(group['w_N'].iloc[0])
    group['a_prime'] = w * (1.0 - np.power(z, -2.0)) / (2.0 * (w + 1.0))
    group['b_prime'] = np.log(z) / (w + 1.0)
    group['sum_prime'] = group['a_prime'] + group['b_prime']
    group['a_second'] = w * np.power(z, -3.0) / (w + 1.0)
    group['b_second'] = 1.0 / ((w + 1.0) * z)
    group['sum_second'] = group['a_second'] + group['b_second']
    return group

def make_figure(data, norms, output_dir):
    configure_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(FIGURE_WIDTH, 4.76))
    profile_axis = figure.add_axes([0.105, 0.585, 0.35, 0.31])
    slope_axis = figure.add_axes([0.62, 0.585, 0.35, 0.31])
    curvature_axis = figure.add_axes([0.105, 0.155, 0.35, 0.295])
    error_axis = figure.add_axes([0.62, 0.155, 0.35, 0.295])
    representative = data[(data['resolution'] == 'display') & np.isclose(data['lambda_ratio'], 1.0) & (data['log2N'] == 30)].sort_values('z')
    z = representative['z']
    profile_axis.plot(z, representative['G_N'], color=OKABE_ITO[0], marker='o', markevery=8, markerfacecolor='white', markeredgewidth=0.7, label='computed $G_N$', zorder=4)
    profile_axis.plot(z, representative['a_w'], color=OKABE_ITO[1], linestyle='-.', label='$a_w$', zorder=3)
    profile_axis.plot(z, representative['b_w'], color=OKABE_ITO[2], linestyle=':', label='$b_w$', zorder=3)
    profile_axis.plot(z, representative['profile_sum'], color='#111111', linestyle='--', linewidth=1.35, label='$a_w+b_w$', zorder=6)
    slope_axis.plot(z, representative['G_prime'], color=OKABE_ITO[0], marker='o', markevery=8, markerfacecolor='white', markeredgewidth=0.7, zorder=4)
    slope_axis.plot(z, representative['a_prime'], color=OKABE_ITO[1], linestyle='-.', zorder=3)
    slope_axis.plot(z, representative['b_prime'], color=OKABE_ITO[2], linestyle=':', zorder=3)
    slope_axis.plot(z, representative['sum_prime'], color='#111111', linestyle='--', linewidth=1.35, zorder=6)
    curvature_axis.plot(z, representative['G_second'], color=OKABE_ITO[0], marker='o', markevery=8, markerfacecolor='white', markeredgewidth=0.7, zorder=4)
    curvature_axis.plot(z, representative['a_second'], color=OKABE_ITO[1], linestyle='-.', zorder=3)
    curvature_axis.plot(z, representative['b_second'], color=OKABE_ITO[2], linestyle=':', zorder=3)
    curvature_axis.plot(z, representative['sum_second'], color='#111111', linestyle='--', linewidth=1.35, zorder=6)
    for axis in [profile_axis, slope_axis, curvature_axis]:
        axis.axvline(1.0, color='#777777', linewidth=0.85, zorder=1)
        axis.set_xlim(0.7, 1.4)
        axis.set_xlabel('multiplicative displacement $z=\\delta/\\Delta_N$')
        style_axis(axis)
    profile_axis.set_ylabel('normalized objective $G_N(z)$')
    slope_axis.set_ylabel('first derivative')
    curvature_axis.set_ylabel('second derivative')
    profile_axis.set_title('Profile, $r=1$, $N=2^{30}$', pad=4.0)
    slope_axis.set_title('Slope', pad=4.0)
    curvature_axis.set_title('Curvature', pad=4.0)
    profile_handles, profile_labels = profile_axis.get_legend_handles_labels()
    figure.legend(profile_handles, profile_labels, loc='upper center', bbox_to_anchor=(0.535, 0.985), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.0)
    line_styles = {1.0: '-', 1.1: '--', 2.0: ':'}
    ratio_markers = {1.0: 'o', 1.1: 's', 2.0: '^'}
    norm_colors = {'C0': OKABE_ITO[0], 'C1': OKABE_ITO[1], 'C2': OKABE_ITO[2]}
    for ratio in [1.0, 1.1, 2.0]:
        group = norms[np.isclose(norms['lambda_ratio'], ratio)].sort_values('log2N')
        for norm_name in ['C0', 'C1', 'C2']:
            error_axis.semilogy(group['log2N'], group[norm_name], color=norm_colors[norm_name], linestyle=line_styles[ratio], marker=ratio_markers[ratio], markerfacecolor='white', markeredgewidth=0.7, zorder=3)
    error_axis.set_xlim(13.3, 30.7)
    error_axis.set_xticks([14, 22, 30])
    error_axis.set_xticklabels(['$2^{14}$', '$2^{22}$', '$2^{30}$'])
    error_axis.set_xlabel('number of insertions $N$')
    error_axis.set_ylabel('maximum grid error')
    error_axis.set_title('Grid error by derivative', pad=4.0)
    style_axis(error_axis)
    handles = [Line2D([0], [0], color=norm_colors[name], linewidth=1.25, label=f'$C^{index}$') for index, name in enumerate(['C0', 'C1', 'C2'])]
    handles += [Line2D([0], [0], color='#555555', linestyle=line_styles[ratio], marker=ratio_markers[ratio], markerfacecolor='white', markeredgewidth=0.7, label=f'$r={ratio:g}$') for ratio in [1.0, 1.1, 2.0]]
    figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.535, 0.008), ncol=6, frameon=False, columnspacing=0.85, handlelength=1.6)
    for axis, label in zip([profile_axis, slope_axis, curvature_axis, error_axis], ['(a)', '(b)', '(c)', '(d)']):
        panel_label(axis, label)
    pdf_path = output_dir / 'local_objective_geometry.pdf'
    png_path = output_dir / 'local_objective_geometry_600dpi.png'
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=600)
    plt.close(figure)
    return (pdf_path, png_path)

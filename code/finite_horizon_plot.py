# Render the finite-horizon phase diagram from stored numerical arrays.
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, Ellipse
import numpy as np
import pandas as pd
import shutil
FIGURE_WIDTH = 6.06
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442']

def configure_plotting():
    if shutil.which('latex') is None:
        raise RuntimeError('A LaTeX executable is required')
    mpl.rcParams.update({'text.usetex': True, 'text.latex.preamble': '\\usepackage{lmodern}', 'font.family': 'serif', 'font.serif': ['Latin Modern Roman'], 'mathtext.fontset': 'cm', 'axes.labelsize': 9.0, 'axes.titlesize': 9.0, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'legend.fontsize': 8.3, 'lines.linewidth': 1.25, 'lines.markersize': 4.0, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None, 'savefig.pad_inches': 0.0})

def style_axis(axis):
    axis.set_axisbelow(True)
    axis.minorticks_on()
    axis.grid(True, which='minor', color='#ECECEC', linewidth=0.24, alpha=1.0)
    axis.grid(True, which='major', color='#D9D9D9', linewidth=0.45, alpha=0.72)
    axis.tick_params(direction='out', length=2.8)
    axis.tick_params(which='minor', length=0.0)

def part_label(axis, label):
    axis.text(0.0, 1.035, label, transform=axis.transAxes, fontsize=9.2, fontweight='bold', ha='left', va='bottom', clip_on=False)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

HORIZON_LINESTYLES = {16: "-", 22: (0, (4.0, 1.6)), 28: (0, (1.0, 1.2)), 32: (0, (4.0, 1.2, 1.0, 1.2))}

def validate_data(data):
    expected_ratios = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0])
    expected_exponents = np.arange(10, 33, 2)
    ratios = np.sort(data['lambda_ratio'].unique())
    exponents = np.sort(data['log2N'].unique())
    counts = data.groupby('lambda_ratio')['N'].nunique()
    checks = {'points_total': int(len(data)), 'ratio_count': int(len(ratios)), 'size_count': int(len(exponents)), 'common_size_grid': bool(counts.nunique() == 1 and int(counts.iloc[0]) == len(expected_exponents)), 'ratios_exact': bool(len(ratios) == len(expected_ratios) and np.allclose(ratios, expected_ratios, atol=1e-13)), 'sizes_exact': bool(len(exponents) == len(expected_exponents) and np.allclose(exponents, expected_exponents, atol=1e-13)), 'points_validated': int((data['accepted'] == 1).sum()), 'points_rejected': int((data['accepted'] != 1).sum()), 'maximum_location_relative_change': float(data['location_rel_change'].max()), 'maximum_value_relative_change': float(data['value_rel_change'].max()), 'maximum_derivative_scaled': float(data['derivative_scaled'].max()), 'minimum_left_neighbor_gap': float(data['neighbor_left'].min()), 'minimum_right_neighbor_gap': float(data['neighbor_right'].min())}
    passed = checks['points_total'] == len(expected_ratios) * len(expected_exponents) and checks['common_size_grid'] and checks['ratios_exact'] and checks['sizes_exact'] and (checks['points_rejected'] == 0) and (checks['maximum_location_relative_change'] < 0.0025) and (checks['maximum_value_relative_change'] < 0.001) and (checks['maximum_derivative_scaled'] < 2e-05)
    if not passed:
        raise RuntimeError(f'Phase-regime dataset failed validation: {checks}')
    return checks

def quotient_exponents(data):
    rows = []
    for ratio, group in data.groupby('lambda_ratio'):
        values = list(group.sort_values('log2N').itertuples(index=False))
        for left, right in zip(values[:-1], values[1:]):
            if not math.isclose(right.log2N - left.log2N, 2.0, abs_tol=1e-12):
                raise RuntimeError('The size-pair grid is not separated by a factor four')
            rows.append({'lambda_ratio': ratio, 'log2N_left': left.log2N, 'log2N_right': right.log2N, 'beta_quotient': -math.log(right.delta_512 / left.delta_512) / math.log(4.0)})
    return pd.DataFrame(rows)

def physical_circle_geometry(axis, center, diameter_points):
    axis.figure.canvas.draw()
    center_display = axis.transData.transform(center)
    radius_pixels = 0.5 * diameter_points * axis.figure.dpi / 72.0
    inverse = axis.transData.inverted()
    right = inverse.transform((center_display[0] + radius_pixels, center_display[1]))
    top = inverse.transform((center_display[0], center_display[1] + radius_pixels))
    radius_x = right[0] - center[0]
    radius_y = top[1] - center[1]
    return {'center': center, 'diameter_points': diameter_points, 'radius_x': radius_x, 'radius_y': radius_y}

def add_physical_circle(axis, geometry):
    circle = Ellipse(geometry['center'], width=2.0 * geometry['radius_x'], height=2.0 * geometry['radius_y'], facecolor='none', edgecolor='#444444', linewidth=0.9, zorder=9, clip_on=True)
    axis.add_patch(circle)
    return circle

def circular_inset_position(parent_axis, left, bottom, diameter_fraction):
    figure = parent_axis.figure
    figure.canvas.draw()
    parent_box = parent_axis.get_position()
    figure_width, figure_height = figure.get_size_inches()
    physical_ratio = parent_box.width * figure_width / (parent_box.height * figure_height)
    return [left, bottom, diameter_fraction, diameter_fraction * physical_ratio]

def line_artist_hash(artist):
    data = np.ascontiguousarray(np.asarray(artist.get_xydata(), dtype=np.float64))
    return hashlib.sha256(data.tobytes()).hexdigest()

def pairwise_linear_intersections(quotient, selected_exponents, x_limits):
    selected = quotient[quotient['log2N_right'].isin(selected_exponents)]
    pivot = selected.pivot(index='lambda_ratio', columns='log2N_right', values='beta_quotient').sort_index()
    x_values = pivot.index.to_numpy(dtype=float)
    intersections = []
    for first, second in itertools.combinations(selected_exponents, 2):
        difference = (pivot[first] - pivot[second]).to_numpy(dtype=float)
        for index in range(len(x_values) - 1):
            left = difference[index]
            right = difference[index + 1]
            if left * right < 0.0:
                crossing = x_values[index] - left * (x_values[index + 1] - x_values[index]) / (right - left)
                if x_limits[0] <= crossing <= x_limits[1]:
                    intersections.append({'first_log2N': int(first), 'second_log2N': int(second), 'lambda_ratio': float(crossing)})
    return intersections

def draw_vector_circular_lens(parent_axis, quotient, selected_exponents, colors, source_geometry, position, title, magnification, grid_positions, title_position=None, line_scale=1.0):
    detail_axis = parent_axis.inset_axes(position, zorder=10)
    detail_axis.patch.set_visible(False)
    clip_circle = Circle((0.5, 0.5), 0.5, transform=detail_axis.transAxes, facecolor='none', edgecolor='none')
    white_base = Circle((0.5, 0.5), 0.5, transform=detail_axis.transAxes, facecolor='white', edgecolor='none', zorder=-5)
    detail_axis.add_patch(white_base)
    layer_artists = []
    for left, right, color, alpha in [(0.0, 1.0, '#DCEAF6', 0.46), (1.0, 2.0, '#F7E2DC', 0.4)]:
        patch = detail_axis.axvspan(left, right, color=color, alpha=alpha, zorder=0)
        patch.set_clip_path(clip_circle)
    for value in grid_positions['minor_x']:
        line = detail_axis.axvline(value, color='#ECECEC', linewidth=0.15, alpha=1.0, zorder=0.3)
        line.set_clip_path(clip_circle)
    for value in grid_positions['minor_y']:
        line = detail_axis.axhline(value, color='#ECECEC', linewidth=0.15, alpha=1.0, zorder=0.3)
        line.set_clip_path(clip_circle)
    for value in grid_positions['major_x']:
        line = detail_axis.axvline(value, color='#D9D9D9', linewidth=0.27, alpha=0.72, zorder=0.4)
        line.set_clip_path(clip_circle)
    for value in grid_positions['major_y']:
        line = detail_axis.axhline(value, color='#D9D9D9', linewidth=0.27, alpha=0.72, zorder=0.4)
        line.set_clip_path(clip_circle)
    for exponent in selected_exponents:
        group = quotient[np.isclose(quotient['log2N_right'], exponent)]
        artist, = detail_axis.plot(group['lambda_ratio'], group['beta_quotient'], color=colors[exponent], linestyle=HORIZON_LINESTYLES[exponent], linewidth=0.48 * line_scale, zorder=3)
        artist.set_clip_path(clip_circle)
        layer_artists.append((f'curve_{exponent}', artist))
    ratio_grid = np.linspace(0.0, 2.0, 1001)
    beta_limit = np.where(ratio_grid <= 1.0, 0.5, 0.5 / np.maximum(ratio_grid, 1e-15))
    asymptotic, = detail_axis.plot(ratio_grid, beta_limit, color='#111111', linestyle='--', linewidth=0.68 * line_scale, zorder=8)
    asymptotic.set_clip_path(clip_circle)
    layer_artists.append(('asymptotic_reference', asymptotic))
    critical = detail_axis.axvline(1.0, color='#222222', linestyle='--', linewidth=0.52 * line_scale, zorder=7)
    critical.set_clip_path(clip_circle)
    layer_artists.append(('critical_r_1', critical))
    crossover = detail_axis.axvline(1.5, color='#666666', linestyle=':', linewidth=0.52 * line_scale, zorder=7)
    crossover.set_clip_path(clip_circle)
    layer_artists.append(('crossover_r_3_2', crossover))
    center_x, center_y = source_geometry['center']
    detail_axis.set_xlim(center_x - source_geometry['radius_x'], center_x + source_geometry['radius_x'])
    detail_axis.set_ylim(center_y - source_geometry['radius_y'], center_y + source_geometry['radius_y'])
    detail_axis.set_aspect('auto')
    if title_position is None:
        detail_axis.set_title(title, fontsize=8.3, pad=1.0)
    else:
        detail_axis.text(title_position[0], title_position[1], title, transform=detail_axis.transAxes, ha='left', va='top', fontsize=8.3, clip_on=False, zorder=30)
    detail_axis.set_xticks([])
    detail_axis.set_yticks([])
    for spine in detail_axis.spines.values():
        spine.set_visible(False)
    lens_border = Circle((0.5, 0.5), 0.5, transform=detail_axis.transAxes, facecolor='none', edgecolor='#444444', linewidth=0.72, zorder=20, clip_on=False)
    detail_axis.add_patch(lens_border)
    return (detail_axis, dict(layer_artists))

def draw_correction_circular_lens(parent_axis, supercritical, selected_exponents, colors, source_geometry, position, magnification, grid_positions):
    detail_axis = parent_axis.inset_axes(position, zorder=10)
    detail_axis.patch.set_visible(False)
    clip_circle = Circle((0.5, 0.5), 0.5, transform=detail_axis.transAxes, facecolor='none', edgecolor='none')
    detail_axis.add_patch(Circle((0.5, 0.5), 0.5, transform=detail_axis.transAxes, facecolor='white', edgecolor='none', zorder=-5))
    for value in grid_positions['minor_x']:
        line = detail_axis.axvline(value, color='#ECECEC', linewidth=0.15, alpha=1.0, zorder=0.3)
        line.set_clip_path(clip_circle)
    for value in grid_positions['minor_y']:
        line = detail_axis.axhline(value, color='#ECECEC', linewidth=0.15, alpha=1.0, zorder=0.3)
        line.set_clip_path(clip_circle)
    for value in grid_positions['major_x']:
        line = detail_axis.axvline(value, color='#D9D9D9', linewidth=0.27, alpha=0.72, zorder=0.4)
        line.set_clip_path(clip_circle)
    for value in grid_positions['major_y']:
        line = detail_axis.axhline(value, color='#D9D9D9', linewidth=0.27, alpha=0.72, zorder=0.4)
        line.set_clip_path(clip_circle)
    layer_artists = []
    for exponent in selected_exponents:
        group = supercritical[supercritical['log2N'] == exponent].sort_values('lambda_ratio')
        artist, = detail_axis.plot(group['lambda_ratio'], group['correction_balance'], color=colors[exponent], linestyle=HORIZON_LINESTYLES[exponent], linewidth=0.48, zorder=3)
        artist.set_clip_path(clip_circle)
        layer_artists.append((f'curve_{exponent}', artist))
    super_grid = np.linspace(1.001, 2.0, 800)
    reference, = detail_axis.plot(super_grid, -1.0 + 3.0 / (2.0 * super_grid), color='#111111', linestyle='--', linewidth=0.68, zorder=8)
    reference.set_clip_path(clip_circle)
    layer_artists.append(('correction_reference', reference))
    zero = detail_axis.axhline(0.0, color='#777777', linewidth=0.42, zorder=2)
    zero.set_clip_path(clip_circle)
    layer_artists.append(('zero_reference', zero))
    crossover = detail_axis.axvline(1.5, color='#666666', linestyle=':', linewidth=0.52, zorder=7)
    crossover.set_clip_path(clip_circle)
    layer_artists.append(('crossover_r_3_2', crossover))
    center_x, center_y = source_geometry['center']
    detail_axis.set_xlim(center_x - source_geometry['radius_x'], center_x + source_geometry['radius_x'])
    detail_axis.set_ylim(center_y - source_geometry['radius_y'], center_y + source_geometry['radius_y'])
    detail_axis.set_aspect('auto')
    detail_axis.set_xticks([])
    detail_axis.set_yticks([])
    for spine in detail_axis.spines.values():
        spine.set_visible(False)
    detail_axis.add_patch(Circle((0.5, 0.5), 0.5, transform=detail_axis.transAxes, facecolor='none', edgecolor='#444444', linewidth=0.72, zorder=20, clip_on=False))
    detail_axis.text(1.03, -0.13, 'near $r=3/2$', transform=detail_axis.transAxes, ha='right', va='top', fontsize=6.2, clip_on=False, zorder=30)
    return (detail_axis, dict(layer_artists))

def add_boundary_connector(figure, source_axis, source_geometry, target_axis):
    figure.canvas.draw()
    source_center = source_axis.transData.transform(source_geometry['center'])
    target_center = target_axis.transAxes.transform((0.5, 0.5))
    direction = target_center - source_center
    direction = direction / np.linalg.norm(direction)
    source_radius = 0.5 * source_geometry['diameter_points'] * figure.dpi / 72.0
    target_box = target_axis.get_window_extent()
    target_radius = 0.5 * min(target_box.width, target_box.height)
    source_boundary_display = source_center + source_radius * direction
    target_boundary_display = target_center - target_radius * direction
    source_boundary = source_axis.transData.inverted().transform(source_boundary_display)
    target_boundary = target_axis.transAxes.inverted().transform(target_boundary_display)
    connector = ConnectionPatch(xyA=source_boundary, coordsA=source_axis.transData, xyB=target_boundary, coordsB=target_axis.transAxes, color='#777777', linewidth=0.55, zorder=2, clip_on=True)
    figure.add_artist(connector)
    return connector

def make_figure(data, output_dir):
    configure_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)
    quotient = quotient_exponents(data)
    selected_exponents = [16, 22, 28, 32]
    selected = data[data['log2N'].isin(selected_exponents)].copy()
    selected['log_normalized_distance'] = np.log(selected['collapse_ratio'])
    supercritical = selected[selected['lambda_ratio'] > 1.0].copy()
    supercritical['stationary_scale'] = np.power((14.0 - supercritical['alpha']) / 16.0, 1.0 / (supercritical['alpha'] - 1.0))
    supercritical['correction_balance'] = -np.log(supercritical['N'] * np.power(supercritical['stationary_scale'], 3.0)) / np.log(supercritical['N'])
    colors = dict(zip(selected_exponents, OKABE_ITO[:len(selected_exponents)]))
    figure = plt.figure(figsize=(FIGURE_WIDTH, 4.55))
    exponent_axis = figure.add_axes([0.105, 0.585, 0.865, 0.32])
    scale_axis = figure.add_axes([0.105, 0.09, 0.35, 0.31])
    correction_axis = figure.add_axes([0.62, 0.09, 0.35, 0.31])
    exponent_axis.axvspan(0.0, 1.0, color='#DCEAF6', alpha=0.46, zorder=0)
    exponent_axis.axvspan(1.0, 2.0, color='#F7E2DC', alpha=0.4, zorder=0)
    parent_layer_artists = {}
    for exponent in selected_exponents:
        group = quotient[np.isclose(quotient['log2N_right'], exponent)]
        artist, = exponent_axis.plot(group['lambda_ratio'], group['beta_quotient'], color=colors[exponent], linestyle=HORIZON_LINESTYLES[exponent], label=f'$2^{{{exponent}}}$', zorder=3)
        parent_layer_artists[f'curve_{exponent}'] = artist
    ratio_grid = np.linspace(0.0, 2.0, 1001)
    beta_limit = np.where(ratio_grid <= 1.0, 0.5, 0.5 / np.maximum(ratio_grid, 1e-15))
    asymptotic, = exponent_axis.plot(ratio_grid, beta_limit, color='#111111', linestyle='--', linewidth=1.35, label='asymptotic reference', zorder=8)
    parent_layer_artists['asymptotic_reference'] = asymptotic
    parent_layer_artists['critical_r_1'] = exponent_axis.axvline(1.0, color='#222222', linestyle='--', linewidth=1.15, zorder=7)
    parent_layer_artists['crossover_r_3_2'] = exponent_axis.axvline(1.5, color='#666666', linestyle=':', linewidth=1.05, zorder=7)
    exponent_axis.text(0.24, 0.414, 'subcritical', ha='center', va='center', fontsize=8.3)
    exponent_axis.text(1.25, 0.505, 'leading transition', ha='center', va='center', fontsize=8.3)
    exponent_axis.text(1.75, 0.415, 'supercritical', ha='center', va='center', fontsize=8.3)
    exponent_axis.set_xlim(0.0, 2.0)
    exponent_axis.set_ylim(0.215, 0.535)
    exponent_axis.set_xticks(np.arange(0.0, 2.0001, 0.25))
    exponent_axis.set_xticklabels(['0.00', '0.25', '0.50', '0.75', '$\\mathbf{r=1}$', '1.25', '$\\mathbf{r=3/2}$', '1.75', '2.00'])
    exponent_axis.set_xlabel('scaled parameter $r=\\lambda/\\lambda^\\star(2)$')
    exponent_axis.set_ylabel('size-pair effective exponent $\\widehat\\beta_N(r)$')
    exponent_axis.legend(loc='lower center', bbox_to_anchor=(0.5, 1.09), ncol=5, frameon=False, columnspacing=0.8, handlelength=2.5)
    style_axis(exponent_axis)
    part_label(exponent_axis, '(a)')
    figure.canvas.draw()
    grid_positions = {'major_x': exponent_axis.get_xticks(minor=False), 'minor_x': exponent_axis.get_xticks(minor=True), 'major_y': exponent_axis.get_yticks(minor=False), 'minor_y': exponent_axis.get_yticks(minor=True)}
    source_specifications = [{'center': (1.0, 0.476), 'diameter_points': 18.0}, {'center': (1.53, 0.3282), 'diameter_points': 28.0}]
    source_geometries = [physical_circle_geometry(exponent_axis, specification['center'], specification['diameter_points']) for specification in source_specifications]
    first_detail_position = circular_inset_position(exponent_axis, 0.215, 0.117, 0.17)
    second_detail_position = circular_inset_position(exponent_axis, 0.515, 0.03, 0.145)
    parent_box = exponent_axis.get_position()
    lens_diameter_points = [fraction * parent_box.width * figure.get_size_inches()[0] * 72.0 for fraction in [0.17, 0.145]]
    magnifications = [diameter / geometry['diameter_points'] for diameter, geometry in zip(lens_diameter_points, source_geometries)]
    lens_line_scales = [1.18, 1.0]
    first_detail, first_lens_artists = draw_vector_circular_lens(exponent_axis, quotient, selected_exponents, colors, source_geometries[0], first_detail_position, 'near $r=1$', magnifications[0], grid_positions, line_scale=lens_line_scales[0])
    second_detail, second_lens_artists = draw_vector_circular_lens(exponent_axis, quotient, selected_exponents, colors, source_geometries[1], second_detail_position, 'near\n' + '$r=3/2$', magnifications[1], grid_positions, title_position=(0.93, 0.26))
    add_physical_circle(exponent_axis, source_geometries[0])
    add_physical_circle(exponent_axis, source_geometries[1])
    add_boundary_connector(figure, exponent_axis, source_geometries[0], first_detail)
    add_boundary_connector(figure, exponent_axis, source_geometries[1], second_detail)
    similarity_matrices = [np.eye(2) * diameter / geometry['diameter_points'] for diameter, geometry in zip(lens_diameter_points, source_geometries)]
    singular_value_ratios = [float(np.linalg.svd(matrix, compute_uv=False).max() / np.linalg.svd(matrix, compute_uv=False).min()) for matrix in similarity_matrices]
    rotations = [float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))) for matrix in similarity_matrices]
    parent_layer_hashes = {layer_id: line_artist_hash(artist) for layer_id, artist in parent_layer_artists.items()}
    lens_layer_hashes = [{layer_id: line_artist_hash(artist) for layer_id, artist in artists.items()} for artists in [first_lens_artists, second_lens_artists]]
    zoom_validation = {'construction': 'vector redraw on source-derived circular coordinate windows', 'rotation_degrees': rotations, 'reflection': False, 'anisotropy_ratio': singular_value_ratios, 'off_diagonal_terms': [[float(matrix[0, 1]), float(matrix[1, 0])] for matrix in similarity_matrices], 'raster_images_per_lens': [len(first_detail.images), len(second_detail.images)], 'base_curve_linewidth_points': 0.48, 'curve_markers': 'none', 'base_asymptotic_linewidth_points': 0.68, 'base_vertical_reference_linewidth_points': 0.52, 'source_layer_content': ['phase background', 'major and minor grid', 'four sampled curves', 'asymptotic reference', 'vertical reference'], 'lenses': [{'center': list(source_geometries[index]['center']), 'source_diameter_points': source_geometries[index]['diameter_points'], 'source_x_limits': [source_geometries[index]['center'][0] - source_geometries[index]['radius_x'], source_geometries[index]['center'][0] + source_geometries[index]['radius_x']], 'source_y_limits': [source_geometries[index]['center'][1] - source_geometries[index]['radius_y'], source_geometries[index]['center'][1] + source_geometries[index]['radius_y']], 'lens_diameter_points': lens_diameter_points[index], 'magnification': lens_diameter_points[index] / source_geometries[index]['diameter_points'], 'curve_linewidth_points': 0.48 * lens_line_scales[index], 'asymptotic_linewidth_points': 0.68 * lens_line_scales[index], 'vertical_reference_linewidth_points': 0.52 * lens_line_scales[index], 'source_artist_hashes': parent_layer_hashes, 'lens_artist_hashes': lens_layer_hashes[index], 'artist_data_hash_match': bool(parent_layer_hashes == lens_layer_hashes[index])} for index in range(2)]}
    second_x_limits = [source_geometries[1]['center'][0] - source_geometries[1]['radius_x'], source_geometries[1]['center'][0] + source_geometries[1]['radius_x']]
    second_intersections = pairwise_linear_intersections(quotient, selected_exponents, second_x_limits)
    if len(second_intersections) < 4:
        raise RuntimeError('The r=3/2 lens does not expose enough finite-size curve crossings')
    zoom_validation['lenses'][1]['visible_pairwise_intersections'] = second_intersections
    for exponent in selected_exponents:
        group = selected[selected['log2N'] == exponent].sort_values('lambda_ratio')
        left = group[group['lambda_ratio'] < 1.0]
        critical = group[np.isclose(group['lambda_ratio'], 1.0)]
        right = group[group['lambda_ratio'] > 1.0]
        for segment in [left, right]:
            scale_axis.plot(segment['lambda_ratio'], segment['log_normalized_distance'], color=colors[exponent], linestyle=HORIZON_LINESTYLES[exponent], zorder=3)
        scale_axis.plot(critical['lambda_ratio'], critical['log_normalized_distance'], color=colors[exponent], marker='o', linestyle='None', markersize=2.15, markeredgewidth=0.24, zorder=5)
    scale_axis.axhline(0.0, color='#111111', linestyle='--', linewidth=1.35, zorder=8)
    scale_axis.axvline(1.0, color='#222222', linestyle='--', linewidth=1.15, zorder=7)
    scale_axis.set_xlim(0.0, 2.0)
    scale_axis.set_xlabel('scaled parameter $r$')
    scale_axis.set_ylabel('$\\log\\{(1-\\gamma_N)/S_N(r)\\}$')
    scale_axis.set_title('Regime-specific normalization', pad=4.0)
    style_axis(scale_axis)
    part_label(scale_axis, '(b)')
    correction_parent_artists = {}
    for exponent in selected_exponents:
        group = supercritical[supercritical['log2N'] == exponent].sort_values('lambda_ratio')
        artist, = correction_axis.plot(group['lambda_ratio'], group['correction_balance'], color=colors[exponent], linestyle=HORIZON_LINESTYLES[exponent], zorder=3)
        correction_parent_artists[f'curve_{exponent}'] = artist
    super_grid = np.linspace(1.001, 2.0, 800)
    correction_reference, = correction_axis.plot(super_grid, -1.0 + 3.0 / (2.0 * super_grid), color='#111111', linestyle='--', linewidth=1.35, zorder=8)
    correction_parent_artists['correction_reference'] = correction_reference
    correction_parent_artists['zero_reference'] = correction_axis.axhline(0.0, color='#777777', linewidth=0.85, zorder=2)
    correction_parent_artists['crossover_r_3_2'] = correction_axis.axvline(1.5, color='#666666', linestyle=':', linewidth=1.05, zorder=7)
    correction_axis.text(1.19, 0.018, 'transient larger', ha='center', va='bottom', fontsize=7.2)
    correction_axis.text(1.75, -0.248, 'stationary larger', ha='center', va='center', fontsize=7.2)
    correction_axis.set_xlim(1.0, 2.0)
    correction_axis.set_ylim(-0.29, 0.53)
    correction_axis.set_xticks([1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0])
    correction_axis.set_xticklabels(['1.0', '1.2', '1.4', '$\\mathbf{1.5}$', '1.6', '1.8', '2.0'])
    correction_axis.set_xlabel('scaled parameter $r$')
    correction_axis.set_ylabel('$\\log(C_{\\rm tr}/C_{\\rm stat})/\\log N$')
    correction_axis.set_title('Subleading-correction crossover', pad=4.0)
    style_axis(correction_axis)
    part_label(correction_axis, '(c)')
    figure.canvas.draw()
    correction_grid_positions = {'major_x': correction_axis.get_xticks(minor=False), 'minor_x': correction_axis.get_xticks(minor=True), 'major_y': correction_axis.get_yticks(minor=False), 'minor_y': correction_axis.get_yticks(minor=True)}
    correction_source_geometry = physical_circle_geometry(correction_axis, (1.5, 0.0), 9.0)
    correction_detail_position = circular_inset_position(correction_axis, 0.593, 0.398, 0.314)
    correction_box = correction_axis.get_position()
    correction_lens_diameter_points = 0.314 * correction_box.width * figure.get_size_inches()[0] * 72.0
    correction_magnification = correction_lens_diameter_points / correction_source_geometry['diameter_points']
    correction_detail, correction_lens_artists = draw_correction_circular_lens(correction_axis, supercritical, selected_exponents, colors, correction_source_geometry, correction_detail_position, correction_magnification, correction_grid_positions)
    add_physical_circle(correction_axis, correction_source_geometry)
    add_boundary_connector(figure, correction_axis, correction_source_geometry, correction_detail)
    correction_parent_hashes = {layer_id: line_artist_hash(artist) for layer_id, artist in correction_parent_artists.items()}
    correction_lens_hashes = {layer_id: line_artist_hash(artist) for layer_id, artist in correction_lens_artists.items()}
    correction_similarity = np.eye(2) * correction_magnification
    zoom_validation['part_c_vector_zoom_validation'] = {'center': list(correction_source_geometry['center']), 'source_diameter_points': correction_source_geometry['diameter_points'], 'lens_diameter_points': correction_lens_diameter_points, 'magnification': correction_magnification, 'source_x_limits': [correction_source_geometry['center'][0] - correction_source_geometry['radius_x'], correction_source_geometry['center'][0] + correction_source_geometry['radius_x']], 'source_y_limits': [correction_source_geometry['center'][1] - correction_source_geometry['radius_y'], correction_source_geometry['center'][1] + correction_source_geometry['radius_y']], 'rotation_degrees': float(np.degrees(np.arctan2(correction_similarity[1, 0], correction_similarity[0, 0]))), 'reflection': False, 'anisotropy_ratio': float(np.linalg.svd(correction_similarity, compute_uv=False).max() / np.linalg.svd(correction_similarity, compute_uv=False).min()), 'raster_images': len(correction_detail.images), 'curve_linewidth_points': 0.48, 'curve_markers': 'none', 'source_artist_hashes': correction_parent_hashes, 'lens_artist_hashes': correction_lens_hashes, 'artist_data_hash_match': bool(correction_parent_hashes == correction_lens_hashes)}
    pdf_path = output_dir / 'finite_horizon_phase_transitions.pdf'
    figure.savefig(pdf_path)
    plt.close(figure)
    return (quotient, supercritical, zoom_validation, pdf_path)


def main():
    parser = argparse.ArgumentParser(description="Render finite-horizon phase transitions from the supplied numerical data.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "finite_horizon_phase_transitions")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the PDF figure.")
    args = parser.parse_args()
    files = sorted(args.data.glob("convergence_phase_regimes_shard_*.csv"))
    if not files:
        raise FileNotFoundError(f"No finite-horizon numerical data found in {args.data}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    validate_data(data)
    _, _, _, pdf_path = make_figure(data, args.output)
    print(pdf_path)


if __name__ == "__main__":
    main()

# Define the typography, colors and layout shared by the scientific figures.
import hashlib
import shutil
import matplotlib as mpl
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

def panel_label(axis, label):
    axis.text(0.0, 1.035, label, transform=axis.transAxes, fontsize=9.2, fontweight='bold', ha='left', va='bottom', clip_on=False)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

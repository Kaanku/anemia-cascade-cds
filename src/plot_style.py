# -*- coding: utf-8 -*-
"""
Tufte-Style Academic Plotting Boilerplate
Minimalist academic style that avoids chart-junk and maximizes
data visibility.

Usage:
    from plot_style import *
    # Style is applied automatically. In addition:
    # save_fig(fig, output_dir, filename)
    # despine_all(ax)
    # add_panel_label(ax, "A")
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
from pathlib import Path
from PIL import Image
import numpy as np

# ==============================================================================
# 1. DESIGN SYSTEM AND COLOR PALETTE
# ==============================================================================
CHOSEN_FONT = 'Arial'

PALETTE = {
    'highlight': '#C0392B',   # Emphasis / Critical (Ruby Red)
    'base1':     '#5D6D7E',   # Primary Data 1 (Steel Blue)
    'base2':     '#BDC3C7',   # Background / Reference (Pale Gray)
    'accent1':   '#27AE60',   # Positive (Emerald Green)
    'accent2':   '#E67E22',   # Primary Data 2 (Muted Orange)
    'accent3':   '#8E44AD',   # Alternative (Amethyst Purple)
}

# ==============================================================================
# 2. MATPLOTLIB GLOBAL SETTINGS
# ==============================================================================
def set_academic_style():
    """Configure matplotlib global settings following Tufte principles."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [CHOSEN_FONT, 'Arial', 'DejaVu Sans'],
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#555555',
        'legend.frameon': False,
        'legend.fontsize': 10,
        'axes.grid': False,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.prop_cycle': cycler(color=[
            PALETTE['base1'], PALETTE['accent2'], PALETTE['accent1'],
            PALETTE['highlight'], PALETTE['base2'], PALETTE['accent3']
        ])
    })

set_academic_style()

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def create_tufte_cmap(base_color):
    """Continuous colormap starting from white."""
    return LinearSegmentedColormap.from_list('tufte_custom', ['#FFFFFF', base_color])

def despine_all(ax):
    """Remove all axis spines."""
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=18):
    """Add a letter label (A, B, C) for multi-panel figures."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='left')

# ==============================================================================
# 4. SAVE FUNCTION (TIFF 600 DPI + PNG 300 DPI)
# ==============================================================================
def save_fig(fig, output_dir, filename, dpi=600):
    """Save as both TIFF (publication) and PNG (preview)."""
    out_path = Path(output_dir)
    tiff_dir = out_path / 'TIFF_600DPI'
    png_dir = out_path / 'PNG_300DPI'
    tiff_dir.mkdir(exist_ok=True, parents=True)
    png_dir.mkdir(exist_ok=True, parents=True)

    base_name = filename.replace('.png', '').replace('.tiff', '').replace('.tif', '')
    tiff_path = tiff_dir / f"{base_name}.tif"
    png_path = png_dir / f"{base_name}.png"

    try:
        fig.savefig(str(tiff_path), dpi=dpi, bbox_inches='tight', format='tif')
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(tiff_path) as img:
            img.save(tiff_path, compression="tiff_lzw")
        fig.savefig(str(png_path), dpi=300, bbox_inches='tight', format='png')
        print(f"  ✅ Saved: {base_name}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

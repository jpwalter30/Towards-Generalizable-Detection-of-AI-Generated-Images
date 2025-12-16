"""
Scientific Corporate Design Configuration
Consistent styling for all thesis plots
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# ============================================================
# COLOR PALETTE - Professional & Print-Safe
# ============================================================

COLORS = {
    # Primary colors (main data)
    'primary_blue': '#1f77b4',      # Multi-trained
    'primary_orange': '#ff7f0e',    # Single-trained
    'primary_green': '#2ca02c',     # Positive/Good
    'primary_red': '#d62728',       # Negative/Bad
    
    # Secondary colors (supporting data)
    'secondary_purple': '#9467bd',
    'secondary_brown': '#8c564b',
    'secondary_pink': '#e377c2',
    'secondary_gray': '#7f7f7f',
    
    # Generator-specific (8 distinct colors)
    'gen_colors': [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ],
    
    # Background colors
    'bg_real': '#808080',           # Real images (medium gray, more visible)
    'bg_fake': '#ff7f0e',           # Fake images (orange)
    
    # Advantage colors
    'advantage_positive': '#27ae60', # Green (multi better)
    'advantage_negative': '#e74c3c', # Red (single better)
    
    # Neutral
    'neutral_gray': '#95a5a6',
    'text_dark': '#2c3e50',
}

# ============================================================
# TYPOGRAPHY
# ============================================================

FONTS = {
    'family': 'sans-serif',
    'sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'size': {
        'title': 14,
        'label': 12,
        'tick': 10,
        'legend': 10,
        'annotation': 9,
    }
}

# ============================================================
# LAYOUT
# ============================================================

LAYOUT = {
    'figure': {
        'dpi': 300,
        'facecolor': 'white',
        'edgecolor': 'none',
    },
    'axes': {
        'titlesize': FONTS['size']['title'],
        'titleweight': 'bold',
        'labelsize': FONTS['size']['label'],
        'labelweight': 'normal',
        'grid': True,
        'grid_alpha': 0.3,
        'grid_linestyle': '--',
        'grid_linewidth': 0.5,
        'spines_top': False,
        'spines_right': False,
        'spines_linewidth': 1.0,
    },
    'legend': {
        'fontsize': FONTS['size']['legend'],
        'frameon': True,
        'framealpha': 0.9,
        'edgecolor': 'black',
        'fancybox': False,
    },
    'lines': {
        'linewidth': 1.5,
        'markersize': 6,
    },
    'bars': {
        'edgecolor': 'black',
        'linewidth': 0.7,
        'alpha': 0.9,
    }
}

# ============================================================
# Apply Theme Function
# ============================================================

def apply_thesis_style():
    """Apply consistent thesis styling to matplotlib."""
    
    # Font settings
    plt.rcParams['font.family'] = FONTS['family']
    plt.rcParams['font.sans-serif'] = FONTS['sans-serif']
    plt.rcParams['font.size'] = FONTS['size']['tick']
    
    # Figure settings
    plt.rcParams['figure.dpi'] = LAYOUT['figure']['dpi']
    plt.rcParams['figure.facecolor'] = LAYOUT['figure']['facecolor']
    plt.rcParams['figure.edgecolor'] = LAYOUT['figure']['edgecolor']
    plt.rcParams['savefig.dpi'] = LAYOUT['figure']['dpi']
    plt.rcParams['savefig.facecolor'] = LAYOUT['figure']['facecolor']
    plt.rcParams['savefig.edgecolor'] = LAYOUT['figure']['edgecolor']
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Axes settings
    plt.rcParams['axes.titlesize'] = LAYOUT['axes']['titlesize']
    plt.rcParams['axes.titleweight'] = LAYOUT['axes']['titleweight']
    plt.rcParams['axes.labelsize'] = LAYOUT['axes']['labelsize']
    plt.rcParams['axes.labelweight'] = LAYOUT['axes']['labelweight']
    plt.rcParams['axes.grid'] = LAYOUT['axes']['grid']
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.edgecolor'] = COLORS['text_dark']
    plt.rcParams['axes.labelcolor'] = COLORS['text_dark']
    plt.rcParams['axes.linewidth'] = LAYOUT['axes']['spines_linewidth']
    
    # Set color cycle
    plt.rcParams['axes.prop_cycle'] = cycler('color', COLORS['gen_colors'])
    
    # Grid
    plt.rcParams['grid.alpha'] = LAYOUT['axes']['grid_alpha']
    plt.rcParams['grid.linestyle'] = LAYOUT['axes']['grid_linestyle']
    plt.rcParams['grid.linewidth'] = LAYOUT['axes']['grid_linewidth']
    plt.rcParams['grid.color'] = COLORS['neutral_gray']
    
    # Legend
    plt.rcParams['legend.fontsize'] = LAYOUT['legend']['fontsize']
    plt.rcParams['legend.frameon'] = LAYOUT['legend']['frameon']
    plt.rcParams['legend.framealpha'] = LAYOUT['legend']['framealpha']
    plt.rcParams['legend.edgecolor'] = LAYOUT['legend']['edgecolor']
    plt.rcParams['legend.fancybox'] = LAYOUT['legend']['fancybox']
    
    # Ticks
    plt.rcParams['xtick.labelsize'] = FONTS['size']['tick']
    plt.rcParams['ytick.labelsize'] = FONTS['size']['tick']
    plt.rcParams['xtick.color'] = COLORS['text_dark']
    plt.rcParams['ytick.color'] = COLORS['text_dark']
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    
    # Lines
    plt.rcParams['lines.linewidth'] = LAYOUT['lines']['linewidth']
    plt.rcParams['lines.markersize'] = LAYOUT['lines']['markersize']

def remove_spines(ax, top=True, right=True, left=False, bottom=False):
    """Remove specific spines from axes."""
    if top:
        ax.spines['top'].set_visible(False)
    if right:
        ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
    if bottom:
        ax.spines['bottom'].set_visible(False)

def add_value_labels(ax, bars, format_str='{:.1f}%', offset=0.5, fontsize=14):  # Increased from 9
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        if height != 0:  # Only label non-zero bars
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   format_str.format(height),
                   ha='center', va='bottom', 
                   fontsize=fontsize, 
                   fontweight='bold',
                   color=COLORS['text_dark'])

def save_figure(fig, output_path, dpi=300):
    """Save figure with consistent settings."""
    fig.savefig(output_path, 
               dpi=dpi, 
               bbox_inches='tight', 
               facecolor='white',
               edgecolor='none',
               pad_inches=0.1)
    plt.close(fig)

# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Apply theme
    apply_thesis_style()
    
    # Create example plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = ['A', 'B', 'C', 'D']
    y1 = [75, 82, 68, 91]
    y2 = [65, 70, 60, 78]
    
    width = 0.35
    x_pos = range(len(x))
    
    bars1 = ax.bar([i - width/2 for i in x_pos], y1, width,
                   label='Multi→Single',
                   color=COLORS['primary_blue'],
                   **LAYOUT['bars'])
    
    bars2 = ax.bar([i + width/2 for i in x_pos], y2, width,
                   label='Single→Multi',
                   color=COLORS['primary_orange'],
                   **LAYOUT['bars'])
    
    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Generator')
    ax.set_title('Example Plot with Thesis Style')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_ylim(0, 100)
    ax.legend()
    
    save_figure(fig, 'example_thesis_style.png')
    print("✓ Example saved to: example_thesis_style.png")

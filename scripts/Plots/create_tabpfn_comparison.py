#!/usr/bin/env python3
"""
Improved visualizations for TabPFN test results
Loads actual results from JSON files and creates comparison plots
Uses consistent scientific corporate design
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse
from collections import defaultdict

# Import thesis style
import sys
sys.path.insert(0, str(Path(__file__).parent))
from thesis_style import (
    apply_thesis_style, COLORS, LAYOUT, save_figure, 
    add_value_labels, remove_spines
)

def load_tabpfn_results(results_dir):
    """Load TabPFN results from JSON files."""
    results_dir = Path(results_dir)
    
    multi_single_results = {}
    single_multi_results = {}
    
    print("Loading results...")
    print("="*60)
    
    # Load multi-single results
    multi_single_dir = results_dir / "multi-single" / "tabpfn"
    if multi_single_dir.exists():
        print(f"\nLoading multi-single from: {multi_single_dir}")
        for json_file in multi_single_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    test_gen = data.get('test_gen')
                    accuracy = data.get('metrics', {}).get('accuracy', 0) * 100
                    
                    if test_gen:
                        if test_gen not in multi_single_results:
                            multi_single_results[test_gen] = []
                        multi_single_results[test_gen].append(accuracy)
                        print(f"  {json_file.name}: {test_gen} = {accuracy:.1f}%")
            except Exception as e:
                print(f"  ✗ Error loading {json_file.name}: {e}")
    
    # Load single-multi results
    single_multi_dir = results_dir / "single-multi" / "tabpfn"
    if single_multi_dir.exists():
        print(f"\nLoading single-multi from: {single_multi_dir}")
        for json_file in single_multi_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    train_gen = data.get('train_gen')
                    accuracy = data.get('metrics', {}).get('accuracy', 0) * 100
                    
                    if train_gen:
                        if train_gen not in single_multi_results:
                            single_multi_results[train_gen] = []
                        single_multi_results[train_gen].append(accuracy)
                        print(f"  {json_file.name}: {train_gen} = {accuracy:.1f}%")
            except Exception as e:
                print(f"  ✗ Error loading {json_file.name}: {e}")
    
    # Average across train sizes
    print("\n" + "="*60)
    print("Averaging results across training sizes...")
    
    multi_single_avg = {gen: np.mean(accs) for gen, accs in multi_single_results.items()}
    single_multi_avg = {gen: np.mean(accs) for gen, accs in single_multi_results.items()}
    
    print("\nMulti-Single (averaged):")
    for gen, acc in sorted(multi_single_avg.items()):
        print(f"  {gen}: {acc:.1f}%")
    
    print("\nSingle-Multi (averaged):")
    for gen, acc in sorted(single_multi_avg.items()):
        print(f"  {gen}: {acc:.1f}%")
    
    return multi_single_avg, single_multi_avg

def clean_generator_name(name):
    """Clean generator names for display - CONSISTENT across all plots."""
    replacements = {
        'ADM': 'ADM',
        'BigGAN': 'BigGAN',
        'glide': 'glide',
        'Midjourney': 'Midjourney',
        'stable_diffusion_v_1_4': 'Stable Diffusion v1.4',
        'stable_diffusion_v_1_5': 'Stable Diffusion v1.5',
        'VQDM': 'VQDM',
        'wukong': 'wukong'
    }
    return replacements.get(name, name)

def style_1_grouped_bars(multi_single, single_multi):
    """Side-by-side comparison with grouped bars - alphabetically sorted - THESIS STYLE."""
    # Get common generators and sort alphabetically
    generators = sorted(set(multi_single.keys()) & set(single_multi.keys()))
    
    if not generators:
        print("✗ No common generators found!")
        return None
    
    # Clean names
    gen_labels = [clean_generator_name(g) for g in generators]
    ms_vals = [multi_single[g] for g in generators]
    sm_vals = [single_multi[g] for g in generators]
    
    x = np.arange(len(generators))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 9))  # Much larger
    
    bars1 = ax.bar(x - width/2, ms_vals, width, label='Multi→Single', 
                   color=COLORS['primary_blue'],  # Thesis blue
                   **LAYOUT['bars'])
    bars2 = ax.bar(x + width/2, sm_vals, width, label='Single→Multi', 
                   color=COLORS['primary_orange'],  # Thesis orange
                   **LAYOUT['bars'])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')  # Larger
    
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')  # Larger
    ax.set_xlabel('Generator', fontsize=18, fontweight='bold')  # Larger
    ax.set_xticks(x)
    ax.set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=14)  # Larger
    ax.tick_params(axis='y', labelsize=14)  # Larger
    ax.legend(fontsize=14, loc='upper left')  # Larger
    ax.set_ylim(50, 100)  # Start at 50
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def style_2_sorted_comparison(multi_single, single_multi):
    """Alphabetically sorted comparison - THESIS STYLE."""
    generators = sorted(set(multi_single.keys()) & set(single_multi.keys()))
    
    if not generators:
        return None
    
    # Sort alphabetically (already done by sorted())
    generators_sorted = generators
    gen_labels = [clean_generator_name(g) for g in generators_sorted]
    ms_sorted = np.array([multi_single[g] for g in generators_sorted])
    sm_sorted = np.array([single_multi[g] for g in generators_sorted])
    diff_sorted = ms_sorted - sm_sorted
    
    x = np.arange(len(generators_sorted))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),  # Much larger
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Grouped bars
    bars1 = ax1.bar(x - width/2, ms_sorted, width, 
                    label='Multi→Single', 
                    color=COLORS['primary_blue'],
                    **LAYOUT['bars'])
    bars2 = ax1.bar(x + width/2, sm_sorted, width, 
                    label='Single→Multi', 
                    color=COLORS['primary_orange'],
                    **LAYOUT['bars'])
    
    # Value labels removed for thesis
    
    ax1.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')  # Larger
    ax1.set_title('TabPFN — Performance Comparison [mean]', 
                 fontsize=20, fontweight='bold', pad=15)  # Larger
    ax1.set_xticks(x)
    ax1.set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=14)  # Larger
    ax1.tick_params(axis='y', labelsize=14)  # Larger
    ax1.legend(loc='upper left', fontsize=14)  # Larger, moved to upper left
    ax1.set_ylim(50, 100)  # Start at 50
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    remove_spines(ax1)
    
    # Bottom: Difference plot
    colors = [COLORS['advantage_positive'] if d > 0 else COLORS['advantage_negative'] 
             for d in diff_sorted]
    bars = ax2.bar(x, diff_sorted, color=colors, 
                   edgecolor='black', linewidth=0.7, alpha=0.8)
    
    for i, (bar, d) in enumerate(zip(bars, diff_sorted)):
        ax2.text(bar.get_x() + bar.get_width()/2., d + (0.5 if d > 0 else -0.5),
                f'{d:+.1f}%',
                ha='center', va='bottom' if d > 0 else 'top', 
                fontsize=12, fontweight='bold',  # Larger
                color=COLORS['text_dark'])
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Advantage (%)', fontsize=16, fontweight='bold')  # Larger
    ax2.set_xlabel('Generator', fontsize=16, fontweight='bold')  # Larger
    ax2.set_xticks(x)
    ax2.set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=14)  # Larger
    ax2.tick_params(axis='y', labelsize=14)  # Larger
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    remove_spines(ax2)
    
    plt.tight_layout()
    return fig

def style_3_connected_dots(multi_single, single_multi):
    """Connected dot plot showing the gap - alphabetically sorted."""
    generators = sorted(set(multi_single.keys()) & set(single_multi.keys()))
    
    if not generators:
        return None
    
    # Already alphabetically sorted
    gen_labels = [clean_generator_name(g) for g in generators]
    ms_vals = [multi_single[g] for g in generators]
    sm_vals = [single_multi[g] for g in generators]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y = np.arange(len(generators))
    
    # Draw connecting lines
    for i in range(len(generators)):
        ax.plot([sm_vals[i], ms_vals[i]], [y[i], y[i]], 
               color='gray', linewidth=2, alpha=0.5, zorder=1)
    
    # Plot points
    ax.scatter(ms_vals, y, s=200, color='#2E86AB', 
              edgecolor='black', linewidth=2, label='Multi→Single', zorder=3)
    ax.scatter(sm_vals, y, s=200, color='#A23B72', 
              edgecolor='black', linewidth=2, label='Single→Multi', zorder=3)
    
    # Add value labels
    for i, (ms, sm) in enumerate(zip(ms_vals, sm_vals)):
        ax.text(ms + 1, y[i], f'{ms:.1f}%', va='center', fontsize=10, fontweight='bold')
        ax.text(sm - 1, y[i], f'{sm:.1f}%', va='center', ha='right', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y)
    ax.set_yticklabels(gen_labels, fontsize=11)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('TabPFN — Performance Comparison [mean]', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    remove_spines(ax)
    
    plt.tight_layout()
    return fig

def style_5_scatter_with_diagonal(multi_single, single_multi):
    """Scatter plot showing correlation."""
    generators = sorted(set(multi_single.keys()) & set(single_multi.keys()))
    
    if not generators:
        return None
    
    gen_labels = [clean_generator_name(g) for g in generators]
    ms_vals = np.array([multi_single[g] for g in generators])
    sm_vals = np.array([single_multi[g] for g in generators])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine axis limits
    min_val = min(sm_vals.min(), ms_vals.min()) - 5
    max_val = max(sm_vals.max(), ms_vals.max()) + 5
    
    # Diagonal line (equal performance)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.3, 
           label='Equal Performance')
    
    # Scatter points
    scatter = ax.scatter(sm_vals, ms_vals, s=300, 
                        c=ms_vals - sm_vals, cmap='RdYlGn',
                        edgecolor='black', linewidth=2, zorder=3,
                        vmin=0, vmax=(ms_vals - sm_vals).max())
    
    # Add generator labels
    for i, gen in enumerate(gen_labels):
        ax.annotate(gen, (sm_vals[i], ms_vals[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', alpha=0.8))
    
    ax.set_xlabel('Single→Multi Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Multi→Single Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('TabPFN — Multi-Training vs Single-Training Performance [mean]', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Multi-Training Advantage (%)', rotation=270, labelpad=20, 
                  fontsize=11, fontweight='bold')
    
    # Add text annotation
    ax.text(0.05, 0.95, 'Points above line:\nMulti-training performs better',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
           verticalalignment='top')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing multi-single/ and single-multi/ results')
    parser.add_argument('--output_dir', type=str, default='comparison_plots',
                       help='Output directory for plots')
    parser.add_argument('--style', type=str, 
                       choices=['1', '2', '3', '5', 'all'],
                       default='all',
                       help='Which visualization style to generate')
    args = parser.parse_args()
    
    # Apply thesis style
    apply_thesis_style()
    
    # Load results
    multi_single, single_multi = load_tabpfn_results(args.results_dir)
    
    if not multi_single or not single_multi:
        print("\n✗ No results loaded! Check your results directory.")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    styles = {
        '1': ('Grouped Bars', style_1_grouped_bars),
        '2': ('Sorted with Difference', style_2_sorted_comparison),
        '3': ('Connected Dots', style_3_connected_dots),
        '5': ('Scatter Plot', style_5_scatter_with_diagonal)
    }
    
    if args.style == 'all':
        selected_styles = styles.items()
    else:
        selected_styles = [(args.style, styles[args.style])]
    
    for style_id, (name, func) in selected_styles:
        print(f"\nStyle {style_id}: {name}")
        fig = func(multi_single, single_multi)
        
        if fig is None:
            print(f"  ✗ Skipped (no data)")
            continue
        
        output_path = output_dir / f'tabpfn_style_{style_id}_{name.lower().replace(" ", "_")}.png'
        save_figure(fig, output_path)
        print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print(f"✓ All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
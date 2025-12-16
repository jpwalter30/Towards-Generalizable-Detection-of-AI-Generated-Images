#!/usr/bin/env python3
"""
Compare TabPFN vs LATTE in Single→Multi mode.
Grid comparison showing performance by training generator.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse
from collections import defaultdict
import sys

# Import thesis style
sys.path.insert(0, str(Path(__file__).parent))
from thesis_style import apply_thesis_style, COLORS, save_figure

def clean_generator_name(name):
    """Clean generator names for display."""
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

def load_tabpfn_single_multi(results_dir):
    """Load TabPFN single-multi results."""
    tabpfn_dir = Path(results_dir) / 'results_tabpfn' / 'single-multi' / 'tabpfn'
    
    if not tabpfn_dir.exists():
        print(f"✗ TabPFN directory not found: {tabpfn_dir}")
        return {}
    
    print(f"\nLoading TabPFN single-multi results from: {tabpfn_dir}")
    
    # Structure: {train_generator: {train_size: accuracy}}
    results = defaultdict(lambda: defaultdict(float))
    
    for json_file in sorted(tabpfn_dir.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Parse filename: single-multi_150_tabpfn_ADM.json
            parts = json_file.stem.split('_')
            
            # Find training size (numeric part after "single-multi")
            train_size = None
            train_gen = None
            
            for i, part in enumerate(parts):
                if part.isdigit():
                    train_size = int(part)
                    # Generator is everything after 'tabpfn'
                    if 'tabpfn' in parts:
                        tabpfn_idx = parts.index('tabpfn')
                        train_gen = '_'.join(parts[tabpfn_idx + 1:])
                    break
            
            if train_size and train_gen:
                # Get accuracy
                if isinstance(data, dict):
                    if 'metrics' in data:
                        accuracy = data['metrics'].get('accuracy', 0) * 100
                    else:
                        accuracy = data.get('accuracy', 0) * 100
                else:
                    continue
                
                results[train_gen][train_size] = accuracy
                print(f"  {json_file.name}: gen={train_gen}, size={train_size}, acc={accuracy:.2f}%")
        
        except Exception as e:
            print(f"  ⚠ Error loading {json_file.name}: {e}")
            continue
    
    return results

def load_latte_single_multi(results_dir):
    """Load LATTE single-multi results."""
    latte_dir = Path(results_dir) / 'results_latte' / 'single-multi' / 'latte'
    
    if not latte_dir.exists():
        print(f"✗ LATTE directory not found: {latte_dir}")
        return {}
    
    print(f"\nLoading LATTE single-multi results from: {latte_dir}")
    
    # Structure: {train_generator: {train_size: accuracy}}
    results = defaultdict(lambda: defaultdict(float))
    
    for json_file in sorted(latte_dir.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Parse filename: single-multi_ADM_150_latte.json
            # Format: single-multi_{GENERATOR}_{SIZE}_latte.json
            stem = json_file.stem.replace('_latte', '')  # Remove _latte suffix
            
            # Remove 'single-multi_' prefix
            if stem.startswith('single-multi_'):
                stem = stem[len('single-multi_'):]
            
            # Split remaining parts
            parts = stem.split('_')
            
            if len(parts) < 2:
                continue
            
            # Last part is training size
            train_size = int(parts[-1])
            # Everything before that is generator name
            train_gen = '_'.join(parts[:-1])
            
            # Get accuracy from nested structure
            if isinstance(data, dict):
                # LATTE format might have nested structure
                for key, value in data.items():
                    if isinstance(value, dict) and 'accuracy' in value:
                        accuracy = value['accuracy'] * 100
                        results[train_gen][train_size] = accuracy
                        print(f"  {json_file.name}: gen={train_gen}, size={train_size}, acc={accuracy:.2f}%")
                        break
                else:
                    # Direct accuracy field
                    if 'accuracy' in data:
                        accuracy = data['accuracy'] * 100
                        results[train_gen][train_size] = accuracy
                        print(f"  {json_file.name}: gen={train_gen}, size={train_size}, acc={accuracy:.2f}%")
        
        except Exception as e:
            print(f"  ⚠ Error loading {json_file.name}: {e}")
            continue
    
    return results

def plot_single_multi_comparison(tabpfn_results, latte_results, output_path, exclude_sizes=None):
    """Create grid comparison plot for single-multi mode."""
    
    # Find common generators and training sizes
    common_gens = sorted(set(tabpfn_results.keys()) & set(latte_results.keys()))
    
    if not common_gens:
        print("✗ No common generators found!")
        return
    
    # Find common training sizes across all generators
    all_sizes = set()
    for gen in common_gens:
        all_sizes.update(tabpfn_results[gen].keys())
        all_sizes.update(latte_results[gen].keys())
    
    # Filter out excluded sizes
    if exclude_sizes:
        all_sizes = all_sizes - set(exclude_sizes)
        print(f"\nExcluding training sizes: {exclude_sizes}")
    
    common_sizes = sorted(all_sizes)
    
    if not common_sizes:
        print("✗ No common training sizes remaining after filtering!")
        return
    
    print(f"\n✓ Found {len(common_gens)} common generators: {common_gens}")
    print(f"✓ Found {len(common_sizes)} training sizes: {common_sizes}")
    
    # Setup grid (3 rows x 3 cols for 8 generators + 1 empty)
    n_cols = 3
    n_rows = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 18),
                            sharex=True, sharey=True)
    axes = axes.flatten()
    
    print("\n" + "="*60)
    print("Creating Single→Multi Comparison Grid")
    print("="*60)
    
    for idx, train_gen in enumerate(common_gens):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        gen_label = clean_generator_name(train_gen)
        
        print(f"\n[{idx+1}/{len(common_gens)}] {gen_label}")
        
        # Get accuracies for this generator
        tabpfn_acc = [tabpfn_results[train_gen].get(size, np.nan) for size in common_sizes]
        latte_acc = [latte_results[train_gen].get(size, np.nan) for size in common_sizes]
        
        print(f"  TabPFN: {tabpfn_acc}")
        print(f"  LATTE: {latte_acc}")
        
        # Plot lines
        ax.plot(common_sizes, tabpfn_acc,
               marker='o', markersize=12, linewidth=3.5,
               color=COLORS['primary_blue'],
               label='TabPFN')
        
        ax.plot(common_sizes, latte_acc,
               marker='s', markersize=12, linewidth=3.5,
               color=COLORS['primary_orange'],
               label='LATTE')
        
        # Formatting
        ax.set_title(gen_label, fontsize=27, fontweight='bold')  # Increased by 5
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        
        # Remove intermediate x-axis ticks
        ax.set_xticks(common_sizes)
        ax.set_xticklabels([str(s) for s in common_sizes])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', labelsize=23)  # Increased by 5
    
    # Hide unused subplots
    for idx in range(len(common_gens), len(axes)):
        axes[idx].axis('off')
    
    # Add shared labels
    # X-axis label removed for thesis - numbers remain visible
    fig.text(0.02, 0.5, 'Accuracy (%)', 
            va='center', rotation='vertical', fontsize=20, fontweight='bold')
    
    # Title removed for thesis
    
    # Add global legend at bottom center
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=COLORS['primary_blue'], 
               linewidth=3.5, markersize=12, label='TabPFN'),
        Line2D([0], [0], marker='s', color=COLORS['primary_orange'], 
               linewidth=3.5, markersize=12, label='LATTE')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=2, fontsize=25, frameon=True,  # Increased by 5
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0.05, 0.06, 1, 0.96])
    save_figure(fig, output_path)
    print(f"\n✓ Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare TabPFN vs LATTE in Single→Multi mode'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base directory containing results_tabpfn and results_latte')
    parser.add_argument('--output', type=str, default='comparison_single_multi_grid.png',
                       help='Output filename')
    parser.add_argument('--exclude_sizes', type=int, nargs='+', default=None,
                       help='Training sizes to exclude (e.g., --exclude_sizes 25 30 75)')
    
    args = parser.parse_args()
    
    apply_thesis_style()
    
    print("="*60)
    print("TabPFN vs LATTE Single→Multi Comparison")
    print("="*60)
    
    # Load results
    tabpfn_results = load_tabpfn_single_multi(args.results_dir)
    latte_results = load_latte_single_multi(args.results_dir)
    
    if not tabpfn_results:
        print("\n✗ No TabPFN results loaded!")
        return
    
    if not latte_results:
        print("\n✗ No LATTE results loaded!")
        return
    
    # Create comparison plot
    plot_single_multi_comparison(tabpfn_results, latte_results, args.output, 
                                 exclude_sizes=args.exclude_sizes)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    common_gens = sorted(set(tabpfn_results.keys()) & set(latte_results.keys()))
    
    for gen in common_gens:
        gen_label = clean_generator_name(gen)
        tabpfn_avg = np.mean(list(tabpfn_results[gen].values()))
        latte_avg = np.mean(list(latte_results[gen].values()))
        diff = tabpfn_avg - latte_avg
        
        print(f"\n{gen_label}:")
        print(f"  TabPFN: {tabpfn_avg:.1f}%")
        print(f"  LATTE:  {latte_avg:.1f}%")
        print(f"  Diff:   {diff:+.1f}%")
    
    # Overall average
    all_tabpfn = [v for gen_dict in tabpfn_results.values() for v in gen_dict.values()]
    all_latte = [v for gen_dict in latte_results.values() for v in gen_dict.values()]
    
    print(f"\nOverall Average:")
    print(f"  TabPFN: {np.mean(all_tabpfn):.1f}%")
    print(f"  LATTE:  {np.mean(all_latte):.1f}%")
    print(f"  Diff:   {np.mean(all_tabpfn) - np.mean(all_latte):+.1f}%")

if __name__ == '__main__':
    main()
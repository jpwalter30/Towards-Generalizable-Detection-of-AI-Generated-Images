#!/usr/bin/env python3
"""
Compare TabPFN vs LATTE for Multi-Single mode
Grid layout with one subplot per test generator
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse
import sys
from collections import defaultdict

# Import thesis style
sys.path.insert(0, str(Path(__file__).parent))
from thesis_style import apply_thesis_style, COLORS, save_figure

def load_multi_single_results(results_dir, model_name):
    """Load multi-single results for a specific model."""
    results_dir = Path(results_dir)
    model_dir = results_dir / "multi-single" / model_name.lower()
    
    if not model_dir.exists():
        print(f"✗ Directory not found: {model_dir}")
        return {}
    
    # Structure: {generator: {size: accuracy}}
    results = defaultdict(dict)
    
    print(f"\nLoading {model_name} multi-single results from: {model_dir}")
    
    for json_file in sorted(model_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                
                # Handle different JSON structures
                # LATTE: nested dict with keys like "multi-single_ADM_150"
                # TabPFN: single dict with test_gen, train_size, metrics
                
                if isinstance(data, dict) and any(key.startswith('multi-single_') for key in data.keys()):
                    # LATTE format: nested structure
                    for key, result in data.items():
                        if not key.startswith('multi-single_'):
                            continue
                        
                        # Extract from key: "multi-single_ADM_150"
                        # Remove prefix: "multi-single_"
                        remaining = key[len('multi-single_'):]
                        
                        # Split by underscore and find the training size (last numeric part)
                        parts = remaining.split('_')
                        
                        # The last part should be the training size
                        try:
                            train_size = int(parts[-1])
                            # Everything before the last part is the generator name
                            test_gen = '_'.join(parts[:-1])
                        except (ValueError, IndexError):
                            continue
                        
                        if not test_gen:
                            continue
                        
                        # Get accuracy
                        accuracy = result.get('accuracy', 0) * 100
                        
                        results[test_gen][train_size] = accuracy
                        print(f"  {key}: gen={test_gen}, size={train_size}, acc={accuracy:.2f}%")
                
                else:
                    # TabPFN format: flat structure
                    test_gen = data.get('test_gen')
                    if not test_gen:
                        continue
                    
                    # Get train_size
                    if 'train_size' in data:
                        train_size = data['train_size']
                    else:
                        # Extract from filename
                        parts = json_file.stem.split('_')
                        for part in parts:
                            if part.isdigit():
                                train_size = int(part)
                                break
                        else:
                            continue
                    
                    # Get accuracy
                    accuracy = data.get('metrics', {}).get('accuracy', 0) * 100
                    
                    results[test_gen][train_size] = accuracy
                    print(f"  {json_file.name}: gen={test_gen}, size={train_size}, acc={accuracy:.2f}%")
                
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    return dict(results)

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

def plot_grid_comparison(tabpfn_data, latte_data, output_path):
    """Create grid of subplots comparing TabPFN vs LATTE per generator."""
    
    if not tabpfn_data or not latte_data:
        print("✗ Missing data for comparison")
        return
    
    # Get common generators
    common_gens = sorted(set(tabpfn_data.keys()) & set(latte_data.keys()))
    
    if not common_gens:
        print("✗ No common generators found!")
        return
    
    print(f"\n✓ Found {len(common_gens)} common generators: {common_gens}")
    
    # Setup grid 
    n_cols = 3
    n_rows = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 18), 
                            sharex=True, sharey=True)
    axes = axes.flatten()
    
    print("\n" + "="*60)
    print("Creating Grid Comparison")
    print("="*60)
    
    for idx, gen in enumerate(common_gens):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        gen_label = clean_generator_name(gen)
        
        print(f"\n[{idx+1}/{len(common_gens)}] Processing {gen_label}...")
        
        # Get data for this generator
        tabpfn_gen = tabpfn_data[gen]
        latte_gen = latte_data[gen]
        
        # Find common sizes
        common_sizes = sorted(set(tabpfn_gen.keys()) & set(latte_gen.keys()))
        
        if not common_sizes:
            print(f"  ✗ No common sizes for {gen_label}")
            ax.text(0.5, 0.5, f'No data\nfor {gen_label}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gen_label, fontsize=10, fontweight='bold')
            continue
        
        print(f"  Common sizes: {common_sizes}")
        
        # Prepare data
        tabpfn_acc = [tabpfn_gen[s] for s in common_sizes]
        latte_acc = [latte_gen[s] for s in common_sizes]
        
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
        ax.set_title(gen_label, fontsize=27, fontweight='bold') 
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        
        # Remove intermediate x-axis ticks and format as regular numbers
        ax.set_xticks(common_sizes)
        ax.set_xticklabels([str(s) for s in common_sizes])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='both', labelsize=23)  
        
        
        # Set y-limits
        ax.set_ylim(50, 100)
    
    # Hide unused subplots
    for idx in range(len(common_gens), len(axes)):
        axes[idx].axis('off')
    
    # Add shared labels
    fig.text(0.02, 0.5, 'Accuracy (%)', 
            va='center', rotation='vertical', fontsize=20, fontweight='bold')  
    
    
    # Add global legend at bottom center
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=COLORS['primary_blue'], 
               linewidth=3.5, markersize=12, label='TabPFN'),
        Line2D([0], [0], marker='s', color=COLORS['primary_orange'], 
               linewidth=3.5, markersize=12, label='LATTE')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=2, fontsize=25, frameon=True,  
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0.05, 0.06, 1, 0.96])  
    save_figure(fig, output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Average Performance by Generator")
    print("="*60)
    print(f"{'Generator':<15} {'TabPFN':<10} {'LATTE':<10} {'Difference':<10}")
    print("-"*60)
    
    for gen in common_gens:
        tabpfn_gen = tabpfn_data[gen]
        latte_gen = latte_data[gen]
        common_sizes = sorted(set(tabpfn_gen.keys()) & set(latte_gen.keys()))
        
        if common_sizes:
            tabpfn_mean = np.mean([tabpfn_gen[s] for s in common_sizes])
            latte_mean = np.mean([latte_gen[s] for s in common_sizes])
            diff = tabpfn_mean - latte_mean
            
            gen_label = clean_generator_name(gen)
            print(f"{gen_label:<15} {tabpfn_mean:>6.2f}%    {latte_mean:>6.2f}%    {diff:>+6.2f}%")
    
    # Overall mean
    all_tabpfn = []
    all_latte = []
    for gen in common_gens:
        tabpfn_gen = tabpfn_data[gen]
        latte_gen = latte_data[gen]
        common_sizes = sorted(set(tabpfn_gen.keys()) & set(latte_gen.keys()))
        all_tabpfn.extend([tabpfn_gen[s] for s in common_sizes])
        all_latte.extend([latte_gen[s] for s in common_sizes])
    
    if all_tabpfn and all_latte:
        print("-"*60)
        print(f"{'Overall Mean:':<15} {np.mean(all_tabpfn):>6.2f}%    "
              f"{np.mean(all_latte):>6.2f}%    "
              f"{np.mean(all_tabpfn) - np.mean(all_latte):>+6.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Compare TabPFN vs LATTE for multi-single')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base results directory')
    parser.add_argument('--tabpfn_dir', type=str, default=None,
                       help='TabPFN results directory')
    parser.add_argument('--latte_dir', type=str, default=None,
                       help='LATTE results directory')
    parser.add_argument('--output', type=str, default='comparison_tabpfn_vs_latte_multi_single.png',
                       help='Output filename')
    
    args = parser.parse_args()
    
    apply_thesis_style()
    
    print("\n" + "="*60)
    print("TabPFN vs LATTE Multi-Single Grid Comparison")
    print("="*60)
    
    base_dir = Path(args.results_dir)
    
    # Determine directories
    tabpfn_dir = Path(args.tabpfn_dir) if args.tabpfn_dir else (base_dir / "results_tabpfn")
    latte_dir = Path(args.latte_dir) if args.latte_dir else (base_dir / "results_latte")
    
    print(f"\nTabPFN directory: {tabpfn_dir}")
    print(f"LATTE directory: {latte_dir}")
    
    # Load results
    tabpfn_data = load_multi_single_results(tabpfn_dir, "tabpfn")
    latte_data = load_multi_single_results(latte_dir, "latte")
    
    if not tabpfn_data:
        print("\n✗ No TabPFN data found!")
        return
    if not latte_data:
        print("\n✗ No LATTE data found!")
        return
    
    # Create comparison plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_grid_comparison(tabpfn_data, latte_data, output_path)

if __name__ == "__main__":
    main()
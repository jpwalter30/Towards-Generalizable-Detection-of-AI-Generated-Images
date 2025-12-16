#!/usr/bin/env python3
"""
Compare TabPFN vs LATTE for Multi-Multi mode
Shows performance comparison across different training sizes
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse
import sys

# Import thesis style
sys.path.insert(0, str(Path(__file__).parent))
from thesis_style import apply_thesis_style, COLORS, save_figure

def load_multi_multi_results(results_dir, model_name):
    """Load multi-multi results for a specific model."""
    results_dir = Path(results_dir)
    model_dir = results_dir / "multi-multi" / model_name.lower()
    
    if not model_dir.exists():
        print(f"✗ Directory not found: {model_dir}")
        return {}
    
    results = {}
    
    print(f"\nLoading {model_name} multi-multi results from: {model_dir}")
    
    for json_file in sorted(model_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                
                # Handle different JSON structures
                # LATTE: nested dict with keys like "multi-multi_150"
                # TabPFN: single dict with train_size, metrics
                
                if isinstance(data, dict) and any(key.startswith('multi-multi_') for key in data.keys()):
                    # LATTE format: nested structure
                    for key, result in data.items():
                        if not key.startswith('multi-multi_'):
                            continue
                        
                        # Extract train_size from key: "multi-multi_150"
                        parts = key.split('_')
                        if len(parts) < 3:
                            continue
                        
                        train_size = int(parts[-1])
                        
                        # Get accuracy
                        accuracy = result.get('accuracy', 0) * 100
                        
                        results[train_size] = accuracy
                        print(f"  {key}: size={train_size}, acc={accuracy:.2f}%")
                
                else:
                    # TabPFN format: flat structure
                    if 'train_size' in data:
                        train_size = data['train_size']
                    else:
                        # Extract from filename: multi-multi_150_latte.json
                        parts = json_file.stem.split('_')
                        for part in parts:
                            if part.isdigit():
                                train_size = int(part)
                                break
                        else:
                            continue
                    
                    # Get accuracy
                    accuracy = data.get('metrics', {}).get('accuracy', 0) * 100
                    
                    results[train_size] = accuracy
                    print(f"  {json_file.name}: size={train_size}, acc={accuracy:.2f}%")
                
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    return results

def plot_comparison(tabpfn_data, latte_data, output_path):
    """Create line plot comparing TabPFN vs LATTE."""
    
    if not tabpfn_data or not latte_data:
        print("✗ Missing data for comparison")
        return
    
    # Get common training sizes
    common_sizes = sorted(set(tabpfn_data.keys()) & set(latte_data.keys()))
    
    if not common_sizes:
        print("✗ No common training sizes found!")
        print(f"  TabPFN sizes: {sorted(tabpfn_data.keys())}")
        print(f"  LATTE sizes: {sorted(latte_data.keys())}")
        return
    
    print(f"\n✓ Found {len(common_sizes)} common training sizes: {common_sizes}")
    
    # Prepare data
    tabpfn_acc = [tabpfn_data[s] for s in common_sizes]
    latte_acc = [latte_data[s] for s in common_sizes]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot lines with markers
    ax.plot(common_sizes, tabpfn_acc, 
           marker='o', markersize=10, linewidth=2.5,
           color=COLORS['primary_blue'], 
           label='TabPFN', zorder=3)
    
    ax.plot(common_sizes, latte_acc, 
           marker='s', markersize=10, linewidth=2.5,
           color=COLORS['primary_orange'], 
           label='LATTE', zorder=3)
    
    # Add value labels on points
    for i, (size, tab_acc, lat_acc) in enumerate(zip(common_sizes, tabpfn_acc, latte_acc)):
        # TabPFN labels (above points)
        ax.text(size, tab_acc + 1.5, f'{tab_acc:.1f}%',
               ha='center', va='bottom', fontsize=14, fontweight='bold',  # Increased from 9
               color=COLORS['primary_blue'])
        
        # LATTE labels (below points)
        ax.text(size, lat_acc - 1.5, f'{lat_acc:.1f}%',
               ha='center', va='top', fontsize=14, fontweight='bold',  # Increased from 9
               color=COLORS['primary_orange'])
    
    # Formatting
    ax.set_xlabel('Training Size (images per class)', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')  # Increased from 12
    
    # Set x-axis with only actual data points (no intermediate ticks)
    ax.set_xscale('log')
    ax.set_xticks(common_sizes)
    ax.set_xticklabels([str(s) for s in common_sizes], fontsize=14)
    # Disable minor ticks to avoid 2×10² notation
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    # Set y-axis with larger font
    ax.tick_params(axis='y', labelsize=14)  # Increased from default
    
    # Set y-axis
    min_acc = min(min(tabpfn_acc), min(latte_acc))
    max_acc = max(max(tabpfn_acc), max(latte_acc))
    y_margin = 5
    ax.set_ylim(max(50, min_acc - y_margin), min(100, max_acc + y_margin))
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=14, loc='lower right', frameon=True)  # Increased from 12
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(f"{'Size':<10} {'TabPFN':<12} {'LATTE':<12} {'Difference':<12}")
    print("-"*60)
    for size, tab, lat in zip(common_sizes, tabpfn_acc, latte_acc):
        diff = tab - lat
        print(f"{size:<10} {tab:>6.2f}%      {lat:>6.2f}%      {diff:>+6.2f}%")
    print("-"*60)
    print(f"{'Mean:':<10} {np.mean(tabpfn_acc):>6.2f}%      "
          f"{np.mean(latte_acc):>6.2f}%      "
          f"{np.mean(tabpfn_acc) - np.mean(latte_acc):>+6.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Compare TabPFN vs LATTE for multi-multi')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base results directory containing tabpfn and latte results')
    parser.add_argument('--tabpfn_dir', type=str, default=None,
                       help='TabPFN results directory (default: results_dir/results_tabpfn)')
    parser.add_argument('--latte_dir', type=str, default=None,
                       help='LATTE results directory (default: results_dir/results_latte)')
    parser.add_argument('--output', type=str, default='comparison_tabpfn_vs_latte_multi_multi.png',
                       help='Output filename')
    
    args = parser.parse_args()
    
    apply_thesis_style()
    
    print("\n" + "="*60)
    print("TabPFN vs LATTE Multi-Multi Comparison")
    print("="*60)
    
    base_dir = Path(args.results_dir)
    
    # Determine directories
    tabpfn_dir = Path(args.tabpfn_dir) if args.tabpfn_dir else (base_dir / "results_tabpfn")
    latte_dir = Path(args.latte_dir) if args.latte_dir else (base_dir / "results_latte")
    
    print(f"\nTabPFN directory: {tabpfn_dir}")
    print(f"LATTE directory: {latte_dir}")
    
    # Load results
    tabpfn_data = load_multi_multi_results(tabpfn_dir, "tabpfn")
    latte_data = load_multi_multi_results(latte_dir, "latte")
    
    if not tabpfn_data:
        print("\n✗ No TabPFN data found!")
        return
    if not latte_data:
        print("\n✗ No LATTE data found!")
        return
    
    # Create comparison plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(tabpfn_data, latte_data, output_path)

if __name__ == "__main__":
    main()

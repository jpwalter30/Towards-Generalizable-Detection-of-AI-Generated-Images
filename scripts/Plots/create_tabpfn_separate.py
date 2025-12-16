#!/usr/bin/env python3
"""
Separate plots for each TabPFN training/testing mode
- Multi-Multi: Multi-trained tested on multi-mixed
- Multi-Single: Multi-trained tested on single generators
- Single-Multi: Single-trained tested on multi-mixed
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
from thesis_style import apply_thesis_style, COLORS, LAYOUT, save_figure, add_value_labels

def load_tabpfn_results(results_dir):
    """Load TabPFN results from JSON files."""
    results_dir = Path(results_dir)
    
    multi_multi_results = {}
    multi_single_results = {}
    single_multi_results = {}
    
    print("Loading results...")
    print("="*60)
    
    # Load multi-multi results
    multi_multi_dir = results_dir / "multi-multi" / "tabpfn"
    if multi_multi_dir.exists():
        print(f"\nLoading multi-multi from: {multi_multi_dir}")
        for json_file in multi_multi_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    train_size = data.get('train_size', 0)
                    accuracy = data.get('metrics', {}).get('accuracy', 0) * 100
                    
                    if train_size not in multi_multi_results:
                        multi_multi_results[train_size] = []
                    multi_multi_results[train_size].append(accuracy)
                    print(f"  {json_file.name}: size={train_size} acc={accuracy:.1f}%")
            except Exception as e:
                print(f"  ✗ Error loading {json_file.name}: {e}")
    
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
    
    # Average results
    print("\n" + "="*60)
    print("Averaging results...")
    
    multi_multi_avg = {size: np.mean(accs) for size, accs in multi_multi_results.items()}
    multi_single_avg = {gen: np.mean(accs) for gen, accs in multi_single_results.items()}
    single_multi_avg = {gen: np.mean(accs) for gen, accs in single_multi_results.items()}
    
    return multi_multi_avg, multi_single_avg, single_multi_avg

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

def plot_multi_multi(data, output_path, exclude_sizes=None):
    """Plot multi-multi results (by training size)."""
    
    if not data:
        print("✗ No multi-multi data")
        return
    
    print("\n" + "="*60)
    print("Creating Multi-Multi Plot")
    print("="*60)
    
    # Filter out excluded sizes
    if exclude_sizes:
        data = {k: v for k, v in data.items() if k not in exclude_sizes}
        print(f"Excluding training sizes: {exclude_sizes}")
    
    if not data:
        print("✗ No data remaining after filtering")
        return
    
    # Sort by training size
    sizes = sorted(data.keys())
    accuracies = [data[s] for s in sizes]
    
    fig, ax = plt.subplots(figsize=(14, 9))  # Much larger
    
    bars = ax.bar(range(len(sizes)), accuracies,
                   color=COLORS['primary_blue'],
                   **LAYOUT['bars'])
    
    add_value_labels(ax, bars, format_str='{:.0f}%', offset=0.5, fontsize=25)  # Much larger
    
    ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold')  # Much larger
    ax.set_xlabel('Training Size', fontsize=20, fontweight='bold')  # Much larger

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], fontsize=18)  # Much larger
    ax.set_ylim(50, 100)  # Start at 50%
    ax.tick_params(axis='y', labelsize=18)  # Much larger
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"✓ Saved to: {output_path}")

def plot_multi_single(data, output_path):
    """Plot multi-single results (by test generator)."""
    
    if not data:
        print("✗ No multi-single data")
        return
    
    print("\n" + "="*60)
    print("Creating Multi-Single Plot")
    print("="*60)
    
    # Sort alphabetically
    generators = sorted(data.keys())
    gen_labels = [clean_generator_name(g) for g in generators]
    accuracies = [data[g] for g in generators]
    
    fig, ax = plt.subplots(figsize=(14, 7))  # Larger figure
    
    bars = ax.bar(range(len(generators)), accuracies,
                   color=COLORS['primary_blue'],
                   **LAYOUT['bars'])
    
    add_value_labels(ax, bars, format_str='{:.0f}%', offset=0.5, fontsize=25)
    
    ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_xlabel('Test Generator', fontsize=16, fontweight='bold')  # Increased from 12

    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=14)  # Increased
    ax.set_ylim(50, 100)
    ax.tick_params(axis='y', labelsize=14)  # Increased y-axis ticks
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"✓ Saved to: {output_path}")

def plot_single_multi(data, output_path):
    """Plot single-multi results (by train generator)."""
    
    if not data:
        print("✗ No single-multi data")
        return
    
    print("\n" + "="*60)
    print("Creating Single-Multi Plot")
    print("="*60)
    
    # Sort alphabetically
    generators = sorted(data.keys())
    gen_labels = [clean_generator_name(g) for g in generators]
    accuracies = [data[g] for g in generators]
    
    fig, ax = plt.subplots(figsize=(14, 7))  # Larger figure
    
    bars = ax.bar(range(len(generators)), accuracies,
                   color=COLORS['primary_orange'],
                   **LAYOUT['bars'])
    
    add_value_labels(ax, bars, format_str='{:.0f}%', offset=0.5, fontsize=25)
    
    ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_xlabel('Training Generator', fontsize=16, fontweight='bold')  # Increased from 12
   
    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=14)  # Increased
    ax.set_ylim(50, 100)
    ax.tick_params(axis='y', labelsize=14)  # Increased y-axis ticks
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"✓ Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='separate_plots',
                       help='Output directory')
    parser.add_argument('--mode', type=str,
                       choices=['multi-multi', 'multi-single', 'single-multi', 'all'],
                       default='all',
                       help='Which plot to generate')
    parser.add_argument('--exclude_sizes', type=int, nargs='+', default=None,
                       help='Training sizes to exclude (e.g., --exclude_sizes 714)')
    
    args = parser.parse_args()
    
    apply_thesis_style()
    
    # Load results
    multi_multi, multi_single, single_multi = load_tabpfn_results(args.results_dir)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    if args.mode in ['multi-multi', 'all']:
        plot_multi_multi(multi_multi, output_dir / 'tabpfn_multi_multi.png', 
                        exclude_sizes=args.exclude_sizes)
    
    if args.mode in ['multi-single', 'all']:
        plot_multi_single(multi_single, output_dir / 'tabpfn_multi_single.png')
    
    if args.mode in ['single-multi', 'all']:
        plot_single_multi(single_multi, output_dir / 'tabpfn_single_multi.png')
    
    print("\n" + "="*60)
    print(f"✓ All plots saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
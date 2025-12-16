#!/usr/bin/env python3
"""
Create confusion matrices for LATTE single-single results.
Shows how well each generator-trained model performs on each test generator.
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
        'Midjourney': 'MJ',  # Shortened for confusion matrix
        'stable_diffusion_v_1_4': 'SD v1.4',
        'stable_diffusion_v_1_5': 'SD v1.5',
        'VQDM': 'VQDM',
        'wukong': 'wukong'
    }
    return replacements.get(name, name)

def load_single_single_results(results_dir):
    """Load LATTE single-single results."""
    latte_dir = Path(results_dir) / 'results_latte' / 'single-single' / 'latte'
    
    if not latte_dir.exists():
        print(f"✗ LATTE single-single directory not found: {latte_dir}")
        return {}
    
    print(f"\nLoading LATTE single-single results from: {latte_dir}")
    
    # Define the 8 actual generators
    GENERATORS = [
        'ADM',
        'BigGAN',
        'glide',
        'Midjourney',
        'stable_diffusion_v_1_4',
        'stable_diffusion_v_1_5',
        'VQDM',
        'wukong'
    ]
    
    # Structure: {train_size: {train_gen: {test_gen: accuracy}}}
    results = defaultdict(lambda: defaultdict(dict))
    
    for json_file in sorted(latte_dir.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Parse filename: single-single_ADM_150_latte.json or single-single_ADM_to_BigGAN_150_latte.json
            stem = json_file.stem.replace('_latte', '')
            
            # Remove 'single-single_' prefix
            if stem.startswith('single-single_'):
                stem = stem[len('single-single_'):]
            
            parts = stem.split('_')
            
            # Strategy: Try all possible splits to find two valid generators
            train_gen = None
            test_gen = None
            train_size = None
            
            # Find training size (last numeric part)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].isdigit():
                    train_size = int(parts[i])
                    # Remove size from parts
                    remaining_parts = parts[:i]
                    break
            
            if not train_size:
                print(f"  ⚠ Could not parse size: {json_file.name}")
                continue
            
            # Check if there's a "_to_" separator
            if 'to' in remaining_parts:
                # Find 'to' index
                to_idx = remaining_parts.index('to')
                train_gen = '_'.join(remaining_parts[:to_idx])
                test_gen = '_'.join(remaining_parts[to_idx + 1:])
            else:
                # Same generator (no _to_)
                train_gen = '_'.join(remaining_parts)
                test_gen = train_gen
            
            # Validate generators
            if train_gen not in GENERATORS or test_gen not in GENERATORS:
                print(f"  ⚠ Invalid generators: train={train_gen}, test={test_gen} in {json_file.name}")
                continue
            
            # Get accuracy from nested structure
            accuracy = None
            if isinstance(data, dict):
                # LATTE format might have nested structure
                for key, value in data.items():
                    if isinstance(value, dict) and 'accuracy' in value:
                        accuracy = value['accuracy'] * 100
                        break
                else:
                    # Direct accuracy field
                    if 'accuracy' in data:
                        accuracy = data['accuracy'] * 100
            
            if accuracy is None:
                print(f"  ⚠ No accuracy found in: {json_file.name}")
                continue
            
            results[train_size][train_gen][test_gen] = accuracy
            print(f"  {json_file.name}: size={train_size}, train={train_gen}, test={test_gen}, acc={accuracy:.2f}%")
        
        except Exception as e:
            print(f"  ⚠ Error loading {json_file.name}: {e}")
            continue
    
    return results

def create_confusion_matrix(results, train_size, output_path):
    """Create confusion matrix for a specific training size."""
    
    if train_size not in results:
        print(f"✗ No results for training size {train_size}")
        return
    
    data = results[train_size]
    
    # Define the 8 generators in the order we want them
    GENERATORS = [
        'ADM',
        'BigGAN',
        'Midjourney',
        'VQDM',
        'glide',
        'stable_diffusion_v_1_4',
        'stable_diffusion_v_1_5',
        'wukong'
    ]
    
    # Filter to only use actual generators that have data
    all_train_gens = [g for g in GENERATORS if g in data]
    all_test_gens = GENERATORS.copy()
    
    print(f"\n✓ Using {len(all_train_gens)} training generators")
    print(f"✓ Using {len(all_test_gens)} test generators")
    
    # Create matrix
    n_train = len(all_train_gens)
    n_test = len(all_test_gens)
    matrix = np.zeros((n_train, n_test))
    
    for i, train_gen in enumerate(all_train_gens):
        for j, test_gen in enumerate(all_test_gens):
            matrix[i, j] = data[train_gen].get(test_gen, 0)
    
    # Clean labels
    train_labels = [clean_generator_name(g) for g in all_train_gens]
    test_labels = [clean_generator_name(g) for g in all_test_gens]
    
    # Create plot - larger for better readability
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use Greens colormap
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom green colormap
    colors_list = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_greens', colors_list, N=n_bins)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=50, vmax=100)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Accuracy (%)', fontsize=20, fontweight='bold', rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=18)
    
    # Set ticks and labels - LARGER
    ax.set_xticks(np.arange(n_test))
    ax.set_yticks(np.arange(n_train))
    ax.set_xticklabels(test_labels, fontsize=18, fontweight='bold')
    ax.set_yticklabels(train_labels, fontsize=18, fontweight='bold')
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations - MUCH LARGER
    for i in range(n_train):
        for j in range(n_test):
            value = matrix[i, j]
            
            # Skip if no data (value is 0)
            if value == 0:
                text_color = 'gray'
                ax.text(j, i, '0.0',
                       ha='center', va='center', 
                       fontsize=16,
                       color=text_color, alpha=0.3)
                continue
            
            # Determine text color based on background
            text_color = 'white' if value > 75 else 'black'
            
            # Bold and box for diagonal (same train/test generator)
            if all_train_gens[i] == all_test_gens[j]:
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center', 
                       fontsize=18, fontweight='bold',
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='none', 
                                edgecolor='black', linewidth=2.5))
            else:
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center', 
                       fontsize=17,
                       color=text_color,
                       fontweight='normal')
    
    # Labels and title - LARGER
    ax.set_xlabel('Test Generator', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('Training Generator', fontsize=22, fontweight='bold', labelpad=15)
    
    # Add subtle grid
    ax.set_xticks(np.arange(n_test + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_train + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2, alpha=0.4)
    ax.tick_params(which='minor', size=0)
    
    # Strong outer box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Statistics for Training Size {train_size}")
    print(f"{'='*60}")
    
    # Diagonal (same generator train/test)
    diagonal_vals = []
    for i in range(n_train):
        if all_train_gens[i] in all_test_gens:
            j = all_test_gens.index(all_train_gens[i])
            if matrix[i, j] > 0:
                diagonal_vals.append(matrix[i, j])
    
    if diagonal_vals:
        print(f"\nDiagonal (same train/test generator):")
        print(f"  Mean: {np.mean(diagonal_vals):.2f}%")
        print(f"  Min:  {np.min(diagonal_vals):.2f}%")
        print(f"  Max:  {np.max(diagonal_vals):.2f}%")
    
    # Off-diagonal (different generators)
    off_diagonal = []
    for i in range(n_train):
        for j in range(n_test):
            if all_train_gens[i] != all_test_gens[j] and matrix[i, j] > 0:
                off_diagonal.append(matrix[i, j])
    
    if off_diagonal:
        print(f"\nOff-diagonal (different train/test generators):")
        print(f"  Mean: {np.mean(off_diagonal):.2f}%")
        print(f"  Min:  {np.min(off_diagonal):.2f}%")
        print(f"  Max:  {np.max(off_diagonal):.2f}%")

def create_all_confusion_matrices(results, output_dir):
    """Create confusion matrices for all training sizes."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Creating LATTE Confusion Matrices for All Training Sizes")
    print(f"{'='*60}")
    
    for train_size in sorted(results.keys()):
        output_path = output_dir / f'confusion_matrix_latte_{train_size}.png'
        print(f"\n--- Training Size: {train_size} ---")
        create_confusion_matrix(results, train_size, output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Create confusion matrices for LATTE single-single results'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base directory containing results_latte')
    parser.add_argument('--output_dir', type=str, default='confusion_matrices_latte',
                       help='Output directory for confusion matrices')
    parser.add_argument('--train_size', type=int, default=None,
                       help='Specific training size to plot (default: all)')
    
    args = parser.parse_args()
    
    apply_thesis_style()
    
    print("="*60)
    print("LATTE Single→Single Confusion Matrices")
    print("="*60)
    
    # Load results
    results = load_single_single_results(args.results_dir)
    
    if not results:
        print("\n✗ No results loaded!")
        return
    
    print(f"\n✓ Loaded results for training sizes: {sorted(results.keys())}")
    
    # Create matrices
    if args.train_size:
        # Single training size
        output_path = Path(args.output_dir) / f'confusion_matrix_latte_{args.train_size}.png'
        create_confusion_matrix(results, args.train_size, output_path)
    else:
        # All training sizes
        create_all_confusion_matrices(results, args.output_dir)

if __name__ == '__main__':
    main()

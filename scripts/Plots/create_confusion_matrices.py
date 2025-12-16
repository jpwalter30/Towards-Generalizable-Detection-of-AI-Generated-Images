#!/usr/bin/env python3
"""
Create confusion matrices for TabPFN single-single results.
Shows how well each generator-trained model performs on each test generator.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse
import seaborn as sns
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
        'Midjourney': 'Midjourney',  # Shortened for confusion matrix
        'VQDM': 'VQDM',
        'glide': 'glide',
        'stable_diffusion_v_1_4': 'Stable Diffusion v1.4',
        'stable_diffusion_v_1_5': 'Stable Diffusion v1.5',
        'wukong': 'wukong'
    }
    return replacements.get(name, name)

def load_single_single_results(results_dir):
    """Load TabPFN single-single results."""
    tabpfn_dir = Path(results_dir) / 'results_tabpfn' / 'single-single' / 'tabpfn'
    
    if not tabpfn_dir.exists():
        print(f"✗ TabPFN single-single directory not found: {tabpfn_dir}")
        return {}
    
    print(f"\nLoading TabPFN single-single results from: {tabpfn_dir}")
    
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
    
    for json_file in sorted(tabpfn_dir.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Parse filename: single-single_714_tabpfn_wukong_wukong.json
            # Format: single-single_{SIZE}_tabpfn_{TRAIN_GEN}_{TEST_GEN}.json
            
            stem = json_file.stem
            parts = stem.split('_')
            
            # Find training size and tabpfn index
            train_size = None
            tabpfn_idx = None
            
            for i, part in enumerate(parts):
                if part == 'tabpfn':
                    tabpfn_idx = i
                elif part.isdigit() and train_size is None:
                    train_size = int(part)
            
            if not train_size or tabpfn_idx is None:
                print(f"  ⚠ Could not parse size/tabpfn: {json_file.name}")
                continue
            
            # Everything after tabpfn are generator names
            gen_parts = parts[tabpfn_idx + 1:]
            
            if len(gen_parts) < 2:
                print(f"  ⚠ Not enough parts: {json_file.name}")
                continue
            
            # Try to match against known generators
            # Strategy: Try all possible splits and see which gives two valid generators
            train_gen = None
            test_gen = None
            
            for split_point in range(1, len(gen_parts)):
                potential_train = '_'.join(gen_parts[:split_point])
                potential_test = '_'.join(gen_parts[split_point:])
                
                if potential_train in GENERATORS and potential_test in GENERATORS:
                    train_gen = potential_train
                    test_gen = potential_test
                    break
            
            if not train_gen or not test_gen:
                print(f"  ⚠ Could not match generators: {json_file.name}")
                print(f"     Parts after tabpfn: {gen_parts}")
                continue
            
            # Get accuracy
            if isinstance(data, dict):
                if 'metrics' in data:
                    accuracy = data['metrics'].get('accuracy', 0) * 100
                else:
                    accuracy = data.get('accuracy', 0) * 100
            else:
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
    
    # Use Greens colormap (similar to the reference image)
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom green colormap - exactly like reference
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
    ax.set_xticklabels(test_labels, fontsize=18, fontweight='bold')  # Increased
    ax.set_yticklabels(train_labels, fontsize=18, fontweight='bold')  # Increased
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations - MUCH LARGER like reference image
    for i in range(n_train):
        for j in range(n_test):
            value = matrix[i, j]
            
            # Skip if no data (value is 0)
            if value == 0:
                text_color = 'gray'
                ax.text(j, i, '0.0',
                       ha='center', va='center', 
                       fontsize=16,  # Larger even for zeros
                       color=text_color, alpha=0.3)
                continue
            
            # Determine text color based on background
            text_color = 'white' if value > 75 else 'black'
            
            # Bold and box for diagonal (same train/test generator)
            if all_train_gens[i] == all_test_gens[j]:
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center', 
                       fontsize=23, fontweight='bold',  # Increased by 5
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='none', 
                                edgecolor='black', linewidth=2.5))
            else:
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center', 
                       fontsize=22,  # Increased by 5
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
    print("Creating Confusion Matrices for All Training Sizes")
    print(f"{'='*60}")
    
    for train_size in sorted(results.keys()):
        output_path = output_dir / f'confusion_matrix_{train_size}.png'
        print(f"\n--- Training Size: {train_size} ---")
        create_confusion_matrix(results, train_size, output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Create confusion matrices for TabPFN single-single results'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base directory containing results_tabpfn')
    parser.add_argument('--output_dir', type=str, default='confusion_matrices',
                       help='Output directory for confusion matrices')
    parser.add_argument('--train_size', type=int, default=None,
                       help='Specific training size to plot (default: all)')
    
    args = parser.parse_args()
    
    apply_thesis_style()
    
    print("="*60)
    print("TabPFN Single→Single Confusion Matrices")
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
        output_path = Path(args.output_dir) / f'confusion_matrix_{args.train_size}.png'
        create_confusion_matrix(results, args.train_size, output_path)
    else:
        # All training sizes
        create_all_confusion_matrices(results, args.output_dir)

if __name__ == '__main__':
    main()
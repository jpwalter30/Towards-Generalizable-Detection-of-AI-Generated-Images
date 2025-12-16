#!/usr/bin/env python3
"""
Create difference heatmap comparing LATTE vs TabPFN in Single-Single mode.
Shows TabPFN - LATTE accuracy difference.
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
        'Midjourney': 'Midjourney',  # Shortened for confusion matrix
        'VQDM': 'VQDM',
        'glide': 'glide',
        'stable_diffusion_v_1_4': 'SD v1.4',
        'stable_diffusion_v_1_5': 'SD v1.5',
        'wukong': 'wukong'
    }
    return replacements.get(name, name)

def load_tabpfn_single_single(results_dir):
    """Load TabPFN single-single results."""
    tabpfn_dir = Path(results_dir) / 'results_tabpfn' / 'single-single' / 'tabpfn'
    
    if not tabpfn_dir.exists():
        print(f"✗ TabPFN directory not found: {tabpfn_dir}")
        return {}
    
    print(f"\nLoading TabPFN single-single results from: {tabpfn_dir}")
    
    GENERATORS = ['ADM', 'BigGAN', 'glide', 'Midjourney', 
                  'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
    
    results = defaultdict(lambda: defaultdict(dict))
    loaded_count = 0
    
    for json_file in sorted(tabpfn_dir.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            stem = json_file.stem
            parts = stem.split('_')
            
            train_size = None
            tabpfn_idx = None
            
            for i, part in enumerate(parts):
                if part == 'tabpfn':
                    tabpfn_idx = i
                elif part.isdigit() and train_size is None:
                    train_size = int(part)
            
            if not train_size or tabpfn_idx is None:
                continue
            
            gen_parts = parts[tabpfn_idx + 1:]
            
            # Try all possible splits
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
                continue
            
            if isinstance(data, dict):
                if 'metrics' in data:
                    accuracy = data['metrics'].get('accuracy', 0) * 100
                else:
                    accuracy = data.get('accuracy', 0) * 100
            else:
                continue
            
            results[train_size][train_gen][test_gen] = accuracy
            loaded_count += 1
        
        except Exception as e:
            continue
    
    print(f"✓ Loaded {loaded_count} TabPFN results")
    return results

def load_latte_single_single(results_dir):
    """Load LATTE single-single results."""
    latte_dir = Path(results_dir) / 'results_latte' / 'single-single' / 'latte'
    
    if not latte_dir.exists():
        print(f"✗ LATTE directory not found: {latte_dir}")
        return {}
    
    print(f"\nLoading LATTE single-single results from: {latte_dir}")
    
    GENERATORS = ['ADM', 'BigGAN', 'glide', 'Midjourney',
                  'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
    
    results = defaultdict(lambda: defaultdict(dict))
    loaded_count = 0
    
    for json_file in sorted(latte_dir.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            stem = json_file.stem.replace('_latte', '')
            
            if stem.startswith('single-single_'):
                stem = stem[len('single-single_'):]
            
            parts = stem.split('_')
            
            train_size = None
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].isdigit():
                    train_size = int(parts[i])
                    remaining_parts = parts[:i]
                    break
            
            if not train_size:
                continue
            
            if 'to' in remaining_parts:
                to_idx = remaining_parts.index('to')
                train_gen = '_'.join(remaining_parts[:to_idx])
                test_gen = '_'.join(remaining_parts[to_idx + 1:])
            else:
                train_gen = '_'.join(remaining_parts)
                test_gen = train_gen
            
            if train_gen not in GENERATORS or test_gen not in GENERATORS:
                continue
            
            accuracy = None
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and 'accuracy' in value:
                        accuracy = value['accuracy'] * 100
                        break
                else:
                    if 'accuracy' in data:
                        accuracy = data['accuracy'] * 100
            
            if accuracy is None:
                continue
            
            results[train_size][train_gen][test_gen] = accuracy
            loaded_count += 1
        
        except Exception as e:
            continue
    
    print(f"✓ Loaded {loaded_count} LATTE results")
    return results

def plot_difference_heatmap(tabpfn_results, latte_results, train_size, output_path, debug=False):
    """Heatmap showing TabPFN - LATTE difference."""
    
    GENERATORS = ['ADM', 'BigGAN', 'Midjourney', 'VQDM', 'glide',
                  'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'wukong']
    
    # Create difference matrix
    n = len(GENERATORS)
    diff_matrix = np.zeros((n, n))
    tabpfn_matrix = np.zeros((n, n))
    latte_matrix = np.zeros((n, n))
    
    print(f"\n{'='*80}")
    print(f"Creating Difference Matrix for Training Size {train_size}")
    print(f"{'='*80}")
    
    for i, train_gen in enumerate(GENERATORS):
        for j, test_gen in enumerate(GENERATORS):
            tabpfn_acc = tabpfn_results[train_size].get(train_gen, {}).get(test_gen, 0)
            latte_acc = latte_results[train_size].get(train_gen, {}).get(test_gen, 0)
            diff = tabpfn_acc - latte_acc
            
            diff_matrix[i, j] = diff
            tabpfn_matrix[i, j] = tabpfn_acc
            latte_matrix[i, j] = latte_acc
            
            if debug and (i == j or abs(diff) > 20):  # Print diagonal and large differences
                print(f"{train_gen:30s} → {test_gen:30s}: TabPFN={tabpfn_acc:6.2f}%, LATTE={latte_acc:6.2f}%, Diff={diff:+7.2f}%")
    
    gen_labels = [clean_generator_name(g) for g in GENERATORS]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use diverging colormap (blue for TabPFN better, red for LATTE better)
    # RdBu: red=negative, blue=positive
    # So: TabPFN - LATTE: positive (blue) = TabPFN better, negative (red) = LATTE better
    im = ax.imshow(diff_matrix, cmap='RdBu', aspect='auto', vmin=-30, vmax=30)
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Accuracy Difference (TabPFN - LATTE) [%]', 
                  fontsize=18, fontweight='bold', rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=16)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(gen_labels, fontsize=18, fontweight='bold', rotation=45, ha='right')  # Match CM
    ax.set_yticklabels(gen_labels, fontsize=18, fontweight='bold')  # Match CM
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            value = diff_matrix[i, j]
            
            # Determine text color based on background
            text_color = 'white' if abs(value) > 15 else 'black'
            
            if i == j:  # Diagonal
                ax.text(j, i, f'{value:+.1f}',
                       ha='center', va='center',
                       fontsize=26, fontweight='bold',  # Further increased for thesis
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='none',
                                edgecolor='black', linewidth=2.5))
            else:
                ax.text(j, i, f'{value:+.1f}',
                       ha='center', va='center',
                       fontsize=25, color=text_color)  # Further increased for thesis
    
    ax.set_xlabel('Test Generator', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('Training Generator', fontsize=22, fontweight='bold', labelpad=15)
    # Title removed for thesis - matches confusion matrix style
    
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2, alpha=0.4)
    ax.tick_params(which='minor', size=0)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"\n✓ Saved difference heatmap to: {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")
    
    # Diagonal statistics
    diagonal_diffs = [diff_matrix[i, i] for i in range(n)]
    print(f"\nDiagonal (In-Domain):")
    print(f"  Mean difference: {np.mean(diagonal_diffs):+.2f}%")
    print(f"  Min difference:  {np.min(diagonal_diffs):+.2f}%")
    print(f"  Max difference:  {np.max(diagonal_diffs):+.2f}%")
    
    # Off-diagonal statistics
    off_diagonal_diffs = [diff_matrix[i, j] for i in range(n) for j in range(n) if i != j]
    print(f"\nOff-Diagonal (Cross-Domain):")
    print(f"  Mean difference: {np.mean(off_diagonal_diffs):+.2f}%")
    print(f"  Min difference:  {np.min(off_diagonal_diffs):+.2f}%")
    print(f"  Max difference:  {np.max(off_diagonal_diffs):+.2f}%")
    
    # Overall statistics
    tabpfn_better = np.sum(diff_matrix > 0)
    latte_better = np.sum(diff_matrix < 0)
    equal = np.sum(diff_matrix == 0)
    
    print(f"\nOverall (all {n*n} combinations):")
    print(f"  TabPFN better:  {tabpfn_better} ({tabpfn_better/(n*n)*100:.1f}%)")
    print(f"  LATTE better:   {latte_better} ({latte_better/(n*n)*100:.1f}%)")
    print(f"  Equal/Missing:  {equal}")

def main():
    parser = argparse.ArgumentParser(
        description='Create difference heatmap: TabPFN vs LATTE Single-Single'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base directory containing results')
    parser.add_argument('--output_dir', type=str, default='comparison_single_single',
                       help='Output directory')
    parser.add_argument('--train_size', type=int, default=625,
                       help='Training size (default: 625)')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information')
    
    args = parser.parse_args()
    
    apply_thesis_style()
    
    print("="*80)
    print("TabPFN vs LATTE Difference Heatmap")
    print("="*80)
    
    # Load results
    tabpfn_results = load_tabpfn_single_single(args.results_dir)
    latte_results = load_latte_single_single(args.results_dir)
    
    if not tabpfn_results or not latte_results:
        print("\n✗ Failed to load results")
        return
    
    if args.train_size not in tabpfn_results:
        print(f"\n✗ TabPFN: Training size {args.train_size} not found")
        print(f"  Available sizes: {sorted(tabpfn_results.keys())}")
        return
    
    if args.train_size not in latte_results:
        print(f"\n✗ LATTE: Training size {args.train_size} not found")
        print(f"  Available sizes: {sorted(latte_results.keys())}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create difference heatmap
    output_path = output_dir / f'difference_heatmap_{args.train_size}.png'
    plot_difference_heatmap(tabpfn_results, latte_results, args.train_size, 
                           output_path, debug=args.debug)
    
    print(f"\n{'='*80}")
    print("Complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
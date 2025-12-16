#!/usr/bin/env python3
"""
PCA Visualization using combined DINOv3 Features
All features are in one file, generators identified by gens.npy
Uses consistent scientific corporate design
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import argparse
import sys

# Import thesis style
sys.path.insert(0, str(Path(__file__).parent))
from thesis_style import apply_thesis_style, COLORS, save_figure

def load_combined_features(feature_dir, use_precomputed_pca=False):
    """Load combined DINOv3 features from single files.
    
    Args:
        feature_dir: Directory containing feature files
        use_precomputed_pca: If True, load X_cls_pca.npy instead of X_cls.npy
    """
    feature_path = Path(feature_dir)
    
    print(f"Loading features from: {feature_path}")
    
    # Choose which feature file to load
    if use_precomputed_pca:
        X_file = feature_path / "X_cls_pca.npy"
        print("  Using pre-computed PCA features (X_cls_pca.npy)")
    else:
        X_file = feature_path / "X_cls.npy"
        print("  Using original features (X_cls.npy) - PCA will be computed")
    
    y_file = feature_path / "y.npy"
    gens_file = feature_path / "gens.npy"
    
    if not X_file.exists() or not y_file.exists() or not gens_file.exists():
        raise FileNotFoundError(
            f"Missing required files in {feature_path}\n"
            f"Expected: {X_file.name}, y.npy, gens.npy"
        )
    
    print(f"  Loading X_cls.npy...")
    X = np.load(X_file)
    
    print(f"  Loading y.npy...")
    y = np.load(y_file)
    
    print(f"  Loading gens.npy...")
    gens = np.load(gens_file, allow_pickle=True)
    
    print(f"  Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  Labels: Real={np.sum(y == 0)}, Fake={np.sum(y == 1)}")
    print(f"  Unique generators: {len(np.unique(gens))}")
    
    return X, y, gens

def get_generator_name(gen_id):
    """Map generator ID to name."""
    # Based on typical GenImage ordering
    gen_map = {
        0: "ADM",
        1: "BigGAN", 
        2: "glide",
        3: "Midjourney",
        4: "Stable Diffusion v1.4",
        5: "Stable Diffusion v1.5",
        6: "VQDM",
        7: "wukong"
    }
    return gen_map.get(gen_id, f"Gen_{gen_id}")

def clean_generator_name(name):
    """Clean generator names from gens.npy strings."""
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

def create_pca_plot(X, y, title, ax=None):
    """Create a single PCA scatter plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Fit PCA
    print(f"    Fitting PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    real_mask = y == 0
    fake_mask = y == 1
    
    ax.scatter(X_pca[real_mask, 0], X_pca[real_mask, 1], 
               c='#1f77b4', alpha=0.5, s=3, label='real images', rasterized=True)
    ax.scatter(X_pca[fake_mask, 0], X_pca[fake_mask, 1], 
               c='#ff7f0e', alpha=0.5, s=3, label='fake', rasterized=True)
    
    ax.set_title(title, fontsize=10, fontweight='normal')
    ax.set_xlabel(f'PC1', fontsize=9)
    ax.set_ylabel(f'PC2', fontsize=9)
    ax.legend(markerscale=3, fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Remove some ticks for cleaner look
    ax.locator_params(axis='both', nbins=5)
    
    return pca

def create_generator_grid(X, y, gens, output_path, max_samples=2500):
    """Create a grid of PCA plots - real images (same in all) + fake per generator - THESIS STYLE."""
    
    # Get unique generators
    unique_gens = sorted(np.unique(gens[y == 1]))  # Only from fake images
    n_gens = len(unique_gens)
    
    print(f"\nFound {n_gens} generators in fake images:")
    for gen_name in unique_gens:
        count = np.sum((gens == gen_name) & (y == 1))
        print(f"  {gen_name}: {count} fake samples")
    
    # Also count real images
    n_real = np.sum(y == 0)
    print(f"\nReal images: {n_real}")
    
    # Subsample real images (same for all plots)
    real_mask = (y == 0)
    real_idx = np.where(real_mask)[0]
    max_real_samples = 8000
    if len(real_idx) > max_real_samples:
        print(f"Subsampling real images to {max_real_samples}...")
        np.random.seed(42)
        real_idx = np.random.choice(real_idx, size=max_real_samples, replace=False)
    
    print(f"Using {len(real_idx)} real samples (shown in all plots)")
    
    # Fit PCA on ALL data first
    print("\nFitting global PCA on all data...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    
    # Setup grid: REAL alone in first row, then generators in 2 rows
    # Layout (3x3):
    # [REAL] [G1] [G2]
    # [G3] [G4] [G5]
    # [G6] [G7] [G8]
    
    n_cols = 3
    n_rows = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18), 
                            sharex=True, sharey=True)
    axes = axes.flatten()
    
    print("\n" + "="*60)
    print("Creating PCA Grid")
    print("="*60)
    
    # REAL IMAGES PLOT - First position (0)
    print(f"\n[1/{n_gens+1}] Real images only...")
    ax = axes[0]
    ax.scatter(X_pca[real_idx, 0], X_pca[real_idx, 1],
              c=COLORS['primary_blue'], alpha=0.3, s=3, label='real images', rasterized=True)
    ax.set_title('real images', fontsize=24, fontweight='bold')  # Larger for thesis
    # No legend on individual plots
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=14)  # Increased from 11
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Hide positions 1 and 2 in first row (empty after REAL)
    # No longer needed - using 3x3 grid now
    
    # GENERATOR PLOTS - Positions 1-8
    for idx, gen_name in enumerate(unique_gens):
        plot_idx = idx + 1  # Start after REAL (position 0)
        gen_label = clean_generator_name(gen_name)  # Clean the name for display
        print(f"\n[{idx+2}/{n_gens+1}] Processing {gen_label}...")
        ax = axes[plot_idx]
        
        # Plot real images (gray background) - SAME IN ALL PLOTS
        ax.scatter(X_pca[real_idx, 0], X_pca[real_idx, 1],
                  c=COLORS['bg_real'], alpha=0.15, s=3, label='real images', rasterized=True)
        
        # Get fake samples for this generator
        fake_mask = (gens == gen_name) & (y == 1)
        fake_idx = np.where(fake_mask)[0]
        
        print(f"  Total fake samples: {len(fake_idx)}")
        
        # Subsample fakes if needed
        if max_samples and len(fake_idx) > max_samples:
            print(f"  Limiting to {max_samples} fake samples...")
            fake_idx = np.random.choice(fake_idx, max_samples, replace=False)
        
        print(f"  Plotting {len(fake_idx)} fake samples")
        
        # Plot fake images for this generator
        if len(fake_idx) > 0:
            ax.scatter(X_pca[fake_idx, 0], X_pca[fake_idx, 1],
                      c=COLORS['bg_fake'], alpha=0.5, s=3, label='fake', rasterized=True)
        
        ax.set_title(gen_label, fontsize=24, fontweight='bold')  # Larger for thesis
        # No legend on individual plots
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=14)  # Increased from 11
        ax.set_aspect('equal', adjustable='box')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add main title
    # Title removed for thesis
    
    # Add shared axis labels
    fig.text(0.5, 0.02, 'PC1', ha='center', fontsize=22, fontweight='bold')  # Larger for thesis
    fig.text(0.02, 0.5, 'PC2', va='center', rotation='vertical', fontsize=22, fontweight='bold')  # Larger for thesis
    
    # Legend removed for thesis
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])  # Adjusted for no legend
    
    # Save
    save_figure(fig, output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    return fig

def create_overlay_plot(X, y, gens, output_path, max_samples=2500):
    """Create a single PCA plot with all generators overlaid."""
    
    print("\n" + "="*60)
    print("Creating Overlay PCA Plot")
    print("="*60)
    
    unique_gens = np.unique(gens)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_gens)))
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Subsample for performance if needed
    if max_samples and len(X) > max_samples * len(unique_gens):
        print(f"Subsampling to {max_samples} per generator...")
        indices = []
        for gen_id in unique_gens:
            gen_mask = gens == gen_id
            gen_indices = np.where(gen_mask)[0]
            if len(gen_indices) > max_samples:
                gen_indices = np.random.choice(gen_indices, max_samples, replace=False)
            indices.extend(gen_indices)
        
        indices = np.array(indices)
        X_sub = X[indices]
        y_sub = y[indices]
        gens_sub = gens[indices]
    else:
        X_sub = X
        y_sub = y
        gens_sub = gens
    
    print(f"Total samples for PCA: {len(X_sub)}")
    
    # Fit PCA on all data
    print("Fitting PCA on all generators combined...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)
    
    # Plot each generator (only fake images)
    for idx, gen_id in enumerate(sorted(unique_gens)):
        gen_name = get_generator_name(gen_id)
        
        # Only plot fake images
        mask = (gens_sub == gen_id) & (y_sub == 1)
        
        if np.any(mask):
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[idx]], alpha=0.5, s=5, label=gen_name, 
                      rasterized=True)
    
    ax.set_title('PCA - All Generators Overlay (Fake Images Only)', fontsize=16)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=14)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=14)
    ax.legend(markerscale=3, loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save
    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved to: {output_file}")
    
    return fig

def create_real_vs_all_fake_plot(X, y, gens, output_path, max_samples=10000):
    """Create PCA showing real images vs all fake generators combined."""
    
    print("\n" + "="*60)
    print("Creating Real vs All-Fake PCA Plot")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Subsample if needed
    if max_samples and len(X) > max_samples:
        print(f"Subsampling to {max_samples} total samples...")
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y
    
    print(f"Total samples: {len(X_sub)}")
    print(f"  Real: {np.sum(y_sub == 0)}, Fake: {np.sum(y_sub == 1)}")
    
    # Fit PCA
    print("Fitting PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)
    
    # Plot
    real_mask = y_sub == 0
    fake_mask = y_sub == 1
    
    ax.scatter(X_pca[real_mask, 0], X_pca[real_mask, 1],
              c='blue', alpha=0.3, s=3, label='Real (all)', rasterized=True)
    ax.scatter(X_pca[fake_mask, 0], X_pca[fake_mask, 1],
              c='red', alpha=0.3, s=3, label='Fake (all generators)', rasterized=True)
    
    ax.set_title('PCA - Real vs All Fake Images Combined', fontsize=16)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=14)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=14)
    ax.legend(markerscale=5, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save
    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved to: {output_file}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Create PCA visualizations from combined DINOv3 features')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Directory containing X_cls.npy, y.npy, gens.npy')
    parser.add_argument('--output_dir', type=str, default='pca_plots',
                       help='Output directory for plots')
    parser.add_argument('--max_samples', type=int, default=2500,
                       help='Maximum samples per generator for grid/overlay (default: 2500)')
    parser.add_argument('--mode', type=str, 
                       choices=['grid', 'overlay', 'combined', 'all'], 
                       default='all',
                       help='Plot mode: grid, overlay, combined (real vs all fake), or all')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for subsampling')
    
    args = parser.parse_args()
    
    # Apply thesis style
    apply_thesis_style()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("\n" + "="*60)
    print("DINOv3 PCA Visualization")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples per generator: {args.max_samples}")
    print(f"Mode: {args.mode}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Load combined features
    X, y, gens = load_combined_features(args.base_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    if args.mode in ['grid', 'all']:
        create_generator_grid(
            X, y, gens,
            output_dir / 'pca_grid.png',
            max_samples=args.max_samples
        )
    
    if args.mode in ['overlay', 'all']:
        create_overlay_plot(
            X, y, gens,
            output_dir / 'pca_overlay.png',
            max_samples=args.max_samples
        )
    
    if args.mode in ['combined', 'all']:
        create_real_vs_all_fake_plot(
            X, y, gens,
            output_dir / 'pca_real_vs_fake.png',
            max_samples=10000
        )
    
    print("\n" + "="*60)
    print("✓ Done!")
    print("="*60)
    print(f"\nOutput files in: {args.output_dir}")
    if args.mode == 'all':
        print("  - pca_grid.png")
        print("  - pca_overlay.png")
        print("  - pca_real_vs_fake.png")

if __name__ == "__main__":
    main()
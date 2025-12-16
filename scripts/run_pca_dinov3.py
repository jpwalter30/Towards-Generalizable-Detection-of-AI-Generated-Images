#!/usr/bin/env python3
"""PCA Dimensionality Reduction for DINOv3 Features.

Applies Incremental PCA to reduce DINOv3 feature dimensions from 768 to a 
specified number of components. Uses a two-pass approach for memory-efficient
processing of large feature matrices.

"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from joblib import dump


def perform_incremental_pca(
    input_path: str,
    output_path: str,
    n_components: int = 500,
    batch_size: int = 50_000,
    save_model: bool = True,
):
    """Perform two-pass Incremental PCA on feature matrix.
    
    The two-pass approach:
        1. First pass: Fit PCA model using partial_fit on batches
        2. Second pass: Transform features and save to disk
    
    This approach enables processing of datasets larger than available RAM.
    
    Args:
        input_path: Path to input feature file (e.g., X_cls.npy).
            Expected shape: (num_samples, num_features).
        output_path: Path to output file for reduced features (e.g., X_cls_pca.npy).
            Will be saved as a standard .npy file.
        n_components: Number of principal components to retain (default: 500).
        batch_size: Batch size for incremental processing (default: 50000).
        save_model: Whether to save the fitted PCA model as pca_model.joblib (default: True).
        
    Raises:
        ValueError: If n_components exceeds feature dimensions.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load feature matrix using memory mapping to avoid loading entire array
    print(f"[INFO] Loading features from: {input_path}")
    features = np.load(input_path, mmap_mode="r")
    num_samples, num_features = features.shape
    print(f"[INFO] Feature matrix shape: {features.shape}")
    print(f"       Samples: {num_samples}")
    print(f"       Features: {num_features}")

    # Validate n_components
    if n_components > num_features:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed feature dimensions ({num_features}). "
            f"Please set n_components ≤ {num_features}."
        )

    # Initialize Incremental PCA
    print(f"\n[INFO] Initializing Incremental PCA with {n_components} components")
    pca_model = IncrementalPCA(n_components=n_components)

      
    # Pass 1: Fit PCA Model
      
    
    print("\n[PASS 1] Fitting PCA model using partial_fit...")
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_features = features[batch_start:batch_end]
        
        pca_model.partial_fit(batch_features)
        print(f"  Processed batch [{batch_start}:{batch_end}] ({batch_end}/{num_samples} samples)")

    print(f"[PASS 1] PCA model fitting completed")
    print(f"         Explained variance ratio: {pca_model.explained_variance_ratio_.sum():.4f}")

      
    # Pass 2: Transform Features
      
    
    # Create output directory
    output_dir = output_path.parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Allocate output array
    print(f"\n[PASS 2] Allocating output array: ({num_samples}, {n_components})")
    reduced_features = np.empty((num_samples, n_components), dtype=np.float32)

    print("[PASS 2] Transforming features...")
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_features = features[batch_start:batch_end]
        
        # Transform and store
        batch_reduced = pca_model.transform(batch_features).astype(np.float32)
        reduced_features[batch_start:batch_end] = batch_reduced
        
        print(f"  Transformed batch [{batch_start}:{batch_end}] ({batch_end}/{num_samples} samples)")

      
    # Save Results
      
    
    print(f"\n[INFO] Saving reduced features to: {output_path}")
    np.save(output_path, reduced_features)
    print("[SUCCESS] PCA-reduced features saved successfully")

    # Optionally save PCA model
    if save_model:
        model_path = output_dir / "pca_model.joblib"
        dump(pca_model, model_path)
        print(f"[INFO] PCA model saved to: {model_path}")

    # Print final summary
    print(f"\n{'='*70}")
    print(f"PCA DIMENSIONALITY REDUCTION COMPLETED")
    print(f"{'='*70}")
    print(f"Input:  {num_features} dimensions")
    print(f"Output: {n_components} dimensions")
    print(f"Samples: {num_samples}")
    print(f"Explained variance: {pca_model.explained_variance_ratio_.sum():.2%}")
    print(f"{'='*70}\n")


def main():
    """Parse arguments and execute PCA dimensionality reduction."""
    
    parser = argparse.ArgumentParser(
        description="Reduce DINOv3 feature dimensions using Incremental PCA"
    )
    parser.add_argument(
        "--x_path",
        type=str,
        default="X_cls.npy",
        help="Path to input feature file (default: X_cls.npy)"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="X_cls_pca.npy",
        help="Path to output file (default: X_cls_pca.npy)"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=500,
        help="Number of principal components (default: 500)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50_000,
        help="Batch size for incremental processing (default: 50000)"
    )
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="Do not save the fitted PCA model"
    )

    args = parser.parse_args()

    perform_incremental_pca(
        input_path=args.x_path,
        output_path=args.out_path,
        n_components=args.n_components,
        batch_size=args.batch_size,
        save_model=not args.no_save_model,
    )


if __name__ == "__main__":
    main()
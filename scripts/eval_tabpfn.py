#!/usr/bin/env python3
"""TabPFN Classifier Evaluation Script.

Evaluates TabPFN classifier on DINOv3 features for AI-generated image detection.
Supports multiple training and testing modes for cross-generator evaluation.

"""

import argparse
import json
from pathlib import Path
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.decomposition import PCA as SKPCA
from tabpfn import TabPFNClassifier


# Path Configuration

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = BASE_DIR / "features"
RESULTS_ROOT = BASE_DIR / Path(os.environ.get("RESULTS_ROOT", "results_tabpfn"))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# Verify required files exist
REQUIRED_FILES = [
    "X_cls_pca.npy",
    "y.npy",
    "paths.npy",
    "gens.npy",
    "splits.npy"
]
missing_files = [f for f in REQUIRED_FILES if not (FEATURE_DIR / f).exists()]
if missing_files:
    raise FileNotFoundError(
        f"Missing required files in {FEATURE_DIR}:\n  "
        + "\n  ".join(missing_files)
    )


# Load Feature Arrays and Metadata

X = np.load(FEATURE_DIR / "X_cls_pca.npy", allow_pickle=True)
y = np.load(FEATURE_DIR / "y.npy", allow_pickle=True).astype(int)
image_paths = np.load(FEATURE_DIR / "paths.npy", allow_pickle=True)
generators = np.load(FEATURE_DIR / "gens.npy", allow_pickle=True)
splits = np.load(FEATURE_DIR / "splits.npy", allow_pickle=True)

# Create Split-Specific Indices
 
# Split masks
train_mask = splits == "train"
val_mask = splits == "val"

# Real/fake indices per split
real_indices_train = np.where((y == 0) & train_mask)[0]
real_indices_val = np.where((y == 0) & val_mask)[0]
fake_indices_train = np.where((y == 1) & train_mask)[0]
fake_indices_val = np.where((y == 1) & val_mask)[0]

# Unique generator names (excluding special values)
unique_generators = sorted(
    g for g in np.unique(generators[y == 1]) 
    if g not in ("", None, "real", "unknown")
)

# Generator-specific fake indices
fake_indices_by_generator_train = {
    gen: np.where((generators == gen) & (y == 1) & train_mask)[0] 
    for gen in unique_generators
}
fake_indices_by_generator_val = {
    gen: np.where((generators == gen) & (y == 1) & val_mask)[0] 
    for gen in unique_generators
}

 
# Utility Functions
 

def parse_evaluation_mode(mode: str):
    """Parse and validate evaluation mode string.
    
    Args:
        mode: Mode string (e.g., "multi-multi", "single->multi").
        
    Returns:
        A tuple (train_mode, test_mode, normalized_mode).
        
    Raises:
        ValueError: If mode is invalid.
    """
    normalized = mode.strip().lower().replace("->", "-")
    train_mode, test_mode = normalized.split("-")
    
    if train_mode not in {"multi", "single"} or test_mode not in {"multi", "single"}:
        raise ValueError(
            "Invalid mode. Must be one of: multi-multi, multi-single, "
            "single-multi, single-single"
        )
        
    return train_mode, test_mode, normalized


def compute_classification_metrics(y_true, y_pred, y_proba):
    """Compute standard classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Prediction probabilities.
        
    Returns:
        Dictionary containing accuracy, precision, recall, F1, ROC-AUC, and sample count.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "n_test": int(len(y_true)),
    }


def create_train_test_split(
    train_size,
    mode,
    train_generator=None,
    test_generator=None,
    random_seed=42,
    max_test_fakes=5000,
    max_test_reals=5000,
):
    """Create train and test indices based on evaluation mode.
    
    Training samples are drawn exclusively from the train split,
    test samples exclusively from the validation split.
    
    Args:
        train_size: Number of samples per generator for training.
        mode: Evaluation mode (e.g., "multi-multi").
        train_generator: Generator name for single-generator training (optional).
        test_generator: Generator name for single-generator testing (optional).
        random_seed: Random seed for reproducibility.
        max_test_fakes: Maximum number of fake samples in test set.
        max_test_reals: Maximum number of real samples in test set.
        
    Returns:
        A tuple (train_indices, test_indices).
        
    Raises:
        ValueError: If insufficient samples are available.
    """
    rng = np.random.RandomState(random_seed)
    train_mode, test_mode, _ = parse_evaluation_mode(mode)

    # ========================================================================
    # Build Training Set (from train split)
    # ========================================================================
    
    train_indices = []
    
    if train_mode == "multi":
        # Multi-generator training: equal samples per generator
        samples_per_gen = min(
            [train_size] + [len(fake_indices_by_generator_train[g]) for g in unique_generators]
        )
        if samples_per_gen < train_size:
            print(
                f"[WARNING] Reduced samples per generator from {train_size} to "
                f"{samples_per_gen} (limited availability in train split)"
            )
        
        for gen in unique_generators:
            if len(fake_indices_by_generator_train[gen]) == 0:
                continue
            train_indices.extend(
                rng.choice(
                    fake_indices_by_generator_train[gen], 
                    samples_per_gen, 
                    replace=False
                )
            )
        
        num_fake_samples = len(train_indices)
        if len(real_indices_train) < num_fake_samples:
            raise ValueError(
                f"Insufficient real samples in train split "
                f"(need {num_fake_samples}, have {len(real_indices_train)})"
            )
        train_indices.extend(
            rng.choice(real_indices_train, num_fake_samples, replace=False)
        )
        
    else:  # Single-generator training
        if not train_generator:
            raise ValueError("train_generator required for single-generator training modes")
        if train_generator not in fake_indices_by_generator_train:
            raise ValueError(f"Unknown generator: '{train_generator}'")
        if len(fake_indices_by_generator_train[train_generator]) < train_size:
            raise ValueError(
                f"Insufficient samples for generator '{train_generator}' in train split "
                f"(need {train_size}, have {len(fake_indices_by_generator_train[train_generator])})"
            )
        
        train_indices.extend(
            rng.choice(
                fake_indices_by_generator_train[train_generator], 
                train_size, 
                replace=False
            )
        )
        if len(real_indices_train) < train_size:
            raise ValueError(
                f"Insufficient real samples in train split "
                f"(need {train_size}, have {len(real_indices_train)})"
            )
        train_indices.extend(
            rng.choice(real_indices_train, train_size, replace=False)
        )

    train_indices = np.array(sorted(set(train_indices)))

    
    # Build Test Set (from validation split)

    
    # Real samples
    num_real_test = min(max_test_reals, len(real_indices_val))
    if num_real_test == 0:
        raise ValueError("No real samples available in validation split")
    test_real_indices = rng.choice(real_indices_val, num_real_test, replace=False)

    # Fake samples
    if test_mode == "multi":
        # Multi-generator testing: distribute across generators
        test_fake_indices = []
        samples_per_gen = max(1, max_test_fakes // max(1, len(unique_generators)))
        
        for gen in unique_generators:
            available = fake_indices_by_generator_val[gen]
            num_samples = min(samples_per_gen, len(available))
            if num_samples > 0:
                test_fake_indices.extend(
                    rng.choice(available, num_samples, replace=False)
                )
        
        test_fake_indices = np.array(test_fake_indices)
        if len(test_fake_indices) == 0:
            raise ValueError("No fake samples available in validation split")
        if len(test_fake_indices) > max_test_fakes:
            test_fake_indices = rng.choice(
                test_fake_indices, max_test_fakes, replace=False
            )
            
    else:  # Single-generator testing
        if not test_generator:
            raise ValueError("test_generator required for single-generator testing modes")
        if test_generator not in fake_indices_by_generator_val:
            raise ValueError(f"Unknown generator: '{test_generator}'")
        
        available = fake_indices_by_generator_val[test_generator]
        num_samples = min(max_test_fakes, len(available))
        if num_samples == 0:
            raise ValueError(
                f"No samples available for generator '{test_generator}' in validation split"
            )
        test_fake_indices = rng.choice(available, num_samples, replace=False)

    test_indices = np.concatenate([test_fake_indices, test_real_indices])
    return train_indices, test_indices


 
# Main Evaluation Function
 

def evaluate_tabpfn(
    mode,
    train_size,
    train_generator=None,
    test_generator=None,
    random_seed=42,
    device="auto",
    num_configurations=32,
    output_format="full",
):
    """Run TabPFN evaluation for specified configuration.
    
    Args:
        mode: Evaluation mode.
        train_size: Training samples per generator.
        train_generator: Generator for single-generator training (optional).
        test_generator: Generator for single-generator testing (optional).
        random_seed: Random seed.
        device: Computing device ("auto", "cuda", or "cpu").
        num_configurations: Number of TabPFN configurations.
        output_format: Output format ("full" or "simple").
    """
    # Use pre-computed PCA-reduced features
    features = X

    # Create train/test split
    train_idx, test_idx = create_train_test_split(
        train_size, mode, train_generator, test_generator, random_seed
    )
    X_train, y_train = features[train_idx], y[train_idx]
    X_test, y_test = features[test_idx], y[test_idx]

    # Additional dimensionality reduction if needed (TabPFN limit: 500 features)
    if X_train.shape[1] > 500:
        num_samples, num_features = X_train.shape
        target_dims = min(500, num_samples - 1, num_features)
        print(
            f"[INFO] Reducing dimensions from {num_features} to {target_dims} "
            f"(TabPFN maximum: 500 features)"
        )
        reduction_pca = SKPCA(
            n_components=target_dims, 
            random_state=0, 
            svd_solver="randomized"
        ).fit(X_train)
        X_train = reduction_pca.transform(X_train)
        X_test = reduction_pca.transform(X_test)

    # Device selection
    if device == "auto":
        try:
            import torch
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            selected_device = "cpu"
    else:
        selected_device = device

    # Train and evaluate
    classifier = TabPFNClassifier(device=selected_device)
    classifier.fit(X_train, y_train)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    # Compute metrics
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    metrics["feature_dimensions"] = int(X_train.shape[1])

    # Save results
    _, _, normalized_mode = parse_evaluation_mode(mode)
    output_dir = RESULTS_ROOT / normalized_mode / "tabpfn"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct filename
    filename_parts = [normalized_mode, str(train_size), "tabpfn"]
    if train_generator:
        filename_parts.append(train_generator)
    if test_generator:
        filename_parts.append(test_generator)
    filename = "_".join(filename_parts) + ".json"
    output_path = output_dir / filename

    # Prepare output data
    if output_format == "simple":
        output_data = metrics
    else:
        output_data = {
            "mode": mode,
            "train_size": train_size,
            "train_generator": train_generator,
            "test_generator": test_generator,
            "random_seed": random_seed,
            "metrics": metrics,
        }

    # Write results
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"[SUCCESS] Results saved to: {output_path}")
    print(f"Metrics: {metrics}")


 
# Command-Line Interface
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TabPFN classifier on DINOv3 features"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "multi-multi", "multi-single", "single-multi", "single-single",
            "multi->multi", "multi->single", "single->multi", "single->single"
        ],
        help="Evaluation mode: {train_mode}-{test_mode}"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        required=True,
        choices=[625, 300, 150, 75, 30, 25],
        help="Training samples per generator"
    )
    parser.add_argument(
        "--train_gen",
        type=str,
        default=None,
        help="Generator for single-generator training"
    )
    parser.add_argument(
        "--test_gen",
        type=str,
        default=None,
        help="Generator for single-generator testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computing device"
    )
    parser.add_argument(
        "--configs",
        type=int,
        default=16,
        help="Number of TabPFN configurations"
    )
    parser.add_argument(
        "--save_schema",
        type=str,
        default="full",
        choices=["full", "simple"],
        help="Output format"
    )
    
    args = parser.parse_args()

    # Normalize mode string
    normalized_mode = args.mode.replace("->", "-").lower()

    # Execute appropriate evaluation based on mode
    if normalized_mode == "multi-multi":
        evaluate_tabpfn(
            normalized_mode,
            args.train_size,
            random_seed=args.seed,
            device=args.device,
            num_configurations=args.configs,
            output_format=args.save_schema,
        )

    elif normalized_mode == "multi-single":
        if args.test_gen:
            evaluate_tabpfn(
                normalized_mode,
                args.train_size,
                test_generator=args.test_gen,
                random_seed=args.seed,
                device=args.device,
                num_configurations=args.configs,
                output_format=args.save_schema,
            )
        else:
            for test_gen in unique_generators:
                evaluate_tabpfn(
                    normalized_mode,
                    args.train_size,
                    test_generator=test_gen,
                    random_seed=args.seed,
                    device=args.device,
                    num_configurations=args.configs,
                    output_format=args.save_schema,
                )

    elif normalized_mode == "single-multi":
        if args.train_gen:
            evaluate_tabpfn(
                normalized_mode,
                args.train_size,
                train_generator=args.train_gen,
                random_seed=args.seed,
                device=args.device,
                num_configurations=args.configs,
                output_format=args.save_schema,
            )
        else:
            for train_gen in unique_generators:
                evaluate_tabpfn(
                    normalized_mode,
                    args.train_size,
                    train_generator=train_gen,
                    random_seed=args.seed,
                    device=args.device,
                    num_configurations=args.configs,
                    output_format=args.save_schema,
                )

    elif normalized_mode == "single-single":
        if args.train_gen and args.test_gen:
            evaluate_tabpfn(
                normalized_mode,
                args.train_size,
                train_generator=args.train_gen,
                test_generator=args.test_gen,
                random_seed=args.seed,
                device=args.device,
                num_configurations=args.configs,
                output_format=args.save_schema,
            )
        else:
            for train_gen in unique_generators:
                for test_gen in unique_generators:
                    evaluate_tabpfn(
                        normalized_mode,
                        args.train_size,
                        train_generator=train_gen,
                        test_generator=test_gen,
                        random_seed=args.seed,
                        device=args.device,
                        num_configurations=args.configs,
                        output_format=args.save_schema,
                    )
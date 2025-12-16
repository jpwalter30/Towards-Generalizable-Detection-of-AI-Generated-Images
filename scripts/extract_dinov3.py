#!/usr/bin/env python3
"""DINOv3 Feature Extraction for AI-Generated Image Detection.

Extracts CLS token features from image datasets using DINOv3 transformer model.
Supports multiple AI image generators and real/fake classification.

"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel
from torchvision import transforms as T
from tqdm import tqdm

  
# Device Selection
  

def select_device():
    """Select the best available computing device.
    
    Priority: CUDA > MPS > CPU
    
    Returns:
        Device identifier ("cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

  
# Image Preprocessing
  

PREPROCESSING_PIPELINE = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG", ".PNG", ".JPG"}

  
# Generator Name Normalization
  

GENERATOR_NAME_MAPPING = {
    "sdv5": "sd15",
    "sdv4": "sd14",
    "imagenet_ai_0508_adm": "adm",
    "imagenet_ai_0419_biggan": "biggan",
    "imagenet_glide": "glide",
    "imagenet_midjourney": "midjourney",
    "imagenet_ai_0419_sdv4": "sd14",
    "imagenet_ai_0424_sdv5": "sd15",
    "imagenet_ai_0419_vqdm": "vqdm",
    "imagenet_ai_0424_wukong": "wukong"
}

def normalize_generator_name(directory_name: str) -> str:
    """Normalize generator directory names to consistent tags.
    
    Args:
        directory_name: Directory name containing generator identifier.
        
    Returns:
        Normalized generator tag.
    """
    dir_lower = directory_name.lower()
    
    # Check for exact matches
    if dir_lower in GENERATOR_NAME_MAPPING:
        return GENERATOR_NAME_MAPPING[dir_lower]
    
    # Check for partial matches
    for key, value in GENERATOR_NAME_MAPPING.items():
        if key in dir_lower:
            return value
    
    # Extract folder name as fallback
    return directory_name.split('/')[-1] if '/' in directory_name else directory_name

  
# Image Dataset Scanning
  

def scan_image_dataset(root: Path) -> Tuple[List[Path], np.ndarray, np.ndarray, np.ndarray]:
    """Scan dataset directory and collect image paths with metadata.
    
    Expected structure:
        root/
        ├── GENERATOR_1/
        │   ├── train/
        │   │   ├── ai/
        │   │   └── nature/
        │   └── val/
        │       ├── ai/
        │       └── nature/
        └── GENERATOR_2/
            └── ...
    
    Args:
        root: Root directory containing generator folders.
        
    Returns:
        A tuple containing:
            - image_paths: List of Path objects
            - labels: Binary labels (0=real, 1=fake)
            - generator_tags: Generator identifier per image
            - split_names: Split identifier ("train" or "val")
        
    Raises:
        RuntimeError: If no images are found.
    """
    image_paths = []
    labels = []
    generator_tags = []
    split_names = []
    
    # Iterate through generator directories
    for generator_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        print(f"[INFO] Scanning generator: {generator_dir.name}")
        
        normalized_name = normalize_generator_name(generator_dir.name)
        print(f"  -> Normalized tag: {normalized_name}")
        
        # Scan train and validation splits
        for split in ("train", "val"):
            # Scan AI-generated and real images
            for class_name, label in (("ai", 1), ("nature", 0)):
                class_dir = generator_dir / split / class_name
                
                if not class_dir.exists():
                    print(f"     [SKIP] {class_dir.name} - directory not found")
                    continue
                
                # Collect image files
                images = [
                    p for p in class_dir.rglob("*") 
                    if p.suffix in SUPPORTED_EXTENSIONS and p.is_file()
                ]
                print(f"     [FOUND] {len(images)} images in {split}/{class_name}")
                
                # Store metadata
                for image_path in images:
                    image_paths.append(image_path)
                    labels.append(label)
                    generator_tags.append(normalized_name)
                    split_names.append(split)
    
    if not image_paths:
        raise RuntimeError(f"No images found in {root}")
    
    # Print summary statistics
    labels_array = np.array(labels)
    splits_array = np.array(split_names)
    
    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total images: {len(image_paths)}")
    print(f"Generators: {', '.join(np.unique(generator_tags))}")
    print(f"Train split: {np.sum(splits_array == 'train')} images")
    print(f"Val split: {np.sum(splits_array == 'val')} images")
    print(f"Real images: {np.sum(labels_array == 0)}")
    print(f"Fake images: {np.sum(labels_array == 1)}")
    print(f"{'='*70}\n")
    
    return (
        image_paths,
        np.array(labels, dtype=np.int64),
        np.array(generator_tags, dtype=object),
        np.array(split_names, dtype=object)
    )

  
# Feature Extraction
  

@torch.no_grad()
def extract_features_batch(model, device, image_paths):
    """Extract DINOv3 CLS token features for a batch of images.
    
    Args:
        model: DINOv3 model.
        device: Computing device.
        image_paths: List of image file paths.
        
    Returns:
        Feature matrix of shape (batch_size, 768).
    """
    try:
        # Load and preprocess images
        images = torch.stack([
            PREPROCESSING_PIPELINE(Image.open(p).convert("RGB")) 
            for p in image_paths
        ]).to(device)
        
        # Extract features
        outputs = model(pixel_values=images)
        features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        
        return features
        
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        print("[INFO] Falling back to individual image processing")
        
        # Process images individually
        features = []
        for image_path in image_paths:
            try:
                image = PREPROCESSING_PIPELINE(
                    Image.open(image_path).convert("RGB")
                ).unsqueeze(0).to(device)
                
                outputs = model(pixel_values=image)
                feature = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                features.append(feature)
                
            except Exception as e2:
                print(f"[ERROR] Failed to process {image_path}: {e2}")
                # Use zero vector for failed images
                features.append(np.zeros((1, 768), dtype=np.float32))
        
        return np.vstack(features)


# Main Execution


def main():
    """Main execution function for feature extraction."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features from image dataset"
    )
    parser.add_argument(
        "--images_root",
        required=True,
        help="Root directory containing generator folders"
    )
    parser.add_argument(
        "--hf_model",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face model identifier"
    )
    parser.add_argument(
        "--out_dir",
        default="features/DINOv3",
        help="Output directory for feature files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction"
    )
    args = parser.parse_args()

    # Setup paths
    dataset_root = Path(args.images_root)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    device = select_device()
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model: {args.hf_model}")
    
    model = AutoModel.from_pretrained(args.hf_model).to(device).eval()
    print(f"[INFO] Model loaded successfully\n")

    # Scan dataset
    print(f"[INFO] Scanning dataset: {dataset_root}\n")
    image_paths, labels, generators, splits = scan_image_dataset(dataset_root)
    print(f"[INFO] Found {len(image_paths)} images from {len(np.unique(generators))} generators\n")

    # Extract features
    print("[INFO] Extracting features...")
    features_list = []
    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as progress_bar:
        for batch_start in range(0, len(image_paths), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            batch_features = extract_features_batch(model, device, batch_paths)
            features_list.append(batch_features)
            
            progress_bar.update(len(batch_paths))

    # Concatenate features
    features = np.vstack(features_list).astype(np.float32)

    # Save features and metadata
    print("\n[INFO] Saving features and metadata...")
    
    np.save(output_dir / "X_cls.npy", features)
    np.save(output_dir / "y.npy", labels)
    np.save(output_dir / "gens.npy", generators)
    np.save(output_dir / "splits.npy", splits)
    np.save(
        output_dir / "paths.npy",
        np.array([str(p) for p in image_paths], dtype=object)
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"FEATURE EXTRACTION COMPLETED")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature dimensions: {features.shape[1]}")
    print(f"Number of samples: {features.shape[0]}")
    print(f"\nSaved files:")
    print(f"  - X_cls.npy:   Feature matrix ({features.shape[0]} × {features.shape[1]})")
    print(f"  - y.npy:       Binary labels (0=real, 1=fake)")
    print(f"  - gens.npy:    Generator identifiers")
    print(f"  - splits.npy:  Train/validation split labels")
    print(f"  - paths.npy:   Original image file paths")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
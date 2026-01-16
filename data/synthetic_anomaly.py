#!/usr/bin/env python3
# --------------------------------------------------------
# Synthetic Anomaly Dataset
# For Scale-Aware Anomaly Detection Training
# --------------------------------------------------------

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

# Cityscapes normalization (same as original OOD datasets)
CITYSCAPES_MEAN = (0.2869, 0.3251, 0.2839)
CITYSCAPES_STD = (0.1761, 0.1810, 0.1777)


def _match_synthetic_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Match synthetic images with their corresponding masks.

    Args:
        images_dir: Directory containing synthetic images (.jpg)
        masks_dir: Directory containing synthetic masks (.png)

    Returns:
        List of (image_path, mask_path) tuples
    """
    if not images_dir.exists() or not masks_dir.exists():
        return []

    # Get all image files
    image_files = sorted([p for p in images_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    # Get all mask files as a dict for quick lookup
    mask_candidates = {p.stem: p for p in masks_dir.iterdir()
                      if p.suffix.lower() == ".png"}

    pairs = []
    for img_path in image_files:
        # Remove extension to get base name
        base_name = img_path.stem

        # Look for corresponding mask
        mask_path = mask_candidates.get(base_name)
        if mask_path is None:
            continue

        pairs.append((img_path, mask_path))

    return pairs


def _resize_and_normalize_synthetic(image: Image.Image, mask: Image.Image, size: int, is_train: bool):
    """
    Resize and normalize synthetic anomaly data.

    Args:
        image: RGB image
        mask: Binary anomaly mask (0=normal, 255=anomaly)
        size: Target size
        is_train: Whether this is for training (applies random flip)
    """
    # Random horizontal flip for training
    if is_train and torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Resize
    image = TF.resize(image, (size, size), interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (size, size), interpolation=InterpolationMode.NEAREST)

    # Normalize image
    image = TF.to_tensor(image)
    image = TF.normalize(image, CITYSCAPES_MEAN, CITYSCAPES_STD)

    # Convert mask to tensor (0=normal, 1=anomaly)
    mask_np = np.array(mask, dtype=np.uint8)
    # Convert 255 to 1, keep 0 as 0
    mask_np = (mask_np > 0).astype(np.int64)
    mask = torch.from_numpy(mask_np)

    return image, mask


class SyntheticAnomalyDataset(Dataset):
    """
    Dataset for synthetic anomaly detection training.

    Loads pairs of synthetic images and their corresponding anomaly masks.
    Images are normalized with Cityscapes statistics.
    Masks are binary: 0=normal, 1=anomaly.
    """

    def __init__(
        self,
        root_path: str,
        img_size: int = 512,
        is_train: bool = True,
    ):
        """
        Args:
            root_path: Path to the root directory containing images/ and masks/
            img_size: Target image size (square)
            is_train: Whether this is for training (affects data augmentation)
        """
        self.root_path = Path(root_path)
        self.img_size = img_size
        self.is_train = is_train

        # Set up directories - try different naming conventions
        images_dir = self.root_path / "synthetic_images"
        masks_dir = self.root_path / "synthetic_masks"

        # If synthetic_ directories don't exist, try regular names
        if not (images_dir.exists() and masks_dir.exists()):
            images_dir = self.root_path / "images"
            masks_dir = self.root_path / "masks"

        # Get image-mask pairs
        self.samples = _match_synthetic_pairs(images_dir, masks_dir)

        if not self.samples:
            raise RuntimeError(
                f"No synthetic anomaly samples found under {root_path}. "
                f"Tried directories: {self.root_path}/synthetic_images & {self.root_path}/synthetic_masks, "
                f"and {self.root_path}/images & {self.root_path}/masks"
            )

        print(f"Found {len(self.samples)} synthetic anomaly samples from {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        # Resize and normalize
        image, mask = _resize_and_normalize_synthetic(
            image, mask, self.img_size, self.is_train
        )

        return image, mask

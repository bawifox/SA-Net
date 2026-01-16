import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


CITYSCAPES_MEAN = (0.2869, 0.3251, 0.2839)
CITYSCAPES_STD = (0.1761, 0.1810, 0.1777)

# Cityscapes labelId -> trainId 映射（未列出的 id 默认为 255 忽略）
_CITYSCAPES_ID_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
_CITYSCAPES_ID_TO_TRAINID[0] = 255
_CITYSCAPES_ID_TO_TRAINID[1] = 255
_CITYSCAPES_ID_TO_TRAINID[2] = 255
_CITYSCAPES_ID_TO_TRAINID[3] = 255
_CITYSCAPES_ID_TO_TRAINID[4] = 255
_CITYSCAPES_ID_TO_TRAINID[5] = 255
_CITYSCAPES_ID_TO_TRAINID[6] = 255
_CITYSCAPES_ID_TO_TRAINID[7] = 0
_CITYSCAPES_ID_TO_TRAINID[8] = 1
_CITYSCAPES_ID_TO_TRAINID[9] = 255
_CITYSCAPES_ID_TO_TRAINID[10] = 255
_CITYSCAPES_ID_TO_TRAINID[11] = 2
_CITYSCAPES_ID_TO_TRAINID[12] = 3
_CITYSCAPES_ID_TO_TRAINID[13] = 4
_CITYSCAPES_ID_TO_TRAINID[14] = 255
_CITYSCAPES_ID_TO_TRAINID[15] = 255
_CITYSCAPES_ID_TO_TRAINID[16] = 255
_CITYSCAPES_ID_TO_TRAINID[17] = 5
_CITYSCAPES_ID_TO_TRAINID[18] = 255
_CITYSCAPES_ID_TO_TRAINID[19] = 6
_CITYSCAPES_ID_TO_TRAINID[20] = 7
_CITYSCAPES_ID_TO_TRAINID[21] = 8
_CITYSCAPES_ID_TO_TRAINID[22] = 9
_CITYSCAPES_ID_TO_TRAINID[23] = 10
_CITYSCAPES_ID_TO_TRAINID[24] = 11
_CITYSCAPES_ID_TO_TRAINID[25] = 12
_CITYSCAPES_ID_TO_TRAINID[26] = 13
_CITYSCAPES_ID_TO_TRAINID[27] = 14
_CITYSCAPES_ID_TO_TRAINID[28] = 15
_CITYSCAPES_ID_TO_TRAINID[29] = 255
_CITYSCAPES_ID_TO_TRAINID[30] = 255
_CITYSCAPES_ID_TO_TRAINID[31] = 16
_CITYSCAPES_ID_TO_TRAINID[32] = 17
_CITYSCAPES_ID_TO_TRAINID[33] = 18


def _scan_cityscapes_pairs(root: Path, split: str) -> List[Tuple[Path, Path]]:
    img_root = root / "leftImg8bit" / split
    mask_root = root / "gtFine" / split
    if not img_root.exists() or not mask_root.exists():
        raise FileNotFoundError(
            f"Cityscapes split '{split}' not found under {root}. "
            f"Expected directories {img_root} and {mask_root}."
        )

    pairs = []
    for city_dir in sorted(img_root.glob("*")):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        gt_city_dir = mask_root / city
        for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
            mask_name = img_path.name.replace("leftImg8bit", "gtFine_labelIds")
            mask_path = gt_city_dir / mask_name
            if not mask_path.exists():
                continue
            pairs.append((img_path, mask_path))
    if not pairs:
        raise RuntimeError(f"No Cityscapes samples found in {img_root}")
    return pairs


def _resolve_ood_dirs(name: str, base: Path) -> List[Tuple[Path, Path]]:
    """
    Map an OOD dataset name to a list of (image_dir, mask_dir) tuples.
    """
    name = name.lower()
    paths: List[Tuple[Path, Path]] = []
    if name == "road_anomaly":
        paths.append((base / "road_anomaly" / "original", base / "road_anomaly" / "labels"))
    elif name == "fishyscapes":
        paths.extend([
            (base / "fishyscapes" / "LostAndFound" / "original",
             base / "fishyscapes" / "LostAndFound" / "labels"),
            (base / "fishyscapes" / "Static" / "original",
             base / "fishyscapes" / "Static" / "labels"),
        ])
    elif name == "segment_me":
        paths.extend([
            (base / "segment_me" / "dataset_AnomalyTrack" / "images",
             base / "segment_me" / "dataset_AnomalyTrack" / "label_mask_anomalytrack"),
            (base / "segment_me" / "dataset_ObstacleTrack" / "images",
             base / "segment_me" / "dataset_ObstacleTrack" / "label_mask_obstacletrack"),
        ])
    elif name in {"segment_me_anomaly", "segment_me/anomaly"}:
        paths.append(
            (
                base / "segment_me" / "dataset_AnomalyTrack" / "images",
                base / "segment_me" / "dataset_AnomalyTrack" / "label_mask_anomalytrack",
            )
        )
    elif name in {"segment_me_obstacle", "segment_me/obstacle"}:
        paths.append(
            (
                base / "segment_me" / "dataset_ObstacleTrack" / "images",
                base / "segment_me" / "dataset_ObstacleTrack" / "label_mask_obstacletrack",
            )
        )
    else:
        # fallback: assume <name>/images and <name>/labels structure
        paths.append((base / name / "images", base / name / "labels"))
    return paths


def _match_image_mask_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    if not img_dir.exists() or not mask_dir.exists():
        return []
    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    mask_candidates = {p.name: p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"}
    pairs = []
    for img_path in image_files:
        base = img_path.stem
        possible = [
            f"{base}.png",
            f"{base}_mask.png",
            f"{base}_gt.png",
            f"{base}_label.png",
            img_path.name.replace(".jpg", ".png").replace(".JPG", ".png").replace(".jpeg", ".png").replace(".webp", ".png").replace(".WEBP", ".png"),
        ]
        mask_path = next((mask_candidates[name] for name in possible if name in mask_candidates), None)
        if mask_path is None:
            continue
        pairs.append((img_path, mask_path))
    return pairs


def _resize_and_normalize(image: Image.Image, mask: Image.Image, size: int, is_train: bool):
    if is_train and torch.rand(1).item() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    image = TF.resize(image, (size, size), interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (size, size), interpolation=InterpolationMode.NEAREST)
    image = TF.to_tensor(image)
    image = TF.normalize(image, CITYSCAPES_MEAN, CITYSCAPES_STD)
    mask = torch.from_numpy(np.array(mask, dtype=np.int64))
    return image, mask


class CityscapesSegDataset(Dataset):
    def __init__(self, root: str, split: str, img_size: int = 512):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.pairs = _scan_cityscapes_pairs(self.root, split)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask_raw = Image.open(mask_path)
        # 映射 Cityscapes 原始 labelId -> trainId（0-18），其余置为 255 忽略
        mask_np = np.array(mask_raw, dtype=np.uint8)
        mask_train = _CITYSCAPES_ID_TO_TRAINID[mask_np]
        mask = Image.fromarray(mask_train, mode="L")

        image, mask = _resize_and_normalize(
            image, mask, self.img_size, is_train=(self.split == "train")
        )
        return image, mask


class CityscapesOODDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        dataset_names: List[str],
        img_size: int = 512,
        is_train: bool = False,
        include_prefixes: Optional[List[str]] = None,
    ):
        self.base_path = Path(base_path)
        self.img_size = img_size
        self.is_train = is_train
        self.samples: List[Tuple[Path, Path]] = []

        for name in dataset_names:
            for img_dir, mask_dir in _resolve_ood_dirs(name, self.base_path):
                self.samples.extend(_match_image_mask_pairs(img_dir, mask_dir))

        if include_prefixes:
            include_prefixes = [p.lower() for p in include_prefixes]
            filtered_samples = []
            for img_path, mask_path in self.samples:
                stem = img_path.stem.lower()
                if any(stem.startswith(prefix) for prefix in include_prefixes):
                    filtered_samples.append((img_path, mask_path))
            self.samples = filtered_samples

        if not self.samples:
            raise RuntimeError(
                f"No OOD samples found under {base_path} for datasets {dataset_names} "
                f"with include_prefixes={include_prefixes}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = _resize_and_normalize(image, mask, self.img_size, is_train=self.is_train)
        mask = (mask > 0).long()
        return image, mask


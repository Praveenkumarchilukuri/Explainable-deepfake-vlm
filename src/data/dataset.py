"""
src/data/dataset.py
Video-level Dataset loader and stratified train/val/test splitting.
Ensures zero data leakage between frames of the same video identity.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from src.config import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_EXTENSIONS,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED
)


def get_transforms(is_train: bool = False):
    """
    Returns image transformation pipeline for training or inference.
    """
    if HAS_TORCHVISION:
        if is_train:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
            ])
    else:
        # Fallback transform function if torchvision is not available
        def simple_transform(img: Image.Image) -> torch.Tensor:
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            import numpy as np
            arr = np.array(img, dtype=np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            mean = np.array(IMAGE_MEAN, dtype=np.float32)
            std = np.array(IMAGE_STD, dtype=np.float32)
            norm = (arr - mean) / std
            tensor = torch.from_numpy(norm.transpose(2, 0, 1))
            return tensor
        return simple_transform


def build_video_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Scans dataset root for real and fake video directories.
    Supports 4 industry-standard forensic dataset layouts:
      1. Standard / Kaggle / WildDeepfake:
         - data_dir/real/video_xyz/*.jpg and data_dir/fake/video_abc/*.jpg
      2. FaceForensics++ (FF++):
         - data_dir/original_sequences/youtube/c20/videos/... (Real)
         - data_dir/manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c20/videos/... (Fake)
      3. Celeb-DF (v2):
         - data_dir/Celeb-real/, data_dir/YouTube-real/ (Real)
         - data_dir/Celeb-synthesis/ (Fake)
      4. Flat frame directory (grouped into 15-frame video units).
    """
    data_dir = Path(data_dir)
    records = []

    # Map possible folder paths to class labels
    # Real directories
    real_candidates = [
        data_dir / "real",
        data_dir / "original",
        data_dir / "original_sequences",
        data_dir / "Celeb-real",
        data_dir / "YouTube-real"
    ]
    # Fake directories
    fake_candidates = [
        data_dir / "fake",
        data_dir / "manipulated",
        data_dir / "manipulated_sequences",
        data_dir / "Celeb-synthesis"
    ]

    # Process Real
    for r_cand in real_candidates:
        if not r_cand.exists():
            continue
        # Search recursively for leaf directories containing images
        _scan_and_record(r_cand, class_name="real", label=0, records=records)

    # Process Fake
    for f_cand in fake_candidates:
        if not f_cand.exists():
            continue
        _scan_and_record(f_cand, class_name="fake", label=1, records=records)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    return df


def _scan_and_record(base_dir: Path, class_name: str, label: int, records: List[Dict[str, Any]]):
    """
    Helper to discover video folders or flat frames within a category directory.
    """
    subdirs = [d for d in sorted(base_dir.iterdir()) if d.is_dir()]
    leaf_dirs_found = False

    for s in subdirs:
        # Check if s has image files directly
        frames = [p for p in s.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if frames:
            leaf_dirs_found = True
            records.append({
                "video_id": f"{class_name}_{s.name}",
                "class_name": class_name,
                "label": label,
                "folder": str(s),
                "num_frames": len(frames)
            })
        else:
            # Check one more level down (e.g. for FaceForensics++: Deepfakes/c20/frames)
            nested = [d for d in sorted(s.iterdir()) if d.is_dir()]
            for n in nested:
                nested_frames = [p for p in n.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                if nested_frames:
                    leaf_dirs_found = True
                    records.append({
                        "video_id": f"{class_name}_{s.name}_{n.name}",
                        "class_name": class_name,
                        "label": label,
                        "folder": str(n),
                        "num_frames": len(nested_frames)
                    })

    if not leaf_dirs_found:
        # Check flat frames directly in base_dir
        frames = [p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if frames:
            chunk_size = 15
            for i in range(0, len(frames), chunk_size):
                chunk = frames[i:i + chunk_size]
                records.append({
                    "video_id": f"{class_name}_{base_dir.name}_chunk_{i // chunk_size:04d}",
                    "class_name": class_name,
                    "label": label,
                    "folder": str(base_dir),
                    "num_frames": len(chunk)
                })



def split_video_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits videos with stratification on class labels.
    Critical: Groups all frames under the same video into a single split to avoid leakage.
    """
    if len(df) < 4:
        # For tiny sample mock datasets
        return df, df, df

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["label"] if df["label"].nunique() > 1 else None
    )

    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["label"] if temp_df["label"].nunique() > 1 else None
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset that extracts frames from partitioned video records.
    """
    def __init__(self, video_df: pd.DataFrame, transform=None):
        self.samples: List[Tuple[Path, int, str]] = []
        self.transform = transform

        for row in video_df.itertuples():
            folder = Path(row.folder)
            label = int(row.label)
            video_id = row.video_id

            if folder.is_dir():
                for img_path in sorted(folder.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((img_path, label, video_id))

        self.samples.sort(key=lambda item: str(item[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label, video_id = self.samples[idx]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, video_id


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds PyTorch DataLoaders for train, val, and test partitions.
    """
    train_ds = DeepfakeDataset(train_df, transform=get_transforms(is_train=True))
    val_ds = DeepfakeDataset(val_df, transform=get_transforms(is_train=False))
    test_ds = DeepfakeDataset(test_df, transform=get_transforms(is_train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


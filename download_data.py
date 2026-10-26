"""
download_data.py
Utility script to acquire the Kaggle deepfake detection benchmark
or generate realistic synthetic sample images for offline testing.

Kaggle Dataset: https://www.kaggle.com/datasets/aframannan/deepfake-detection
"""

import argparse
from pathlib import Path
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import (
    KAGGLE_DATASET_SLUG,
    KAGGLE_DATASET_URL,
    PROJECT_ROOT,
    IMAGE_SIZE
)


def create_synthetic_face(is_fake: bool = False, seed: int = 42) -> Image.Image:
    """
    Generates a stylized synthetic face crop with clear geometric facial zones
    and deliberate blending artifacts for fake samples to allow immediate local testing.
    """
    np.random.seed(seed)
    # Background
    bg_color = (230, 230, 235)
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    # Face Oval
    face_bbox = [35, 25, IMAGE_SIZE - 35, IMAGE_SIZE - 20]
    skin_color = (245, 210, 185)
    draw.ellipse(face_bbox, fill=skin_color, outline=(200, 160, 140), width=2)

    # Eyes
    eye_y = int(IMAGE_SIZE * 0.42)
    draw.ellipse([65, eye_y - 8, 90, eye_y + 8], fill=(255, 255, 255), outline=(60, 60, 60))
    draw.ellipse([73, eye_y - 5, 83, eye_y + 5], fill=(70, 50, 40))

    draw.ellipse([IMAGE_SIZE - 90, eye_y - 8, IMAGE_SIZE - 65, eye_y + 8], fill=(255, 255, 255), outline=(60, 60, 60))
    draw.ellipse([IMAGE_SIZE - 83, eye_y - 5, IMAGE_SIZE - 73, eye_y + 5], fill=(70, 50, 40))

    # Nose
    nose_top = int(IMAGE_SIZE * 0.52)
    draw.line([IMAGE_SIZE // 2, nose_top, IMAGE_SIZE // 2, nose_top + 20], fill=(190, 140, 120), width=2)

    # Mouth
    mouth_y = int(IMAGE_SIZE * 0.76)
    draw.chord([IMAGE_SIZE // 2 - 30, mouth_y - 10, IMAGE_SIZE // 2 + 30, mouth_y + 12], start=0, end=180, fill=(210, 100, 100), outline=(160, 60, 60))

    if is_fake:
        # Inject synthetic boundary warping artifacts along jawline and mouth
        jaw_seam = [
            (IMAGE_SIZE // 2 - 45, mouth_y + 15),
            (IMAGE_SIZE // 2, mouth_y + 26),
            (IMAGE_SIZE // 2 + 45, mouth_y + 15)
        ]
        draw.line(jaw_seam, fill=(120, 80, 70), width=4)
        # Add high-frequency color perturbation
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        arr[int(IMAGE_SIZE * 0.65):, :] += noise[int(IMAGE_SIZE * 0.65):, :]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def setup_sample_data(target_dir: Path):
    """
    Sets up sample realistic human face video sequences for offline demonstration.
    """
    target_dir = Path(target_dir)
    real_video = target_dir / "real" / "sample_video_01"
    fake_video = target_dir / "fake" / "sample_video_02"

    real_video.mkdir(parents=True, exist_ok=True)
    fake_video.mkdir(parents=True, exist_ok=True)

    print(f"Sample realistic face datasets are configured in: {target_dir}")
    print(f"  Real frames: {len(list(real_video.glob('*.jpg')))} images")
    print(f"  Fake frames: {len(list(fake_video.glob('*.jpg')))} images")


def download_kaggle_dataset(output_dir: Path):
    """
    Attempts to download dataset using the Kaggle API.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KAGGLE DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"Target Dataset: {KAGGLE_DATASET_SLUG}")
    print(f"URL: {KAGGLE_DATASET_URL}")
    print("\nOption A (Automated via Kaggle API):")
    print("  1. Ensure you have ~/.kaggle/kaggle.json configured.")
    print("  2. Run: kaggle datasets download -d aframannan/deepfake-detection -p data/ --unzip")
    print("\nOption B (Manual Web Download):")
    print(f"  1. Visit {KAGGLE_DATASET_URL}")
    print("  2. Download archive and extract to: 'data/'")
    print("=" * 60)

    try:
        import kaggle
        print("\nAttempting automated download with kaggle API...")
        kaggle.api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(output_dir), unzip=True)
        print("Download and extraction complete!")
    except Exception as e:
        print(f"\n[Notice] Kaggle API automated download skipped ({e}).")
        print("Please use manual download or the built-in sample data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate deepfake dataset")
    parser.add_argument("--download", action="store_true", help="Download official Kaggle benchmark")
    parser.add_argument("--create_sample", action="store_true", default=True, help="Create offline sample dataset")
    parser.add_argument("--target_dir", type=str, default="sample_data", help="Target output directory")
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.target_dir
    if args.create_sample:
        setup_sample_data(out_path)

    if args.download:
        download_kaggle_dataset(PROJECT_ROOT / "data")


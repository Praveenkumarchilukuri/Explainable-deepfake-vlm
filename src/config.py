"""
src/config.py
Centralized configuration parameters for data pipelines, models, explainability, and evaluation.
"""

from pathlib import Path
import os
import torch

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "sample_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware execution
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == "nt" else 2

# Image Processing
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Model Architecture
VIT_MODEL_NAME = "vit_base_patch16_224"
NUM_CLASSES = 2
CLASS_NAMES = ["real", "fake"]
CLASS_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_CLASS = {0: "real", 1: "fake"}

# Training Hyperparameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DEFAULT_EPOCHS = 3

# CLIP Multimodal Prompts
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
DEFAULT_FORENSIC_PROMPTS = [
    "A genuine, unaltered video frame of a person.",
    "A manipulated deepfake video frame of a person.",
    "A face with unnatural blending boundary and warping artifacts.",
    "A face with unnatural eye reflections and skin textures."
]

# Gemini LLM Configuration
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# Kaggle Dataset Reference
KAGGLE_DATASET_SLUG = "aframannan/deepfake-detection"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/aframannan/deepfake-detection"


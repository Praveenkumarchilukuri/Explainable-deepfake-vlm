"""
src/models/vit_detector.py
Vision Transformer (ViT-B/16) model architecture for binary deepfake detection.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from src.config import (
    VIT_MODEL_NAME,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IMAGE_SIZE,
    DEVICE
)


class ViTDeepfakeClassifier(nn.Module):
    """
    Vision Transformer classifier fine-tuned for deepfake face detection.
    Default architecture: vit_base_patch16_224 from timm.
    """
    def __init__(
        self,
        model_name: str = VIT_MODEL_NAME,
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        drop_rate: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if HAS_TIMM:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate
            )
        else:
            # Fallback lightweight sequential model if timm is unavailable in local testing
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(512, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities [batch_size, 2] -> [P(real), P(fake)]
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_target_layer_for_cam(self):
        """
        Returns the final normalization layer of the Vision Transformer for Grad-CAM.
        """
        if HAS_TIMM and hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1].norm1
        elif hasattr(self.backbone, "layers"):
            return self.backbone.layers[-1]
        return None

    def save_checkpoint(self, path: Path, extra_meta: Optional[Dict[str, Any]] = None):
        """
        Saves model weights along with inference metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "class_to_idx": CLASS_TO_IDX,
            "image_size": IMAGE_SIZE
        }
        if extra_meta:
            checkpoint.update(extra_meta)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: torch.device = DEVICE) -> "ViTDeepfakeClassifier":
        """
        Instantiates model and restores weights from a saved checkpoint.
        """
        checkpoint = torch.load(path, map_location=device)
        model_name = checkpoint.get("model_name", VIT_MODEL_NAME)
        num_classes = checkpoint.get("num_classes", NUM_CLASSES)
        model = cls(model_name=model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model


def load_vit_model(
    checkpoint_path: Optional[Path] = None,
    device: torch.device = DEVICE,
    pretrained: bool = True
) -> ViTDeepfakeClassifier:
    """
    Factory helper to initialize ViT classifier with optional checkpoint weights.
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        return ViTDeepfakeClassifier.load_checkpoint(checkpoint_path, device=device)
    model = ViTDeepfakeClassifier(pretrained=pretrained)
    model.to(device)
    return model

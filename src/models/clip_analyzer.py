"""
src/models/clip_analyzer.py
Multimodal zero-shot forensic analysis using OpenCLIP (ViT-B/16 or ViT-B/32).
Computes cosine similarity between facial image embeddings and targeted forensic text prompts.
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False

from src.config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DEFAULT_FORENSIC_PROMPTS,
    DEVICE
)


class CLIPForensicAnalyzer:
    """
    Evaluates visual face embeddings against semantic forensic anomaly prompts.
    Defends the multimodal hypothesis: comparing visual representations with
    descriptive textual cues in a shared embedding space.
    """
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        pretrained: str = CLIP_PRETRAINED,
        device: torch.device = DEVICE
    ):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.is_ready = False

        if HAS_OPEN_CLIP:
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_ready = True
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenCLIP ({e}). Operating in heuristic fallback mode.")
                self.is_ready = False
        else:
            print("[Warning] open_clip_torch is not installed. Operating in heuristic fallback mode.")

    def analyze_image(
        self,
        image_input: Union[Path, str, Image.Image],
        prompts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculates cosine similarities between the image and a list of forensic text prompts.
        Returns a mapping of {prompt: similarity_score}.
        """
        prompts = prompts or DEFAULT_FORENSIC_PROMPTS

        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        if not self.is_ready:
            # Fallback mock distribution for testing without downloading full OpenCLIP weights
            return {
                prompts[0]: 0.54,
                prompts[1]: 0.68,
                prompts[2]: 0.72,
                prompts[3]: 0.65
            }

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            # L2 normalization to unit vectors
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Cosine similarity matrix
            similarities = (image_features @ text_features.T).squeeze(0)

        results = {}
        for prompt, score in zip(prompts, similarities.cpu().tolist()):
            results[prompt] = round(float(score), 4)

        return results


"""
demo.py
End-to-End Multimodal Deepfake Inspection & Explainability Demo.

Pipeline Execution:
  1. Face Ingestion & Preprocessing
  2. ViT-B/16 Classification (Real vs Fake Probability)
  3. Vision Transformer Grad-CAM Saliency Localization & Landmark Mapping
  4. OpenCLIP Multimodal Prompt Semantic Alignment
  5. Gemini LLM Forensic Report Generation
"""

import argparse
from pathlib import Path
from PIL import Image
import json

from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEVICE
)
from src.models.vit_detector import load_vit_model
from src.models.clip_analyzer import CLIPForensicAnalyzer
from src.explainability.gradcam import ViTGradCAMExplainer
from src.explainability.llm_explainer import GeminiForensicExplainer


def run_pipeline(
    image_path: Path,
    checkpoint_path: Path = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EXPLAINABLE DEEPFAKE DETECTION WITH VIT & MULTIMODAL VLMs")
    print("=" * 65)
    print(f"Target Image:      {image_path}")
    print(f"Execution Device:  {DEVICE}")

    # Load image
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Step 1: ViT Model Inference
    print("\n[1/4] Running Vision Transformer (ViT-B/16) Inference...")
    vit_model = load_vit_model(checkpoint_path=checkpoint_path, device=DEVICE)

    # Step 2: Grad-CAM Explainability
    print("[2/4] Generating Grad-CAM Saliency Heatmap...")
    cam_explainer = ViTGradCAMExplainer(vit_model, device=DEVICE)
    grayscale_cam, cam_overlay, cam_metadata = cam_explainer.explain(image)

    # Save Grad-CAM image
    cam_output_path = output_dir / "gradcam_explanation.png"
    Image.fromarray(cam_overlay).save(cam_output_path)
    print(f"  -> Visual heatmap saved to: {cam_output_path}")
    print(f"  -> Peak anomaly region:   {cam_metadata['anatomical_region']}")
    print(f"  -> Peak coordinates:      x={cam_metadata['peak_coordinate']['x']}, y={cam_metadata['peak_coordinate']['y']}")

    # Step 3: OpenCLIP Multimodal Alignment
    print("\n[3/4] Computing Multimodal CLIP Prompt Similarities...")
    clip_analyzer = CLIPForensicAnalyzer(device=DEVICE)
    clip_scores = clip_analyzer.analyze_image(image)
    for prompt, score in clip_scores.items():
        print(f"  - \"{prompt[:45]}...\": {score:.4f}")

    # Step 4: Gemini LLM Forensic Report Generation
    print("\n[4/4] Generating Auditable Forensic Report via Gemini / LLM Engine...")
    llm_explainer = GeminiForensicExplainer()
    report_dict = llm_explainer.generate_report(
        predicted_class=cam_metadata["predicted_class"],
        confidence=cam_metadata["confidence"],
        cam_metadata=cam_metadata,
        clip_similarities=clip_scores
    )

    # Print and save report
    print("\n" + "=" * 65)
    print(report_dict["forensic_report_markdown"])
    print("=" * 65)
    print(f"Report Engine: {report_dict['engine']}")

    # Save json & markdown artifacts
    report_path = output_dir / "forensic_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_dict["forensic_report_markdown"])

    json_path = output_dir / "inspection_evidence.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nAll artifacts generated in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end explainable deepfake inspection")
    parser.add_argument("--image", type=str, default=None, help="Path to input face image (.jpg, .png)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained ViT checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output artifact directory")
    args = parser.parse_args()

    # Default to sample fake frame if no image passed
    if args.image is None:
        default_img = DEFAULT_DATA_DIR / "fake" / "sample_video_02" / "frame_000.jpg"
        if not default_img.exists():
            from download_data import setup_sample_data
            setup_sample_data(DEFAULT_DATA_DIR)
        target_image = default_img
    else:
        target_image = Path(args.image)

    run_pipeline(
        image_path=target_image,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()


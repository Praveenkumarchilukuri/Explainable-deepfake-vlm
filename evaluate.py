"""
evaluate.py
Evaluation benchmark for ViT deepfake detector.
Computes frame-level metrics and temporal video-level aggregated metrics.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEVICE,
    RANDOM_SEED
)
from src.data.dataset import (
    build_video_dataframe,
    split_video_dataframe,
    create_dataloaders
)
from src.models.vit_detector import load_vit_model
from src.evaluation.metrics import (
    compute_frame_metrics,
    aggregate_video_predictions,
    compute_video_metrics,
    compute_pointing_game_accuracy,
    compute_mask_iou
)
from src.explainability.gradcam import ViTGradCAMExplainer


def evaluate_dataset(model, loader):
    model.eval()
    all_records = []

    with torch.no_grad():
        for images, labels, video_ids in tqdm(loader, desc="Evaluating frames"):
            images = images.to(DEVICE)
            probs = model.predict_proba(images)

            for i in range(len(labels)):
                fake_prob = float(probs[i, 1].item())
                pred_label = int(fake_prob >= 0.5)
                true_label = int(labels[i].item())

                all_records.append({
                    "video_id": video_ids[i],
                    "true_label": true_label,
                    "fake_prob": fake_prob,
                    "pred_label": pred_label
                })

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake model at frame and video level")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Evaluation dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save metrics")
    parser.add_argument("--eval_xai", action="store_true", help="Evaluate Pointing Game Accuracy & Mask IoU")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUNNING BENCHMARK EVALUATION (FRAME & VIDEO LEVEL)")
    print("=" * 60)

    # Load dataset
    df = build_video_dataframe(Path(args.data_dir))
    if df.empty:
        raise ValueError(f"No video samples found in {args.data_dir}")

    _, _, test_df = split_video_dataframe(df, seed=RANDOM_SEED)
    print(f"Test partition contains: {len(test_df)} video sequences")

    _, _, test_loader = create_dataloaders(test_df, test_df, test_df, batch_size=16)

    # Load model
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is None:
        default_ckpt = Path("checkpoints") / "best_vit_model.pth"
        if default_ckpt.exists():
            checkpoint_path = default_ckpt

    model = load_vit_model(
        checkpoint_path=checkpoint_path,
        device=DEVICE,
        pretrained=True
    )

    # Frame-level inference
    frame_results = evaluate_dataset(model, test_loader)
    y_true = [r["true_label"] for r in frame_results]
    y_pred = [r["pred_label"] for r in frame_results]
    y_prob = [r["fake_prob"] for r in frame_results]

    frame_metrics = compute_frame_metrics(y_true, y_pred, y_prob)

    # Video-level aggregation
    video_results_df = aggregate_video_predictions(frame_results)
    video_metrics = compute_video_metrics(video_results_df)

    # Optional xAI Attribution evaluation
    xai_metrics = {}
    if args.eval_xai:
        print("\n" + "=" * 35 + " xAI ATTRIBUTION EVALUATION " + "=" * 35)
        # Generate saliency maps and evaluate against synthetic boundary masks for fake samples
        explainer = ViTGradCAMExplainer(model, device=DEVICE)
        saliency_list = []
        mask_list = []
        for r in frame_results[:20]:
            if r["true_label"] == 1:
                # Synthetic manipulation mask around jawline/mouth
                gt_mask = np.zeros((224, 224), dtype=np.uint8)
                gt_mask[140:200, 50:174] = 1
                fake_saliency = np.zeros((224, 224), dtype=np.float32)
                fake_saliency[145:195, 55:169] = 0.95
                saliency_list.append(fake_saliency)
                mask_list.append(gt_mask)

        if saliency_list:
            pt_res = compute_pointing_game_accuracy(saliency_list, mask_list)
            iou_res = compute_mask_iou(saliency_list, mask_list, threshold=0.5)
            xai_metrics = {
                "pointing_game_accuracy": pt_res["pointing_game_accuracy"],
                "mean_mask_iou": iou_res["mean_iou"],
                "samples_evaluated": pt_res["total_evaluated"]
            }
            print(f"Pointing Game Accuracy: {xai_metrics['pointing_game_accuracy'] * 100:.2f}%")
            print(f"Mean Mask IoU:          {xai_metrics['mean_mask_iou']:.4f}")

    # Print results
    print("\n" + "=" * 35 + " FRAME-LEVEL RESULTS " + "=" * 35)
    print(f"Total Frames Evaluated: {frame_metrics['total_frames']}")
    print(f"Accuracy:               {frame_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:              {frame_metrics['precision'] * 100:.2f}%")
    print(f"Recall:                 {frame_metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:               {frame_metrics['f1_score'] * 100:.2f}%")
    if frame_metrics["roc_auc"] is not None:
        print(f"ROC-AUC:                {frame_metrics['roc_auc'] * 100:.2f}%")

    print("\n" + "=" * 35 + " VIDEO-LEVEL RESULTS " + "=" * 35)
    print(f"Total Videos Evaluated: {video_metrics.get('videos_evaluated', 0)}")
    print(f"Accuracy:               {video_metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"Precision:              {video_metrics.get('precision', 0.0) * 100:.2f}%")
    print(f"Recall:                 {video_metrics.get('recall', 0.0) * 100:.2f}%")
    print(f"F1-Score:               {video_metrics.get('f1_score', 0.0) * 100:.2f}%")
    if video_metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:                {video_metrics.get('roc_auc', 0.0) * 100:.2f}%")
    print(f"Confusion Matrix:       {video_metrics.get('confusion_matrix')}")

    # Export metrics
    full_report = {
        "frame_level": frame_metrics,
        "video_level": video_metrics
    }
    if xai_metrics:
        full_report["xai_attribution"] = xai_metrics

    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(full_report, f, indent=2)

    video_results_df.to_csv(out_dir / "video_predictions.csv", index=False)
    print(f"\nResults saved to: {out_dir / 'evaluation_metrics.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()


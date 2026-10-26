"""
src/evaluation/metrics.py
Quantitative evaluation for frame-level and video-level deepfake predictions.
Includes temporal mean pooling aggregation, ROC-AUC, F1, precision, recall, and confusion matrix.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_frame_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob_fake: List[float]
) -> Dict[str, Any]:
    """
    Computes standard classification metrics at the individual image/frame level.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "total_frames": len(y_true),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4)
    }

    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob_fake)), 4)
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics


def aggregate_video_predictions(
    frame_results: List[Dict[str, Any]],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Aggregates frame-level fake probabilities into a single video-level score
    using temporal mean pooling.
    S_video = (1/M) * sum(P_fake(frame_i))
    """
    df = pd.DataFrame(frame_results)
    if df.empty:
        return pd.DataFrame()

    aggregated = []
    for video_id, group in df.groupby("video_id"):
        true_label = group["true_label"].iloc[0]
        mean_fake_score = group["fake_prob"].mean()
        pred_label = int(mean_fake_score >= threshold)

        aggregated.append({
            "video_id": video_id,
            "true_label": int(true_label),
            "fake_score": round(float(mean_fake_score), 4),
            "predicted_label": pred_label,
            "frames_evaluated": len(group)
        })

    return pd.DataFrame(aggregated)


def compute_video_metrics(video_results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes performance metrics across aggregated video samples.
    """
    if video_results_df.empty:
        return {}

    y_true = video_results_df["true_label"].tolist()
    y_pred = video_results_df["predicted_label"].tolist()
    y_score = video_results_df["fake_score"].tolist()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "videos_evaluated": len(video_results_df),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4)
    }

    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Real", "Fake"], output_dict=True, zero_division=0
    )

    return metrics


def compute_pointing_game_accuracy(
    saliency_maps: List[np.ndarray],
    ground_truth_masks: List[np.ndarray]
) -> Dict[str, Any]:
    """
    Computes Pointing Game Accuracy for explainability / saliency maps against ground truth forgery masks.
    A hit is scored if the maximum saliency coordinate (y, x) lies inside the ground-truth manipulated region:
        Hit = 1 if Mask_gt(y_max, x_max) > 0 else 0
    Pointing Game Accuracy = Hits / Total_Valid_Samples
    """
    hits = 0
    total_evaluated = 0

    for saliency, mask in zip(saliency_maps, ground_truth_masks):
        if saliency is None or mask is None:
            continue

        # Ensure 2D arrays
        saliency_2d = np.squeeze(saliency)
        mask_2d = np.squeeze(mask)

        # Ignore pristine masks with no manipulated pixels
        if np.sum(mask_2d > 0) == 0:
            continue

        total_evaluated += 1
        peak_idx = np.unravel_index(np.argmax(saliency_2d), saliency_2d.shape)
        peak_y, peak_x = peak_idx[0], peak_idx[1]

        if mask_2d[peak_y, peak_x] > 0:
            hits += 1

    acc = (hits / total_evaluated) if total_evaluated > 0 else 0.0
    return {
        "pointing_game_accuracy": round(float(acc), 4),
        "total_evaluated": total_evaluated,
        "hits": hits,
        "misses": total_evaluated - hits
    }


def compute_mask_iou(
    saliency_maps: List[np.ndarray],
    ground_truth_masks: List[np.ndarray],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes Intersection-over-Union (IoU) between binarized saliency maps and ground-truth forgery masks:
        IoU = |Saliency_bin \\cap Mask_gt| / |Saliency_bin \\cup Mask_gt|
    """
    ious = []

    for saliency, mask in zip(saliency_maps, ground_truth_masks):
        if saliency is None or mask is None:
            continue

        saliency_2d = np.squeeze(saliency)
        mask_2d = np.squeeze(mask)

        # Skip pristine samples with no manipulation ground truth
        if np.sum(mask_2d > 0) == 0:
            continue

        # Normalize saliency to [0, 1] if not already
        s_min, s_max = saliency_2d.min(), saliency_2d.max()
        if s_max > s_min:
            s_norm = (saliency_2d - s_min) / (s_max - s_min)
        else:
            s_norm = saliency_2d

        bin_saliency = (s_norm >= threshold).astype(bool)
        bin_mask = (mask_2d > 0).astype(bool)

        intersection = np.logical_and(bin_saliency, bin_mask).sum()
        union = np.logical_or(bin_saliency, bin_mask).sum()

        iou = (intersection / union) if union > 0 else 0.0
        ious.append(iou)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {
        "mean_iou": round(mean_iou, 4),
        "samples_evaluated": len(ious),
        "threshold_used": threshold
    }



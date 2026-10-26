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

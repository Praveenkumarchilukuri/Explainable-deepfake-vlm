"""
train.py
Fine-tunes a Vision Transformer (ViT-B/16) for binary facial deepfake detection.
Implements video-level stratified splitting, AdamW optimization, and checkpoint preservation.
"""

import argparse
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import (
    VIT_MODEL_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    RANDOM_SEED
)
from src.data.dataset import (
    build_video_dataframe,
    split_video_dataframe,
    create_dataloaders
)
from src.models.vit_detector import ViTDeepfakeClassifier


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _ in tqdm(loader, leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train ViT deepfake detector")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--model_name", type=str, default=VIT_MODEL_NAME, help="Model architecture")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Checkpoint directory")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_vit_model.pth"

    print("=" * 60)
    print("STARTING VISION TRANSFORMER TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device:           {DEVICE}")
    print(f"Data directory:   {args.data_dir}")
    print(f"Architecture:     {args.model_name}")
    print(f"Epochs:           {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")

    # Build video dataset
    df = build_video_dataframe(Path(args.data_dir))
    if df.empty:
        raise ValueError(f"No valid video frames found in {args.data_dir}. Run 'python download_data.py' first.")

    print(f"Total videos identified: {len(df)} across classes {df['class_name'].value_counts().to_dict()}")

    train_df, val_df, test_df = split_video_dataframe(df, seed=RANDOM_SEED)
    print(f"Video Partition -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    # Initialize model
    model = ViTDeepfakeClassifier(model_name=args.model_name, pretrained=True)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4)
        })

        print(f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                best_model_path,
                extra_meta={"best_val_loss": best_val_loss, "val_accuracy": val_acc}
            )
            print(f"  -> Checkpoint updated: {best_model_path.name}")

    # Save training logs
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump({"epochs": args.epochs, "best_val_loss": best_val_loss, "history": history}, f, indent=2)

    print("=" * 60)
    print(f"Training successfully finished. Model saved to: {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

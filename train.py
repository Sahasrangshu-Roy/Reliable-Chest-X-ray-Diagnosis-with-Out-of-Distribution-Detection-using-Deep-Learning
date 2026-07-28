#!/usr/bin/env python
import argparse
import os
import torch
from src.dataset import load_nih_data, create_dataloaders, calculate_class_weights
from src.model import get_densenet_model, get_weighted_bce_loss, save_model


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 for Chest X-ray Pathology Classification")
    parser.add_argument("--data_dir", type=str, default=".", help="Path to NIH ChestX-ray14 dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for Adam optimizer")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--weight_cap", type=float, default=50.0, help="Maximum cap for class imbalance weights")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to save the best model weights")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading NIH dataset from: {args.data_dir}")
    train_df, val_df, test_df, _ = load_nih_data(args.data_dir)
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # 2. Compute Class Weights
    _, capped_weights = calculate_class_weights(train_df, cap=args.weight_cap)

    # 3. Create DataLoaders
    train_loader, val_loader, _, _ = create_dataloaders(
        train_df, val_df, test_df, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # 4. Initialize Model & Loss & Optimizer
    model = get_densenet_model(num_classes=14, pretrained=True, device=device)
    criterion = get_weighted_bce_loss(capped_weights, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("\nStarting Training...")
    print("-" * 60)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:02d}/{args.epochs:02d} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation Pass
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoint & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(model, args.checkpoint)
            print(f"  ✓ Saved new best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{args.patience} epochs")

        print("-" * 60)

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()

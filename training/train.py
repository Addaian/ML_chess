#!/usr/bin/env python3
"""Train the neural network evaluator and export weights for runtime use.

Usage:
    python3 -m training.train
    python3 -m training.train --data training/data/parsed --epochs 10 --batch-size 4096
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from chess_engine.nn.model import get_chess_net, export_weights
from training.config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    TRAIN_DATA_DIR, WEIGHTS_PATH, CHECKPOINT_PATH,
)


class ChessDataset(Dataset):
    def __init__(self, data_dir: str):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"No .npz chunks found in {data_dir}")
        features_list, labels_list = [], []
        for path in paths:
            data = np.load(path)
            features_list.append(data['features'])
            labels_list.append(data['labels'])
        self.features = torch.from_numpy(np.concatenate(features_list))
        self.labels = torch.from_numpy(np.concatenate(labels_list))
        print(f"Loaded {len(self.features):,} positions from {len(paths)} chunks")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train(data_dir: str, epochs: int, batch_size: int):
    dataset = ChessDataset(data_dir)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False, num_workers=0)

    model = get_chess_net()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=False)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    print(f"\nTraining {len(train_ds):,} positions, validating {len(val_ds):,}")
    print(f"Epochs: {epochs}  Batch: {batch_size}  LR: {LEARNING_RATE}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(features)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                preds = model(features)
                val_loss += criterion(preds, labels).item() * len(features)
        val_loss /= val_size

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        marker = " ← best" if val_loss < best_val_loss else ""
        print(f"Epoch {epoch:2d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}  lr={lr:.1e}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(CHECKPOINT_PATH) or '.', exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Load best checkpoint and export
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    os.makedirs(os.path.dirname(WEIGHTS_PATH) or '.', exist_ok=True)
    export_weights(model, WEIGHTS_PATH)
    print(f"\nWeights exported to: {WEIGHTS_PATH}")
    print(f"Best val loss     : {best_val_loss:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the chess NN evaluator")
    parser.add_argument("--data", default=os.path.join(TRAIN_DATA_DIR, "parsed"),
                        help="Directory containing .npz chunks")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size)

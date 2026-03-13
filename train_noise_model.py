#!/usr/bin/env python3
"""
Train the noise-prediction PyTorch MLP on noisecapture_prepared.parquet.
Uses wandb for logging. Saves checkpoint and preprocess (scaler, encoders) for inference/heatmap.

Usage:
  python train_noise_model.py --data data_prep/noisecapture_prepared.parquet --epochs 30 --out-dir checkpoints
  WANDB_PROJECT=noise-map python train_noise_model.py --data noisecapture_prepared.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
except ImportError:
    wandb = None


def load_and_prepare(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["AEF_embed"]).copy()
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    aef_cols = [f"A{b:02d}" for b in range(64)]
    aef = np.stack(df["AEF_embed"].values)
    for i, c in enumerate(aef_cols):
        df[c] = aef[:, i]

    le_country = LabelEncoder()
    df["country_idx"] = le_country.fit_transform(df["country"].astype(str))
    country_classes = list(le_country.classes_)
    feature_cols = aef_cols + ["lat", "lon", "hour_sin", "hour_cos"]
    X_num = df[feature_cols].values
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    country_onehot = ohe.fit_transform(df[["country_idx"]])
    X = np.hstack([X_num, country_onehot]).astype(np.float32)
    y = df["noise_level_dB"].values.astype(np.float32)

    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=42,
            stratify=df["country_idx"],
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42
        )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    return (
        X_train_s,
        y_train,
        X_test_s,
        y_test,
        scaler,
        ohe,
        le_country,
        country_classes,
        aef_cols,
        feature_cols,
        len(X[0]),
    )


class NoiseMLP(nn.Module):
    def __init__(self, n_in, hidden=(256, 256, 128, 64)):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.BatchNorm1d(h)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    p = argparse.ArgumentParser(description="Train noise prediction MLP with wandb")
    p.add_argument("--data", type=Path, default=Path("data_prep/noisecapture_prepared.parquet"), help="Parquet path")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128, 64])
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--wandb-project", type=str, default="noise-map")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")
    (
        X_train_s,
        y_train,
        X_test_s,
        y_test,
        scaler,
        ohe,
        le_country,
        country_classes,
        aef_cols,
        feature_cols,
        n_features,
    ) = load_and_prepare(args.data)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train.reshape(-1, 1)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_s),
        torch.from_numpy(y_test.reshape(-1, 1)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    config = {
        "n_features": n_features,
        "hidden": args.hidden,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    if wandb:
        wandb.init(project=args.wandb_project, config=config, name="noise_mlp")

    model = NoiseMLP(n_features, tuple(args.hidden)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.squeeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.squeeze(1).to(device)
                test_loss += criterion(model(xb), yb).item() * xb.size(0)
        test_loss /= len(test_loader.dataset)
        test_rmse = np.sqrt(test_loss)

        if wandb:
            wandb.log(
                {"train_loss": train_loss, "test_loss": test_loss, "test_rmse": test_rmse, "epoch": epoch}
            )
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  test_rmse={test_rmse:.2f} dB")

    model.load_state_dict(best_state)
    if wandb:
        wandb.finish()

    torch.save(
        {"model_state": model.state_dict(), "config": config, "n_features": n_features},
        out_dir / "noise_mlp_checkpoint.pt",
    )
    import joblib
    joblib.dump(
        {
            "scaler": scaler,
            "ohe": ohe,
            "le_country": le_country,
            "country_classes": country_classes,
            "aef_cols": aef_cols,
            "feature_cols": feature_cols,
        },
        out_dir / "noise_mlp_preprocess.joblib",
    )
    print(f"Checkpoint saved to {out_dir / 'noise_mlp_checkpoint.pt'}")
    print(f"Preprocess saved to {out_dir / 'noise_mlp_preprocess.joblib'}")


if __name__ == "__main__":
    main()

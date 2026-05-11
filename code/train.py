"""
train.py
--------
Training loop for all three model types.

Multi-task loss:
    L_total = λ_rating * L_MSE(rating) + λ_theme * L_CE(theme)

Supports:
  - Early stopping on validation MAE
  - LR scheduling (ReduceLROnPlateau)
  - Checkpoint saving (best model by val MAE)
  - Loss / metric history returned for plotting
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm  # to see the progress in colab
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


# loss weights
LAMBDA_RATING = 1.0
LAMBDA_THEME  = 0.1         


# helpers
def _rating_metrics(pred: np.ndarray, true: np.ndarray, rating_min=1000, rating_max=2500):
    """Return MAE and RMSE in original rating scale."""
    scale = rating_max - rating_min
    pred_r = pred * scale + rating_min
    true_r = true * scale + rating_min
    mae  = np.mean(np.abs(pred_r - true_r))
    rmse = np.sqrt(np.mean((pred_r - true_r) ** 2))
    return mae, rmse


def _theme_metrics(logits: np.ndarray, true: np.ndarray):
    pred = np.argmax(logits, axis=1)
    acc  = accuracy_score(true, pred)
    f1   = f1_score(true, pred, average='macro', zero_division=0)
    return acc, f1


# single epoch 
def _run_epoch(model, loader, optimizer, device, model_type, is_train):
    model.train() if is_train else model.eval()

    mse_loss_fn = nn.MSELoss()
    ce_loss_fn  = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_rating_pred, all_rating_true = [], []
    all_theme_logits, all_theme_true = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for boards, move_counts, ratings, themes in tqdm(loader, desc='train' if is_train else 'val', leave=False):
            boards  = boards.to(device)
            move_counts = move_counts.to(device)
            ratings = ratings.to(device)
            themes  = themes.to(device)

            if model_type == 'single':
                rating_pred  = model(boards, move_counts)
                loss = mse_loss_fn(rating_pred, ratings)
                theme_logits = None
            elif model_type == 'multi':
                rating_pred, theme_logits = model(boards, move_counts)
                loss = (
                    LAMBDA_RATING * mse_loss_fn(rating_pred, ratings)
                    + LAMBDA_THEME * ce_loss_fn(theme_logits, themes)
                )
            else:  # mlp — boards treated as flat features externally
                rating_pred, theme_logits = model(boards, move_counts)
                loss = (
                    LAMBDA_RATING * mse_loss_fn(rating_pred, ratings)
                    + LAMBDA_THEME * ce_loss_fn(theme_logits, themes)
                )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * boards.size(0)
            all_rating_pred.append(rating_pred.detach().cpu().numpy())
            all_rating_true.append(ratings.detach().cpu().numpy())
            if theme_logits is not None:
                all_theme_logits.append(theme_logits.detach().cpu().numpy())
                all_theme_true.append(themes.detach().cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / n

    rating_pred_np = np.concatenate(all_rating_pred)
    rating_true_np = np.concatenate(all_rating_true)
    mae, rmse = _rating_metrics(rating_pred_np, rating_true_np)

    metrics = {'loss': avg_loss, 'mae': mae, 'rmse': rmse}

    if all_theme_logits:
        logits_np = np.concatenate(all_theme_logits)
        true_np   = np.concatenate(all_theme_true)
        acc, f1   = _theme_metrics(logits_np, true_np)
        metrics.update({'theme_acc': acc, 'theme_f1': f1})

    return metrics


# main training function
def train(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    model_type:   str   = 'multi',     # 'mlp' | 'single' | 'multi'
    epochs:       int   = 50,
    lr:           float = 1e-3,
    patience:     int   = 8,
    save_path:    str   = 'best_model.pt',
    device:       str   = None,
):
    """
    Train the model and return history dict.

    Args:
        model       : nn.Module
        train_loader: DataLoader
        val_loader  : DataLoader
        model_type  : 'mlp' | 'single' | 'multi'
        epochs      : max epochs
        lr          : initial learning rate
        patience    : early-stopping patience (on val MAE)
        save_path   : where to save the best checkpoint
        device      : 'cuda' | 'cpu' | None (auto-detect)

    Returns:
        history : dict with lists 'train_*' and 'val_*' per epoch
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=2, min_lr=1e-6
    )

    history = {k: [] for k in [
        'train_loss', 'train_mae', 'train_rmse',
        'val_loss',   'val_mae',   'val_rmse',
        'train_theme_acc', 'train_theme_f1',
        'val_theme_acc',   'val_theme_f1',
    ]}

    best_val_mae = float('inf')
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_m = _run_epoch(model, train_loader, optimizer, device, model_type, is_train=True)
        val_m   = _run_epoch(model, val_loader,   optimizer, device, model_type, is_train=False)

        scheduler.step(val_m['mae'])

        # Store history
        for k, v in train_m.items():
            history[f'train_{k}'].append(v)
        for k, v in val_m.items():
            history[f'val_{k}'].append(v)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss {train_m['loss']:.4f} MAE {train_m['mae']:.1f} | "
            f"Val Loss {val_m['loss']:.4f} MAE {val_m['mae']:.1f} | "
            f"{elapsed:.1f}s"
            + (f" | ThemeAcc {val_m.get('theme_acc', 0):.3f} F1 {val_m.get('theme_f1', 0):.3f}"
               if 'theme_acc' in val_m else "")
        )

        # Early stopping
        if val_m['mae'] < best_val_mae:
            best_val_mae = val_m['mae']
            no_improve   = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val MAE={best_val_mae:.2f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[train] Early stopping at epoch {epoch} (patience={patience})")
                break

    # Reload best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"[train] Training complete. Best val MAE: {best_val_mae:.2f}")
    return history

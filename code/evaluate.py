"""
Evaluate a trained model on the test set and produce:
  - Rating: MAE, RMSE
  - Theme : Accuracy, Precision, Recall, F1 (weighted)
  - Confusion matrix
  - Sample failure cases (worst-predicted puzzles)
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import chess
import chess.svg


# evaluation
@torch.no_grad()
def evaluate(model, loader, model_type: str, device: str = None, label_names: list = None):
    """
    Run model on all batches in loader and compute metrics.

    Returns:
        results : dict with all metrics + raw arrays for further analysis
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    all_rating_pred, all_rating_true = [], []
    all_theme_logits, all_theme_true = [], []

    for boards, move_counts, ratings, themes_int in loader:
        boards      = boards.to(device)
        ratings     = ratings.to(device)
        move_counts = move_counts.to(device)
        themes_int  = themes_int.to(device)

        if model_type == 'single':
            rating_pred  = model(boards, move_counts)
            theme_logits = None
        else:
            rating_pred, theme_logits = model(boards, move_counts)

        all_rating_pred.append(rating_pred.cpu().numpy())
        all_rating_true.append(ratings.cpu().numpy())

        if theme_logits is not None:
            all_theme_logits.append(theme_logits.cpu().numpy())
            all_theme_true.append(themes_int.cpu().numpy())

    # De-normalise ratings (1000–2500 scale)
    scale, offset = 1500, 1000
    pred_r = np.concatenate(all_rating_pred) * scale + offset
    true_r = np.concatenate(all_rating_true) * scale + offset

    mae  = float(np.mean(np.abs(pred_r - true_r)))
    rmse = float(np.sqrt(np.mean((pred_r - true_r) ** 2)))

    results = {
        'mae': mae, 'rmse': rmse,
        'rating_pred': pred_r, 'rating_true': true_r,
    }

    if all_theme_logits:
        logits = np.concatenate(all_theme_logits)
        true   = np.concatenate(all_theme_true)
        preds  = np.argmax(logits, axis=1)

        results.update({
            'theme_acc': float(accuracy_score(true, preds)),
            'theme_f1_macro': float(f1_score(true, preds, average='macro', zero_division=0)),
            'theme_f1_weighted': float(f1_score(true, preds, average='weighted', zero_division=0)),
            'theme_pred': preds,
            'theme_true': true,
            'theme_logits': logits,
        })
        results['theme_f1'] = results['theme_f1_macro']

        report = classification_report(
            true, preds,
            target_names=label_names or [str(i) for i in range(logits.shape[1])],
            zero_division=0,
        )
        results['classification_report'] = report
        print(report)

    print(f"\n{'='*40}")
    print(f"  Rating MAE  : {mae:.2f} Elo")
    print(f"  Rating RMSE : {rmse:.2f} Elo")
    if 'theme_acc' in results:
        print(f"  Theme Acc   : {results['theme_acc']:.4f}")
        print(f"  Theme F1 macro    : {results['theme_f1_macro']:.4f}")
        print(f"  Theme F1 weighted : {results['theme_f1_weighted']:.4f}")
    print(f"{'='*40}\n")

    return results


# plots
def plot_training_history(history: dict, title: str = 'Training History', save_path: str = None):
    """Plot loss and MAE curves for train and validation."""
    epochs = range(1, len(history['train_loss']) + 1)

    has_theme = bool(history.get('val_theme_acc'))
    n_plots   = 3 if has_theme else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')

    # MAE
    axes[1].plot(epochs, history['train_mae'], label='Train')
    axes[1].plot(epochs, history['val_mae'],   label='Val')
    axes[1].set_title('Rating MAE (Elo)'); axes[1].legend(); axes[1].set_xlabel('Epoch')

    # Theme accuracy (if available)
    if has_theme:
        axes[2].plot(epochs, history.get('train_theme_acc', []), label='Train')
        axes[2].plot(epochs, history.get('val_theme_acc',   []), label='Val')
        axes[2].set_title('Theme Accuracy'); axes[2].legend(); axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(results: dict, label_names: list, title: str = '', save_path: str = None):
    """Plot confusion matrix for theme classification."""
    if 'theme_pred' not in results:
        print("No theme predictions found.")
        return

    cm = confusion_matrix(results['theme_true'], results['theme_pred'])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title or 'Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_rating_scatter(results: dict, title: str = '', save_path: str = None):
    """Scatter plot: predicted vs true Elo rating."""
    pred_r = results['rating_pred']
    true_r = results['rating_true']

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true_r, pred_r, alpha=0.3, s=10, color='steelblue')
    lo = min(true_r.min(), pred_r.min()) - 50
    hi = max(true_r.max(), pred_r.max()) + 50
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlabel('True Rating'); ax.set_ylabel('Predicted Rating')
    ax.set_title(title or 'Rating Prediction')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def failure_cases(results: dict, df_test: pd.DataFrame, n: int = 5, label_names: list = None):
    """
    Print the n puzzles with the largest rating prediction error.
    """
    errors = np.abs(results['rating_pred'] - results['rating_true'])
    worst  = np.argsort(errors)[::-1][:n]

    print(f"\n{'='*60}")
    print(f"  Top {n} Worst Rating Predictions")
    print(f"{'='*60}")
    for rank, idx in enumerate(worst, 1):
        row = df_test.iloc[idx]
        print(
            f"  #{rank}  FEN: {row.get('fen','?')}\n"
            f"       True={results['rating_true'][idx]:.0f}  "
            f"Pred={results['rating_pred'][idx]:.0f}  "
            f"Error={errors[idx]:.0f} Elo\n"
            f"       Theme: {row.get('primary_theme','?')}"
        )
        if label_names and 'theme_pred' in results:
            pred_theme = label_names[results['theme_pred'][idx]]
            true_theme = label_names[results['theme_true'][idx]]
            print(f"       ThemePred={pred_theme}  ThemeTrue={true_theme}")
        print()

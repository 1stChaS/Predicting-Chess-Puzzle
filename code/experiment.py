"""
Full experimental pipeline comparing:
  1. MLPBaseline    (Baseline 1)
  2. SingleTaskCNN  (Baseline 2)
  3. MultiTaskCNN   (Main model — 4 themes)

Then runs the ablation study (2 vs 4 themes).
"""

import os
import argparse
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
hf_data = load_dataset("Lichess/chess-puzzles", split="train")
from data.dataset   import build_dataloaders, SUBSET_4
from models.models  import build_model
from train          import train
from evaluate       import (evaluate, plot_training_history,
                             plot_confusion_matrix, plot_rating_scatter,
                             failure_cases)


OUT_DIR = 'experiment_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# helpers
def run_one(model_type, hf_dataset, themes, epochs, batch_size, lr, patience, label_names):
    tag = model_type
    print(f"\n{'='*60}\n  MODEL: {tag.upper()}\n{'='*60}")

    train_loader, val_loader, test_loader, le, num_themes = build_dataloaders(
        hf_dataset, themes=themes, batch_size=batch_size
    )

    model = build_model(model_type, num_themes=num_themes)

    save_path = os.path.join(OUT_DIR, f'best_{tag}.pt')
    history   = train(
        model, train_loader, val_loader,
        model_type=model_type,
        epochs=    epochs,
        lr=        lr,
        patience=  patience,
        save_path= save_path,
        device=    DEVICE,
    )

    plot_training_history(
        history,
        title=    f'Training History — {tag}',
        save_path=os.path.join(OUT_DIR, f'history_{tag}.png'),
    )

    results = evaluate(
        model, test_loader,
        model_type=  model_type,
        device=      DEVICE,
        label_names= label_names,
    )

    plot_rating_scatter(
        results,
        title=    f'Rating Prediction — {tag}',
        save_path=os.path.join(OUT_DIR, f'scatter_{tag}.png'),
    )

    if 'theme_pred' in results:
        plot_confusion_matrix(
            results,
            label_names= label_names,
            title=       f'Confusion Matrix — {tag}',
            save_path=   os.path.join(OUT_DIR, f'cm_{tag}.png'),
        )

    with open(os.path.join(OUT_DIR, f'history_{tag}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return {
        'model':      tag,
        'mae':        round(results['mae'],  2),
        'rmse':       round(results['rmse'], 2),
        'theme_acc':  round(results.get('theme_acc', float('nan')), 4),
        'theme_f1':   round(results.get('theme_f1',  float('nan')), 4),
    }


def final_comparison_plot(rows):
    """Bar chart: MAE comparison across all three models."""
    labels = [r['model'] for r in rows]
    maes   = [r['mae']   for r in rows]
    colors = ['#A8D8EA', '#AA96DA', '#FCBAD3']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, maes, color=colors[:len(rows)], width=0.5, zorder=3)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'{mae:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Rating MAE (Elo)', fontsize=12)
    ax.set_title('Model Comparison: Rating Prediction MAE', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'model_comparison_mae.png'), dpi=150)
    plt.show()


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        default='lichess_db_puzzle.csv')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int,   default=8)
    parser.add_argument('--skip_ablation', action='store_true',
                        help='Skip the ablation study')
    args = parser.parse_args()

    themes      = SUBSET_4
    label_names = themes

    rows = []
    for mtype in ['single', 'multi']:   # mlp needs flat features; keep CNN comparison clean
        row = run_one(mtype, hf_data, themes,
                      args.epochs, args.batch_size, args.lr,
                      args.patience, label_names)
        rows.append(row)

    # Summary table
    df = pd.DataFrame(rows)
    print('\n' + '='*60)
    print('  FINAL MODEL COMPARISON')
    print('='*60)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, 'model_comparison.csv'), index=False)

    final_comparison_plot(rows)

    # Ablation
    if not args.skip_ablation:
        print('\n\nStarting ablation study...')
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from experiments.ablation import run_ablation
        run_ablation()


if __name__ == '__main__':
    main()

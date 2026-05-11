"""
Ablation study: compare multi-task CNN performance with
  - 2 themes  (fork, mate)
  - 4 themes  (fork, pin, skewer, mate)
"""

import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets import load_dataset
hf_data = load_dataset("Lichess/chess-puzzles", split="train")
from dataset  import build_dataloaders, SUBSET_2, SUBSET_4
from models import MultiTaskCNN
from train         import train
from evaluate      import evaluate, plot_training_history, plot_confusion_matrix


# config
hf_dataset   = os.environ.get('LICHESS_CSV', 'lichess_db_puzzle.csv')
EPOCHS     = 20
BATCH_SIZE = 256
LR         = 1e-3
PATIENCE   = 8
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR    = 'ablation_outputs'
os.makedirs(OUT_DIR, exist_ok=True)


def run_experiment(
    themes: list[str],
    tag: str,
    hf_dataset=None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = PATIENCE,
):
    """Run one full train+eval cycle and return metrics dict."""
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {tag}  |  themes={themes}")
    print(f"{'#'*60}\n")

    train_loader, val_loader, test_loader, le, num_themes = build_dataloaders(
        hf_dataset, themes=themes, batch_size=batch_size
    )

    model = MultiTaskCNN(num_themes=num_themes, dropout=0.3)

    save_path = os.path.join(OUT_DIR, f'best_{tag}.pt')
    history   = train(
        model, train_loader, val_loader,
        model_type=  'multi',
        epochs=      epochs,
        lr=          lr,
        patience=    patience,
        save_path=   save_path,
        device=      DEVICE,
    )

    # Save training history plot
    plot_training_history(
        history,
        title=    f'Training History — {tag}',
        save_path=os.path.join(OUT_DIR, f'history_{tag}.png'),
    )

    # Evaluate on test set
    results = evaluate(
        model, test_loader,
        model_type=  'multi',
        device=      DEVICE,
        label_names= list(le.classes_),
    )

    # Confusion matrix
    plot_confusion_matrix(
        results,
        label_names= list(le.classes_),
        title=       f'Theme Confusion Matrix — {tag}',
        save_path=   os.path.join(OUT_DIR, f'cm_{tag}.png'),
    )

    # Save history json
    with open(os.path.join(OUT_DIR, f'history_{tag}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return {
        'experiment': tag,
        'num_themes': num_themes,
        'themes':     ', '.join(themes),
        'mae':        round(results['mae'],  2),
        'rmse':       round(results['rmse'], 2),
        'theme_acc':  round(results.get('theme_acc', 0), 4),
        'theme_f1_macro':   round(results.get('theme_f1_macro', 0), 4),
        'theme_f1_weighted': round(results.get('theme_f1_weighted', 0), 4),
    }


def compare_mae_bar(rows: list[dict]):
    """Bar chart comparing MAE across ablation conditions."""
    labels = [r['experiment'] for r in rows]
    maes   = [r['mae']   for r in rows]
    rmses  = [r['rmse']  for r in rows]

    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, maes, color=['#4C9BE8', '#E87B4C'], width=0.5, zorder=3)

    for bar, mae, rmse in zip(bars, maes, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'MAE={mae}\nRMSE={rmse}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Rating MAE (Elo)', fontsize=11)
    ax.set_title('Ablation Study: Effect of Theme Count on Rating Prediction', fontsize=12)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ablation_mae_comparison.png'), dpi=150)
    plt.show()


def compare_theme_f1_bar(rows: list[dict]):
    """Bar chart comparing theme F1 across conditions."""
    labels = [r['experiment'] for r in rows]
    f1s    = [r['theme_f1_macro'] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, f1s, color=['#4C9BE8', '#E87B4C'], width=0.5)
    ax.set_ylabel('Theme F1 (macro)', fontsize=11)
    ax.set_title('Theme Classification F1 by Experiment', fontsize=12)
    ax.set_ylim(0, 1)
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ablation_f1_comparison.png'), dpi=150)
    plt.show()


def run_ablation(
    hf_data,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = PATIENCE,
):
    """Run both theme-set experiments and produce comparison table."""
    experiments = [
        (SUBSET_2, '2themes_fork_mate'),
        (SUBSET_4, '4themes_all'),
    ]

    rows = []
    for themes, tag in experiments:
        row = run_experiment(
            themes,
            tag,
            hf_data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )
        rows.append(row)

    # Print summary table
    df_summary = pd.DataFrame(rows)
    print('\n' + '='*70)
    print('  ABLATION STUDY SUMMARY')
    print('='*70)
    print(df_summary.to_string(index=False))
    print('='*70)

    df_summary.to_csv(os.path.join(OUT_DIR, 'ablation_results.csv'), index=False)
    print(f"\n[ablation] Results saved to {OUT_DIR}/ablation_results.csv")

    # Visualise
    compare_mae_bar(rows)
    compare_theme_f1_bar(rows)

    # Interpretation
    mae_2, mae_4 = rows[0]['mae'], rows[1]['mae']
    diff = mae_2 - mae_4
    if diff > 0:
        print(f"\n4-theme model is BETTER for rating prediction by {diff:.1f} Elo MAE.")
        print("Learning more tactical diversity helps the shared backbone encode difficulty.")
    elif diff < 0:
        print(f"\n2-theme model is BETTER for rating prediction by {abs(diff):.1f} Elo MAE.")
        print("Simpler theme set may provide cleaner gradient signal for rating head.")
    else:
        print("\n🟰 No difference between 2-theme and 4-theme conditions.")

    return df_summary


if __name__ == '__main__':
    run_ablation()

"""
Loads the Lichess puzzle CSV, filters by theme and rating range,
encodes FEN strings into (12+4) = 16-plane board tensors, and
returns PyTorch Dataset / DataLoader objects.

Plane layout (dim 0):
  0-5   : White pieces  (P N B R Q K)
  6-11  : Black pieces  (p n b r q k)
  12    : Side to move  (1 = white, 0 = black, broadcast over board)
  13    : White castling rights (any)
  14    : Black castling rights (any)
  15    : Material balance (clamped to [-1,1])
"""

import numpy as np
import pandas as pd
import chess
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# constants
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}
MATERIAL_VALUES = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}

ALL_THEMES    = ['fork', 'pin', 'skewer', 'mate']
SUBSET_2      = ['fork', 'mate']          # ablation: 2 themes
SUBSET_4      = ['fork', 'pin', 'skewer', 'mate']  # full: 4 themes

RATING_MIN    = 1000
RATING_MAX    = 2500
MIN_PLAYS     = 50
SAMPLE_CAP = 50000


# FEN encoder
def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert a FEN string → float32 tensor of shape (16, 8, 8)."""
    board = chess.Board(fen)
    planes = np.zeros((16, 8, 8), dtype=np.float32)

    # Piece planes
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            row = 7 - (sq // 8)
            col = sq % 8
            planes[PIECE_TO_PLANE[piece.symbol()], row, col] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling
    if board.has_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0

    # Material balance (white - black, clamped to [-1,1])
    material = 0
    total = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = MATERIAL_VALUES.get(piece.symbol().upper(), 0)
        material += val if piece.color == chess.WHITE else -val
        total += val
    if total > 0:
        planes[15, :, :] = np.clip(material / total, -1.0, 1.0)

    return torch.from_numpy(planes)


def _primary_theme(themes_val, allowed: list) -> str:
    if hasattr(themes_val, '__iter__') and not isinstance(themes_val, str):
        tags = [str(t).lower() for t in themes_val]
    else:
        tags = str(themes_val).lower().split()
    for allowed_tag in allowed:
        for tag in tags:
            if allowed_tag == 'mate':
                if 'mate' in tag:
                    return 'mate'
            else:
                if allowed_tag == tag:
                    return allowed_tag
    return None


def load_and_filter(hf_dataset, themes: list) -> pd.DataFrame:
    df = hf_dataset.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    print(f"[dataset] Total puzzles loaded : {len(df):,}")
    df = df[(df['rating'] >= RATING_MIN) & (df['rating'] <= RATING_MAX)]
    print(f"[dataset] After rating filter  : {len(df):,}")
    if 'nbplays' in df.columns:
        df = df[df['nbplays'] >= MIN_PLAYS]
    elif 'plays' in df.columns:
        df = df[df['plays'] >= MIN_PLAYS]
    print(f"[dataset] After plays filter   : {len(df):,}")
    df['primary_theme'] = df['themes'].apply(lambda t: _primary_theme(t, themes))
    df = df[df['primary_theme'].notna()].copy()
    print(f"[dataset] After theme filter   : {len(df):,}")
    df['move_count']  = df['moves'].apply(lambda m: len(str(m).split()) / 20.0)
    df['rating_norm'] = (df['rating'] - RATING_MIN) / (RATING_MAX - RATING_MIN)
    per_theme = SAMPLE_CAP // len(themes)
    df = df.groupby('primary_theme').apply(
        lambda g: g.sample(n=min(len(g), per_theme), random_state=42)
    ).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    print(f"[dataset] After sampling       : {len(df):,}")
    print(df['primary_theme'].value_counts().to_string())
    return df.reset_index(drop=True)

# Dataset 
class ChessPuzzleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder):
        self.df          = df.reset_index(drop=True)
        self.le          = label_encoder
        self.theme_labels = torch.tensor(self.le.transform(self.df['primary_theme'].values), dtype=torch.long)
        self.ratings      = torch.tensor(self.df['rating_norm'].values, dtype=torch.float32)
        self.move_counts  = torch.tensor(self.df['move_count'].values,  dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        board = fen_to_tensor(self.df.loc[idx, 'fen'])
        return board, self.move_counts[idx], self.ratings[idx], self.theme_labels[idx]


def build_dataloaders(hf_dataset, themes=SUBSET_4, batch_size=256,
                      val_size=0.15, test_size=0.15, seed=42):
    df = load_and_filter(hf_dataset, themes)
    le = LabelEncoder()
    le.fit(themes)
    train_val, test_df = train_test_split(df, test_size=test_size, stratify=df['primary_theme'], random_state=seed)
    train_df,  val_df  = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val['primary_theme'], random_state=seed)
    def make_loader(split_df, shuffle):
        ds = ChessPuzzleDataset(split_df.reset_index(drop=True), le)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)
    return make_loader(train_df, True), make_loader(val_df, False), make_loader(test_df, False), le, len(themes)

"""
Fine-tune XLM-RoBERTa-base on EmoBank for continuous VAD regression.

Output: a model that predicts (valence, arousal, dominance) as three
continuous floats from text — no discrete classes, no lookup tables.

Training data: EmoBank (10k English sentences, reader perspective)
Base model: FacebookAI/xlm-roberta-base (100 languages, 270M params)
Expected runtime: ~20 min on M4 Mac (CPU/MPS)
Expected Pearson r: ~0.65 valence, ~0.55 arousal, ~0.60 dominance
(matches published results on EmoBank with similar models)

Japanese transfer: zero-shot via XLM-RoBERTa multilingual pretraining.
Not explicitly trained on Japanese VAD but reasonable for clear affect.

Usage:
    python train_vad.py
    → saves model to ./vad_model/
    → update emotion_classifier.py to load from ./vad_model/
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from scipy.stats import pearsonr
import urllib.request

# ── Config ────────────────────────────────────────────────────────────

MODEL_NAME  = "FacebookAI/xlm-roberta-base"
OUTPUT_DIR  = "./vad_model"
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 2e-5
MAX_LEN     = 128
WARMUP_FRAC = 0.1
DEVICE      = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"Device: {DEVICE}")


# ── Download EmoBank ──────────────────────────────────────────────────

EMOBANK_URL = (
    "https://raw.githubusercontent.com/JULIELab/EmoBank/"
    "master/corpus/emobank.csv"
)
EMOBANK_PATH = "./emobank.csv"

def download_emobank():
    if not os.path.exists(EMOBANK_PATH):
        print("Downloading EmoBank...")
        urllib.request.urlretrieve(EMOBANK_URL, EMOBANK_PATH)
        print("  → downloaded")
    else:
        print("EmoBank already downloaded")

download_emobank()


# ── Dataset ───────────────────────────────────────────────────────────

class EmoBankDataset(Dataset):
    """
    EmoBank reader-perspective VAD dataset.

    We use the reader perspective (V, A, D) — how the reader perceives
    the emotion — rather than writer perspective. Reader perspective has
    higher inter-annotator agreement and is more relevant for our use
    case: we want to detect how the user's message feels to an observer,
    not what emotion the user intended to express.

    EmoBank scores are in [1, 5]. We normalise to:
        valence   → [-1, +1]  via (V - 3) / 2
        arousal   → [ 0,  1]  via (A - 1) / 4
        dominance → [ 0,  1]  via (D - 1) / 4
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts     = df["text"].tolist()
        # EmoBank columns: V, A, D (reader perspective)
        self.valence   = ((df["V"] - 3) / 2).tolist()      # → [-1, +1]
        self.arousal   = ((df["A"] - 1) / 4).tolist()      # → [0, 1]
        self.dominance = ((df["D"] - 1) / 4).tolist()      # → [0, 1]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels"        : torch.tensor(
                [self.valence[idx], self.arousal[idx], self.dominance[idx]],
                dtype=torch.float,
            ),
        }


# ── Model ─────────────────────────────────────────────────────────────

class VADRegressor(nn.Module):
    """
    XLM-RoBERTa + regression head for VAD.

    Architecture:
        XLM-RoBERTa [CLS] pooling
            ↓
        Dropout(0.1)
            ↓
        Linear(hidden → 3)
            ↓
        tanh  (valence)  → [-1, +1]
        sigmoid (arousal, dominance) → [0, 1]

    The output activations match our ARM-E expected ranges exactly.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden       = self.encoder.config.hidden_size  # 768 for base
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        cls = self.drop(cls)
        raw = self.head(cls)                   # [B, 3]

        valence   = torch.tanh(raw[:, 0:1])        # [-1, +1]
        arousal   = torch.sigmoid(raw[:, 1:2])     # [0, 1]
        dominance = torch.sigmoid(raw[:, 2:3])     # [0, 1]

        return torch.cat([valence, arousal, dominance], dim=1)  # [B, 3]


# ── Training ──────────────────────────────────────────────────────────

def pearson(preds, targets):
    """Mean Pearson r across V, A, D."""
    rs = []
    for i in range(3):
        r, _ = pearsonr(preds[:, i], targets[:, i])
        rs.append(r)
    return rs

def train():
    # Load data
    df = pd.read_csv(EMOBANK_PATH)
    # EmoBank has columns: id, split, V, A, D, text (reader perspective)
    # Some versions also have V.r, A.r, D.r — use V/A/D (reader)
    if "V.r" in df.columns:
        df = df.rename(columns={"V.r": "V", "A.r": "A", "D.r": "D"})

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    dev_df   = df[df["split"] == "dev"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)}  Dev: {len(dev_df)}  Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = EmoBankDataset(train_df, tokenizer, MAX_LEN)
    dev_ds   = EmoBankDataset(dev_df,   tokenizer, MAX_LEN)
    test_ds  = EmoBankDataset(test_df,  tokenizer, MAX_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = VADRegressor(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps  = len(train_dl) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # MSE loss — standard for regression
    # We weight valence slightly higher: it's the most important signal
    # for ARM-E quadrant assignment (drives Q1/Q2/Q3/Q4)
    loss_weights = torch.tensor([1.5, 1.0, 1.0]).to(DEVICE)

    best_dev_r = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)

            # Weighted MSE per dimension
            loss = ((preds - labels) ** 2 * loss_weights).mean()
            loss.backward()

            # Gradient clipping — important for stable fine-tuning
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        # ── Evaluate on dev ───────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_dl:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                preds = model(input_ids, attention_mask).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(batch["labels"].numpy())

        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        r_scores   = pearson(all_preds, all_labels)

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"loss={avg_loss:.4f}  "
              f"dev Pearson  V={r_scores[0]:.3f}  "
              f"A={r_scores[1]:.3f}  D={r_scores[2]:.3f}  "
              f"mean={np.mean(r_scores):.3f}")

        # Save best checkpoint
        if np.mean(r_scores) > best_dev_r:
            best_dev_r = np.mean(r_scores)
            best_epoch = epoch
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.encoder.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            torch.save(model.head.state_dict(), f"{OUTPUT_DIR}/head.pt")
            print(f"  → saved best model (epoch {epoch})")

    # ── Final test evaluation ─────────────────────────────────────────
    print(f"\nBest epoch: {best_epoch}  dev Pearson mean: {best_dev_r:.3f}")
    print("Evaluating on test set...")

    # Reload best model
    model = VADRegressor(MODEL_NAME).to(DEVICE)
    model.encoder = AutoModel.from_pretrained(OUTPUT_DIR).to(DEVICE)
    model.head.load_state_dict(torch.load(f"{OUTPUT_DIR}/head.pt", map_location=DEVICE))
    model.eval()

    print(f"Device: {DEVICE}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            preds = model(input_ids, attention_mask).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    r_scores   = pearson(all_preds, all_labels)

    print(f"\n=== Test Pearson r ===")
    print(f"  Valence   : {r_scores[0]:.4f}")
    print(f"  Arousal   : {r_scores[1]:.4f}")
    print(f"  Dominance : {r_scores[2]:.4f}")
    print(f"  Mean      : {np.mean(r_scores):.4f}")
    print(f"\nModel saved to {OUTPUT_DIR}/")
    print("Update emotion_classifier.py to load from this path.")

    # Quick sanity check on your actual use case sentences
    print("\n=== Sanity check on conversational sentences ===")
    test_sentences = [
        "I hate this! so complicated I can't understand",
        "This is absolutely fascinating! I finally understand!",
        "That makes sense, thanks.",
        "I'm confused and stuck. Nothing makes sense.",
        "こんにちは！日本語を勉強しています。",
        "また間違えた...本当に難しい。",
    ]

    for sent in test_sentences:
        enc = tokenizer(
            sent, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            pred = model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE)
            ).squeeze().cpu().tolist()
        print(f"  {sent[:50]:<50}  V={pred[0]:+.3f}  A={pred[1]:.3f}  D={pred[2]:.3f}")


if __name__ == "__main__":
    train()
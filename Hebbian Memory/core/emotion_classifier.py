"""
emotion_classifier.py
Continuous VAD classifier using fine-tuned XLM-RoBERTa.

Loads the model trained by train_vad.py (./vad_model/).
Falls back to the two-stage Hartmann approach if the fine-tuned
model is not found — so the system works before training completes.

Output: VADScore(valence, arousal, dominance, confidence)
    valence   : [-1, +1]  negative ← → positive
    arousal   : [ 0,  1]  calm ← → excited
    dominance : [ 0,  1]  controlled ← → in-control

Language support:
    Fine-tuned model: English (direct), Japanese (zero-shot transfer)
    Fallback model:   English only (Hartmann)
"""

import os
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
from dataclasses import dataclass
from typing import Optional


# ── Output dataclass ──────────────────────────────────────────────────

@dataclass
class VADScore:
    valence   : float   # [-1, +1]
    arousal   : float   # [0,  1]
    dominance : float   # [0,  1]
    confidence: float   # [0,  1]  distance from neutral
    raw_scores: dict    # debug info


# ── Fine-tuned VAD model ──────────────────────────────────────────────

class VADRegressor(nn.Module):
    """
    Same architecture as train_vad.py — must match exactly.
    XLM-RoBERTa [CLS] → Dropout → Linear(768, 3) → tanh/sigmoid
    """
    def __init__(self, encoder, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden       = encoder.config.hidden_size
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0, :])
        raw = self.head(cls)
        return torch.cat([
            torch.tanh(raw[:, 0:1]),
            torch.sigmoid(raw[:, 1:2]),
            torch.sigmoid(raw[:, 2:3]),
        ], dim=1)


# ── Fallback: Hartmann + NRC-VAD ─────────────────────────────────────
# Used when vad_model/ doesn't exist yet (before training).

NRC_VAD = {
    "joy"     : ( 0.98,  0.73,  0.72),
    "surprise": ( 0.45,  0.78,  0.35),
    "neutral" : ( 0.00,  0.10,  0.50),
    "sadness" : (-0.72,  0.28, -0.42),
    "fear"    : (-0.60,  0.82, -0.64),
    "disgust" : (-0.78,  0.52, -0.58),
    "anger"   : (-0.86,  0.88,  0.30),
}


# ── EmotionClassifier ─────────────────────────────────────────────────

class EmotionClassifier:
    """
    VAD classifier with automatic model selection.

    Priority:
        1. Fine-tuned XLM-RoBERTa from vad_model_path (if exists)
           → continuous VAD, EN + JA zero-shot transfer
        2. Hartmann + NRC-VAD fallback (EN only)
           → used before training completes

    Usage:
        clf   = EmotionClassifier()
        score = clf("I hate this! so confusing!")
        # score.valence ≈ -0.82, score.arousal ≈ 0.85, score.dominance ≈ 0.28
    """

    HARTMANN_ID = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(
        self,
        device         : str  = "auto",
        vad_model_path : str  = "./vad_model",
        max_len        : int  = 128,
        min_confidence : float = 0.15,
    ):
        self.max_len        = max_len
        self.min_confidence = min_confidence
        self._device        = self._resolve_device(device)
        self.vad_model_path = vad_model_path

        # Lazy-loaded
        self._finetuned_model     = None
        self._finetuned_tokenizer = None
        self._fallback_pipeline   = None
        self._mode                = None   # "finetuned" or "fallback"

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available():          return "cuda"
        return "cpu"

    def _load(self):
        if self._mode is not None:
            return

        if os.path.exists(self.vad_model_path) and \
           os.path.exists(os.path.join(self.vad_model_path, "head.pt")):
            # Load fine-tuned XLM-RoBERTa
            print(f"Loading fine-tuned VAD model from {self.vad_model_path}...")
            self._finetuned_tokenizer = AutoTokenizer.from_pretrained(
                self.vad_model_path
            )
            encoder = AutoModel.from_pretrained(self.vad_model_path)
            self._finetuned_model = VADRegressor(encoder)
            self._finetuned_model.head.load_state_dict(
                torch.load(
                    os.path.join(self.vad_model_path, "head.pt"),
                    map_location=self._device,
                )
            )
            self._finetuned_model.to(self._device)
            self._finetuned_model.eval()
            self._mode = "finetuned"
            print("  → fine-tuned VAD model ready (EN + JA zero-shot)")

        else:
            # Fallback to Hartmann
            print(f"vad_model/ not found — using Hartmann fallback (EN only).")
            print(f"Run train_vad.py to train the multilingual model.")
            device_id = -2 if self._device == "mps" else \
                         0  if self._device == "cuda" else -1
            self._fallback_pipeline = pipeline(
                "text-classification",
                model=self.HARTMANN_ID,
                return_all_scores=True,
                device=device_id,
            )
            self._mode = "fallback"
            print("  → Hartmann fallback ready (EN only)")

    def _predict_finetuned(self, text: str) -> VADScore:
        enc = self._finetuned_tokenizer(
            text, max_length=self.max_len,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            pred = self._finetuned_model(input_ids, attention_mask)
            pred = pred.squeeze().cpu().tolist()

        valence, arousal, dominance = pred[0], pred[1], pred[2]

        # Confidence: mean absolute deviation from neutral (0, 0.5, 0.5)
        confidence = (
            abs(valence) +
            abs(arousal   - 0.5) * 2 +
            abs(dominance - 0.5) * 2
        ) / 3.0

        return VADScore(
            valence   = float(valence),
            arousal   = float(arousal),
            dominance = float(dominance),
            confidence= float(confidence),
            raw_scores= {"mode": "finetuned", "v": valence,
                         "a": arousal, "d": dominance},
        )

    def _predict_fallback(self, text: str) -> VADScore:
        results = self._fallback_pipeline(text[:512])[0]
        scores  = {r["label"]: r["score"] for r in results}

        valence   = sum(scores.get(e, 0) * v for e, (v, _, _) in NRC_VAD.items())
        arousal   = sum(scores.get(e, 0) * a for e, (_, a, _) in NRC_VAD.items())
        dominance = sum(scores.get(e, 0) * d for e, (_, _, d) in NRC_VAD.items())

        probs      = torch.tensor(list(scores.values()))
        entropy    = -(probs * probs.log().clamp(min=-10)).sum()
        max_entropy = torch.log(torch.tensor(float(len(probs))))
        confidence  = float(1.0 - (entropy / max_entropy))

        if confidence < self.min_confidence:
            valence   *= 0.3
            arousal    = 0.5 + (arousal   - 0.5) * 0.3
            dominance  = 0.5 + (dominance - 0.5) * 0.3

        return VADScore(
            valence=float(valence), arousal=float(arousal),
            dominance=float(dominance), confidence=confidence,
            raw_scores={"mode": "fallback", **scores},
        )

    def predict(self, text: str) -> VADScore:
        if not text or not text.strip():
            return VADScore(0.0, 0.5, 0.5, 0.0, {"mode": "empty"})
        self._load()
        if self._mode == "finetuned":
            return self._predict_finetuned(text)
        return self._predict_fallback(text)

    def __call__(self, text: str) -> VADScore:
        return self.predict(text)

    @property
    def mode(self) -> str:
        return self._mode or "not loaded"


# ── ARM-E integration ─────────────────────────────────────────────────

def make_arme_emotion_fn(classifier: EmotionClassifier):
    """Drop-in replacement for ARME._stub_classifier()."""
    def _classify(text: str):
        score = classifier.predict(text)
        return score.valence, score.arousal
    return _classify


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    clf = EmotionClassifier(device="auto", vad_model_path="./core/vad_model")

    test_cases = [
        ("I hate this! so complicated I can't understand",
         "Q2: anger/frustration → V<0, A high, D moderate"),
        ("This is absolutely fascinating! I finally understand!",
         "Q1: joy → V≈+0.98, A≈0.73, D≈+0.72"),
        ("That makes sense, thanks.",
         "Q4: calm satisfaction → V>0, A low"),
        ("I'm confused and stuck. Nothing makes sense.",
         "Q2/Q3: fear/sadness → V<0"),
        ("こんにちは！日本語を勉強しています。",
         "Q1/Q4: greeting + study → positive"),
        ("また間違えた...本当に難しい。",
         "Q3: mild frustration → V<0, A moderate"),
        ("ok",
         "low confidence: short ambiguous"),
    ]

    print(f"Mode: {clf.mode if clf._mode else 'not loaded yet'}")
    print(f"\n{'Text':<48} {'V':>7} {'A':>7} {'D':>7} {'conf':>6}")
    print("-" * 80)

    try:
        from arme_v2 import compute_e_t_quadrant
        has_arme = True
    except ImportError:
        has_arme = False

    for text, note in test_cases:
        score = clf(text)
        short = (text[:45] + "..") if len(text) > 47 else text

        q = ""
        if has_arme:
            em = compute_e_t_quadrant(score.valence, score.arousal)
            q  = f" [{em.quadrant}]"

        print(f"{short:<48} {score.valence:>+7.3f} {score.arousal:>7.3f} "
              f"{score.dominance:>+7.3f} {score.confidence:>6.3f}{q}")
        print(f"  → {note}")
        print()

    print(f"\nMode used: {clf.mode}")
    if clf.mode == "fallback":
        print("Run 'python train_vad.py' to train the multilingual model.")
        print("After training, restart — emotion_classifier.py will auto-load it.")
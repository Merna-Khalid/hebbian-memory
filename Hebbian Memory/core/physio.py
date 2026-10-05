"""
physio.py
Heart-rate → arousal mapping, and a physiology-fused emotion classifier
for the second-brain ambient path.

Text sentiment is a reasonable valence proxy but a weak arousal proxy —
arousal is fundamentally physiological (Kensinger 2004, see arme_v2.py's
docstring). A BCI's heart-rate channel is a much more direct arousal
signal than word choice, so PhysioFusedEmotionClassifier keeps valence
(and dominance) from the existing text classifier but overrides arousal
from a live heart-rate reading.
"""

import math
import os
import sys
from typing import Optional

# core/ modules import each other bare — see tutor_engine.py's docstring.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_classifier import VADScore   # noqa: E402
from signals import PhysioSample, SignalSource  # noqa: E402


def hr_to_arousal(
    hr_bpm      : Optional[float],
    baseline_bpm: float = 68.0,
    sensitivity : float = 25.0,
) -> float:
    """
    Normalized deviation of heart rate from a resting baseline, mapped
    to [0, 1] via tanh so large deviations saturate instead of clipping
    hard. hr_bpm == baseline_bpm → 0.5 (matches VADScore's neutral
    default), missing hr_bpm → 0.5 (neutral, no signal).

    sensitivity: bpm deviation that maps to ~0.88 arousal (tanh(1)).
    A real device would calibrate baseline_bpm per person; for the
    simulator each scenario supplies its own target around a shared
    resting baseline.
    """
    if hr_bpm is None:
        return 0.5
    deviation = (hr_bpm - baseline_bpm) / sensitivity
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(deviation)))


class PhysioFusedEmotionClassifier:
    """
    Drop-in replacement for EmotionClassifier — same __call__(text) ->
    VADScore-shaped protocol — so it plugs into
    IngestPipeline(emotion_classifier=...) with no pipeline changes.

    valence/dominance still come from the wrapped text classifier;
    arousal is overridden from the physio source's latest heart-rate
    reading via hr_to_arousal().
    """

    def __init__(
        self,
        text_classifier,              # callable(text) -> VADScore-shaped
        physio_source: SignalSource,
        baseline_bpm : float = 68.0,
        sensitivity  : float = 25.0,
    ):
        self.text_classifier = text_classifier
        self.physio_source   = physio_source
        self.baseline_bpm    = baseline_bpm
        self.sensitivity     = sensitivity

    def __call__(self, text: str) -> VADScore:
        base   = self.text_classifier(text)
        physio = self.physio_source.current_physio()
        arousal = hr_to_arousal(physio.hr_bpm, self.baseline_bpm, self.sensitivity)

        return VADScore(
            valence   = base.valence,
            arousal   = arousal,
            dominance = base.dominance,
            confidence= getattr(base, "confidence", 0.5),
            raw_scores= {
                "mode"        : "physio_fused",
                "text_valence": base.valence,
                "hr_bpm"      : physio.hr_bpm,
                "hr_arousal"  : arousal,
            },
        )

"""
signals.py
Hardware-agnostic contract for the "second brain" ambient input: a stream
of decoded utterances (from subvocal EMG, eventually) plus physiological
samples (heart rate, attention).

Pull-based rather than async/streaming — matches how ingestion is already
driven (one call per event) and avoids designing a real-time transport
layer against a device that doesn't exist yet. SimulatedSignalSource
(signals_sim.py) implements this now; a real EMG-backed source implements
the same two methods later with nothing downstream changing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class UtteranceEvent:
    text      : str
    timestamp : float
    confidence: float   # decoder confidence [0, 1]
    source    : str     # "emg" | "sim" | "chat"


@dataclass
class PhysioSample:
    timestamp : float
    hr_bpm    : Optional[float]   # heart rate, beats per minute
    attention : Optional[float]   # [0, 1] — engagement/focus proxy


class SignalSource(ABC):
    """
    Contract every BCI signal adapter implements — simulated or real.
    """

    @abstractmethod
    def next_utterance(self) -> Optional[UtteranceEvent]:
        """Return the next decoded utterance if one is ready, else None."""
        raise NotImplementedError

    @abstractmethod
    def current_physio(self) -> PhysioSample:
        """Return the latest heart-rate / attention reading."""
        raise NotImplementedError

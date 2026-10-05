"""
signals_sim.py
Simulated BCI signal source — stands in for the real subvocal-EMG +
heart-rate + attention rig while it's being built.

Scripted scenarios (in the spirit of arme_v2.py's `sessions` demo list)
span a deliberate spread of physiological/emotional states so the
Hebbian/ARM-E pipeline — including the new attention gate — gets
exercised across its whole range: deep focus, distraction, stress,
excitement, boredom, calm reflection.

next_utterance() pops one scripted line per call (None once exhausted).
current_physio() returns the active scenario's target HR/attention with
gaussian noise — no continuous interpolation, since each ingest already
consumes one atomic (text, hr, attention) sample rather than a waveform.
"""

import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# core/ modules import each other bare — put this file's directory on
# sys.path first so it works whether imported as `core.signals_sim` or
# `signals_sim` (see tutor_engine.py's docstring for the same pattern).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signals import PhysioSample, SignalSource, UtteranceEvent  # noqa: E402


@dataclass
class _Scenario:
    label           : str
    text            : str
    target_hr       : float   # bpm
    target_attention: float   # [0, 1]


BASELINE_HR = 68.0

DEFAULT_SCENARIOS: List[_Scenario] = [
    _Scenario(
        "flow state — coding",
        "The recursion bottoms out cleaner if I pass the accumulator "
        "instead of mutating state — worth refactoring the parser this way.",
        target_hr=74.0, target_attention=0.95,
    ),
    _Scenario(
        "distracted browsing",
        "Someone in the group chat linked a video about octopus camouflage, "
        "kind of neat I guess.",
        target_hr=70.0, target_attention=0.25,
    ),
    _Scenario(
        "stressed deadline",
        "There's no way I finish the deck before the 3pm review, I still "
        "haven't touched the budget slide.",
        target_hr=98.0, target_attention=0.55,
    ),
    _Scenario(
        "excited idea",
        "Wait — if the eligibility trace already tracks co-activation, I "
        "could reuse it for the attention gate instead of a new signal!",
        target_hr=92.0, target_attention=0.9,
    ),
    _Scenario(
        "bored, drifting",
        "This meeting could have been an email, I've read the same slide "
        "three times now.",
        target_hr=64.0, target_attention=0.15,
    ),
    _Scenario(
        "calm reflection",
        "Good day overall — the morning run cleared my head before the "
        "harder conversation with the team.",
        target_hr=62.0, target_attention=0.6,
    ),
    _Scenario(
        "anxious rumination",
        "I keep replaying what I said in standup, it probably came across "
        "worse than I meant it.",
        target_hr=88.0, target_attention=0.35,
    ),
    _Scenario(
        "deep focus — writing",
        "The second section needs to establish the constraint before the "
        "reader sees the workaround, otherwise the fix looks arbitrary.",
        target_hr=71.0, target_attention=0.92,
    ),
]


class SimulatedSignalSource(SignalSource):
    """
    Deterministic-ish simulated signal source. Seed for reproducibility.

    Usage:
        sim = SimulatedSignalSource(seed=7)
        while (utt := sim.next_utterance()) is not None:
            physio = sim.current_physio()
            engine.ingest_ambient(utt, physio, session_id)
    """

    def __init__(
        self,
        scenarios : Optional[List[_Scenario]] = None,
        hr_noise  : float = 3.0,
        att_noise : float = 0.06,
        seed      : Optional[int] = None,
    ):
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self.hr_noise  = hr_noise
        self.att_noise = att_noise
        self._rng      = random.Random(seed)
        self._idx      = -1   # index of the scenario `current_physio` should reflect

    def next_utterance(self) -> Optional[UtteranceEvent]:
        self._idx += 1
        if self._idx >= len(self.scenarios):
            return None
        s = self.scenarios[self._idx]
        return UtteranceEvent(
            text       = s.text,
            timestamp  = time.time(),
            confidence = self._rng.uniform(0.75, 0.98),   # simulated decoder confidence
            source     = "sim",
        )

    def current_physio(self) -> PhysioSample:
        if self._idx < 0 or self._idx >= len(self.scenarios):
            return PhysioSample(timestamp=time.time(), hr_bpm=BASELINE_HR, attention=0.5)
        s = self.scenarios[self._idx]
        hr  = max(40.0, s.target_hr + self._rng.gauss(0, self.hr_noise))
        att = min(1.0, max(0.0, s.target_attention + self._rng.gauss(0, self.att_noise)))
        return PhysioSample(timestamp=time.time(), hr_bpm=hr, attention=att)

    @property
    def current_label(self) -> str:
        if 0 <= self._idx < len(self.scenarios):
            return self.scenarios[self._idx].label
        return "—"

    def reset(self):
        self._idx = -1

"""
practice_scheduler.py — adaptive, outcome-driven practice scheduling.

Replaces the fixed 3-day staleness constant with per-concept-type
half-lives that adapt to observed recall outcomes:

    correct answer → that type's half-life ×= 1.25   (memory holds longer)
    wrong answer   → that type's half-life ×= 0.75   (forgetting faster
                                                      than we assumed)

clamped to [0.5, 30] days and persisted to disk so estimates survive
restarts. Recall probability follows the Ebbinghaus-style exponential:

    p_recall(Δt) = exp(−Δt · ln 2 / h_type)

This is a lightweight online variant of half-life regression
(Zaidi et al., AIED 2020; Settles & Meeder HLR) — no training corpus
needed; the tutor bootstraps from priors and personalizes as you practice.
"""
import json
import math
import os
import time

LN2 = math.log(2.0)

# Priors per concept type (days) — loosely ordered by consolidation depth.
DEFAULT_HALF_LIFE_DAYS = {
    "vocabulary": 2.0,
    "example":   2.0,
    "question":  2.0,
    "task":      1.0,
    "event":     1.5,
    "fact":      3.0,
    "idea":      4.0,
    "plan":      4.0,
    "procedure": 6.0,
    "feeling":   6.0,
    "entity":    7.0,
    "grammar":   10.0,
    "cultural":  14.0,
}
NEUTRAL_HALF_LIFE_DAYS = 3.0

GROWTH_ON_CORRECT = 1.25
SHRINK_ON_WRONG   = 0.75
MIN_HALF_LIFE_DAYS = 0.5
MAX_HALF_LIFE_DAYS = 30.0


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Plain-python cosine similarity for interference checks."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class PracticeScheduler:
    """
    Owns:
      - per-type adaptive half-lives (days), updated on graded answers
      - last-practiced timestamps per node (interleave/recently-seen guard)
      - recall-probability queries used to rank the practice queue

    State persists as JSON under state_dir so the model survives restarts.
    """

    def __init__(
        self,
        state_dir : str  = "./practice_state",
        relearn_gap_s : float = 10 * 60,     # don't re-serve an item within 10 min
    ):
        self.state_dir     = state_dir
        self.state_path    = os.path.join(state_dir, "scheduler.json")
        self.relearn_gap_s = relearn_gap_s

        self._halflife_days : dict[str, float] = {}
        self._last_practiced: dict[str, float] = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            self._halflife_days  = {k: float(v)
                                    for k, v in data.get("halflife_days", {}).items()}
            self._last_practiced = {k: float(v)
                                    for k, v in data.get("last_practiced", {}).items()}
        except (OSError, json.JSONDecodeError, ValueError):
            pass    # fresh start

    def save(self):
        os.makedirs(self.state_dir, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "halflife_days"  : self._halflife_days,
                "last_practiced" : self._last_practiced,
                "saved_at"       : time.time(),
            }, f)
        os.replace(tmp, self.state_path)

    # ── Queries ────────────────────────────────────────────────────────

    def halflife_s(self, concept_type: str) -> float:
        h_days = self._halflife_days.get(
            concept_type,
            DEFAULT_HALF_LIFE_DAYS.get(concept_type, NEUTRAL_HALF_LIFE_DAYS),
        )
        return max(MIN_HALF_LIFE_DAYS,
                   min(MAX_HALF_LIFE_DAYS, h_days)) * 24 * 3600

    def recall_prob(self, elapsed_s: float, concept_type: str) -> float:
        """Ebbinghaus exponential: p = 2^(−Δt/h). Low p → practice next."""
        return 2.0 ** (-max(elapsed_s, 0.0) / self.halflife_s(concept_type))

    def ready_again(self, node_id: str) -> bool:
        """False while an item was practiced too recently (spacing guard)."""
        last = self._last_practiced.get(node_id)
        return last is None or (time.time() - last) >= self.relearn_gap_s

    def halflife_table(self) -> dict[str, float]:
        """Current adaptive half-lives (days) — for dashboards/debugging."""
        merged = dict(DEFAULT_HALF_LIFE_DAYS)
        merged.update(self._halflife_days)
        return {k: round(v, 3) for k, v in merged.items()}

    # ── Updates ────────────────────────────────────────────────────────

    def record(self, node_ids: list[str], concept_types: list[str],
               correct: bool):
        """
        Update half-lives from a graded exercise and stamp practice times.
        All target types shift together — m_t was global for the exercise.
        """
        factor = GROWTH_ON_CORRECT if correct else SHRINK_ON_WRONG
        # dedupe: one exercise shifts each affected type once, however
        # many of its concepts were targeted
        for ctype in dict.fromkeys(concept_types):
            current = self.halflife_s(ctype) / (24 * 3600)
            updated = max(MIN_HALF_LIFE_DAYS,
                          min(MAX_HALF_LIFE_DAYS, current * factor))
            self._halflife_days[ctype] = round(updated, 4)

        now = time.time()
        for nid in node_ids:
            self._last_practiced[nid] = now

        # prune stale entries (>90 days idle)
        cutoff = now - 90 * 24 * 3600
        self._last_practiced = {
            k: v for k, v in self._last_practiced.items() if v >= cutoff
        }
        self.save()

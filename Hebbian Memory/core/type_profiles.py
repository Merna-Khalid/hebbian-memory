"""
type_profiles.py — per-concept-type memory dynamics.

Different kinds of knowledge decay at different rates: vocabulary items
fade fast without use (and are cheap to re-learn), grammar rules and
cultural frameworks are slow to fade once consolidated. Inspired by
per-memory, type-conditioned temporal decay (ScrubJay episodic-memory
principles, arXiv:2608.04746) rather than one global decay constant.

Two knobs per type:
    tau_stale_s   — staleness timescale for the fading/practice queue
                    (seconds since last activation before the item is
                    considered "stale"; smaller = practice sooner)
    lam_mult      — multiplier on Oja's λ for Hebbian edge decay during
                    ingest-time updates of edges leaving this type's nodes
                    (>1 = weights erode faster when not reinforced)

Unknown types fall back to NEUTRAL.
"""
import math

SECONDS_PER_DAY = 24 * 3600

TYPE_PROFILES: dict[str, dict[str, float]] = {
    # ── Japanese-tutor domain ──────────────────────────────────────────
    "vocabulary": dict(tau_stale_s=2.0 * SECONDS_PER_DAY,  lam_mult=1.5),
    "grammar":    dict(tau_stale_s=10.0 * SECONDS_PER_DAY, lam_mult=0.7),
    "example":    dict(tau_stale_s=2.0 * SECONDS_PER_DAY,  lam_mult=1.3),
    "question":   dict(tau_stale_s=2.0 * SECONDS_PER_DAY,  lam_mult=1.2),
    "fact":       dict(tau_stale_s=3.0 * SECONDS_PER_DAY,  lam_mult=1.0),
    "entity":     dict(tau_stale_s=7.0 * SECONDS_PER_DAY,  lam_mult=0.8),
    "procedure":  dict(tau_stale_s=7.0 * SECONDS_PER_DAY,  lam_mult=0.8),
    "cultural":   dict(tau_stale_s=14.0 * SECONDS_PER_DAY, lam_mult=0.6),
    # ── Second-brain domain ────────────────────────────────────────────
    "idea":       dict(tau_stale_s=4.0 * SECONDS_PER_DAY,  lam_mult=1.1),
    "task":       dict(tau_stale_s=1.0 * SECONDS_PER_DAY,  lam_mult=1.6),  # tasks rot fast
    "plan":       dict(tau_stale_s=5.0 * SECONDS_PER_DAY,  lam_mult=1.0),
    "feeling":    dict(tau_stale_s=6.0 * SECONDS_PER_DAY,  lam_mult=0.9),
    # ── Shared / fallback ──────────────────────────────────────────────
    "event":      dict(tau_stale_s=1.5 * SECONDS_PER_DAY,  lam_mult=1.5),
}

NEUTRAL = dict(tau_stale_s=3.0 * SECONDS_PER_DAY, lam_mult=1.0)


def profile_for(concept_type: str) -> dict[str, float]:
    """Decay profile for a concept type; unknown types get neutral values."""
    return TYPE_PROFILES.get(concept_type, NEUTRAL)


def staleness(elapsed_s: float, concept_type: str) -> float:
    """Exponential staleness in [0, 1) on the type's own timescale."""
    tau = profile_for(concept_type)["tau_stale_s"]
    return 1.0 - math.exp(-max(elapsed_s, 0.0) / tau)


def lam_for(concept_type: str, base_lam: float) -> float:
    """Oja decay rate scaled by the type's erosion multiplier."""
    return base_lam * profile_for(concept_type)["lam_mult"]

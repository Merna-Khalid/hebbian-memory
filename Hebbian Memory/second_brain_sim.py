"""
second_brain_sim.py
Simulated BCI "second brain" driver — CLI.

Runs the second_brain persona (core.tutor_engine's DOMAIN_PRESETS)
against SimulatedSignalSource (core/signals_sim.py) instead of a real
subvocal-EMG + heart-rate + attention rig, which is still being built.
Every scripted scenario becomes one ambient ingest via
TutorEngine.ingest_ambient() — same Hebbian/ARM-E pipeline the Japanese
tutor uses, no chat turn, gated by the new attention signal.

Usage:
    python second_brain_sim.py
    HEBBIAN_STUB=1 python second_brain_sim.py   # no model downloads

Config via env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
"""

import os

from core.tutor_engine import build_engine, build_stub_engine
from core.signals_sim import SimulatedSignalSource


def main():
    print("\nsecond brain (simulated) — starting...")

    sim = SimulatedSignalSource(seed=7)

    if os.environ.get("HEBBIAN_STUB") == "1":
        print("HEBBIAN_STUB=1 — stub models, rule-based text valence + HR-fused arousal")
        engine = build_stub_engine(domain="second_brain", signal_source=sim)
    else:
        engine = build_engine(domain="second_brain", signal_source=sim)

    session_id = engine.start_session(trigger="simulated_bci")

    print("\n" + "=" * 88)
    print("second brain — simulated ambient ingest")
    print("=" * 88)
    header = (f"{'scenario':<24} {'Q':<4} {'m_t':<7} {'gated':<7} "
              f"{'attn':<6} {'hr':<6} {'Δw':<9} {'concepts'}")
    print(header)
    print("-" * 88)

    try:
        while (utt := sim.next_utterance()) is not None:
            physio = sim.current_physio()
            result = engine.ingest_ambient(utt, physio, session_id)

            arme = result["arme"]
            if arme is not None:
                gated_str = "YES" if arme["gated"] else "no"
                print(f"{sim.current_label:<24} {arme['quadrant']:<4} "
                      f"{arme['m_t']:<7.3f} {gated_str:<7} "
                      f"{result['attention']:<6.2f} "
                      f"{physio.hr_bpm:<6.1f} "
                      f"{arme['delta_w_mean']:<+9.4f} "
                      f"{', '.join(c['label'] for c in result['concepts'])}")
            else:
                print(f"{sim.current_label:<24} (no concepts extracted)")
    finally:
        engine.end_session(session_id)
        engine.close()

    print("\nSession saved — the memory persists in Neo4j.")
    print("Inspect it the same way as the Japanese tutor's graph "
          "(GET /api/graph, or japanese_tutor.py's top-edges query) — "
          "concepts are tagged source_type='sensor'.")


if __name__ == "__main__":
    main()

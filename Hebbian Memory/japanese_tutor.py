"""
japanese_tutor.py
Hebbian RAG Japanese learning companion — CLI.

Thin shell over core.tutor_engine.TutorEngine, which owns the actual
chat turn (retrieve → respond → extract/ingest → co-retrieval
reinforcement). server.py wraps the same engine for the web UI.

    BGE-M3 embeddings (EN+JA aligned)
    LFM2.5-1.2B-JP for conversation and summaries
    vad-bert for emotion → ARM-E modulation
    HippocampalGNN + GraphStore for Hebbian memory

ARM-E ensures:
    Confusion (Q2/Q3) → suppressed Hebbian update → bad associations don't stick
    Curiosity / breakthrough (Q1) → amplified update → insights encode deeply

Usage:
    python japanese_tutor.py
    Type in English or Japanese. 'quit' to exit.

Config via env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
"""

from core.tutor_engine import build_engine


def main():
    print("\n日本語学習システム起動中...")

    engine     = build_engine()   # loads models + connects to Neo4j
    session_id = engine.start_session(trigger="manual")

    print("\n" + "="*60)
    print("日本語学習システム — Japanese Learning System")
    print("="*60)
    print("英語か日本語でどうぞ。Type 'quit' to exit, '/practice' to review fading concepts.")
    print("="*60 + "\n")

    def practice_round():
        """One retrieval-practice exercise from the most faded concepts."""
        exercise = engine.next_practice(session_id)
        if exercise is None:
            print("\n[practice] Nothing to practice yet — chat first so "
                  "concepts enter memory.\n")
            return

        print(f"\n{'─'*60}")
        print(f"[practice] {exercise['exercise_type']}  "
              f"(targets: {', '.join(exercise['targets'])})")
        print(f"\n{exercise['prompt']}")
        if exercise["hint"]:
            print(f"   hint: {exercise['hint']}")
        try:
            answer = input("\nYour answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not answer:
            return
        result = engine.submit_practice_answer(session_id, answer)
        if result is None:
            return
        mark = "○ 正解！" if result["correct"] else "× Not quite"
        print(f"\n{mark} {result['feedback']}")
        if not result["correct"] and result["answer"]:
            print(f"   expected: {result['answer']}")
        if result["reinforced"]:
            print("   [memory: edges reinforced — forgetting timer reset]")
        print(f"{'─'*60}\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n終了します。")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "終了"):
                print("またね！またいつでもどうぞ。")
                break
            if user_input.lower() in ("/practice", "practice"):
                practice_round()
                continue

            result = engine.chat(user_input, session_id)
            print(f"\nLFM: {result['response']}\n")

            # ── ARM-E state ───────────────────────────────────────────
            arme = result["arme"]
            if arme is not None:
                gated_str = "gated" if arme["gated"] else f"m_t={arme['m_t']:.2f}"
                print(f"  [memory: {arme['quadrant']} {gated_str} "
                      f"Δw={arme['delta_w_mean']:+.4f} "
                      f"V={arme['valence']:+.2f} A={arme['arousal']:.2f} "
                      f"D={arme['dominance']:.2f}]")

            # ── Top edges (Hebbian learning visible) ─────────────────
            top_edges = engine.store._run(
                "MATCH (a:Concept)-[r:ASSOCIATED_WITH {layer: 'hippocampal'}]"
                "->(b:Concept) "
                "RETURN a.label AS src, b.label AS dst, r.hebb_weight AS w "
                "ORDER BY r.hebb_weight DESC LIMIT 3"
            )
            if top_edges:
                print("  [top edges]")
                for e in top_edges:
                    print(f"    {e['src'][:25]:<25} →  {e['dst'][:25]:<25} "
                          f" w={e['w']:.4f}")
            print()
    finally:
        engine.close()

    print("Session saved — the memory persists in Neo4j.")
    print("Run again to continue.")


if __name__ == "__main__":
    main()

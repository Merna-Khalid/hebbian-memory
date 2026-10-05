package com.mobilerag.hebbian.signals

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Simulated BCI signal source — stands in for the real subvocal-EMG + heart-rate +
 * attention rig while it's being built. Port of `core/signals_sim.py`.
 *
 * Scripted scenarios span a deliberate spread of physiological/emotional states (deep
 * focus, distraction, stress, excitement, boredom, calm reflection) so the Hebbian/ARM-E
 * pipeline gets exercised across its whole range.
 *
 * [nextUtterance] pops one scripted line per call (null once exhausted).
 * [currentPhysio] returns the active scenario's target HR/attention with gaussian noise.
 */
class SimulatedSignalSource(
    private val scenarios: List<Scenario> = DEFAULT_SCENARIOS,
    private val hrNoise: Double = 3.0,
    private val attNoise: Double = 0.06,
    seed: Long? = null,
) : SignalSource {

    data class Scenario(
        val label: String,
        val text: String,
        val targetHr: Double,        // bpm
        val targetAttention: Double, // [0, 1]
    )

    private val rng: Random = seed?.let { Random(it) } ?: Random
    private var idx = -1 // index of the scenario currentPhysio() should reflect

    override fun nextUtterance(): UtteranceEvent? {
        idx += 1
        if (idx >= scenarios.size) return null
        val s = scenarios[idx]
        return UtteranceEvent(
            text = s.text,
            timestamp = nowSeconds(),
            confidence = rng.nextDouble(0.75, 0.98), // simulated decoder confidence
            source = "sim",
        )
    }

    override fun currentPhysio(): PhysioSample {
        if (idx < 0 || idx >= scenarios.size) {
            return PhysioSample(timestamp = nowSeconds(), hrBpm = BASELINE_HR, attention = 0.5)
        }
        val s = scenarios[idx]
        val hr = maxOf(40.0, s.targetHr + gaussian(hrNoise))
        val att = (s.targetAttention + gaussian(attNoise)).coerceIn(0.0, 1.0)
        return PhysioSample(timestamp = nowSeconds(), hrBpm = hr, attention = att)
    }

    val currentLabel: String
        get() = if (idx in scenarios.indices) scenarios[idx].label else "—"

    fun reset() {
        idx = -1
    }

    /** Standard normal scaled by [sigma] — Box–Muller on the seeded [rng]. */
    private fun gaussian(sigma: Double): Double {
        val u1 = rng.nextDouble().let { if (it <= 0.0) Double.MIN_VALUE else it }
        val u2 = rng.nextDouble()
        return sigma * sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }

    private fun nowSeconds(): Double = System.currentTimeMillis() / 1000.0

    companion object {
        const val BASELINE_HR = 68.0

        val DEFAULT_SCENARIOS: List<Scenario> = listOf(
            Scenario(
                "flow state — coding",
                "The recursion bottoms out cleaner if I pass the accumulator " +
                    "instead of mutating state — worth refactoring the parser this way.",
                targetHr = 74.0, targetAttention = 0.95,
            ),
            Scenario(
                "distracted browsing",
                "Someone in the group chat linked a video about octopus camouflage, " +
                    "kind of neat I guess.",
                targetHr = 70.0, targetAttention = 0.25,
            ),
            Scenario(
                "stressed deadline",
                "There's no way I finish the deck before the 3pm review, I still " +
                    "haven't touched the budget slide.",
                targetHr = 98.0, targetAttention = 0.55,
            ),
            Scenario(
                "excited idea",
                "Wait — if the eligibility trace already tracks co-activation, I " +
                    "could reuse it for the attention gate instead of a new signal!",
                targetHr = 92.0, targetAttention = 0.9,
            ),
            Scenario(
                "bored, drifting",
                "This meeting could have been an email, I've read the same slide " +
                    "three times now.",
                targetHr = 64.0, targetAttention = 0.15,
            ),
            Scenario(
                "calm reflection",
                "Good day overall — the morning run cleared my head before the " +
                    "harder conversation with the team.",
                targetHr = 62.0, targetAttention = 0.6,
            ),
            Scenario(
                "anxious rumination",
                "I keep replaying what I said in standup, it probably came across " +
                    "worse than I meant it.",
                targetHr = 88.0, targetAttention = 0.35,
            ),
            Scenario(
                "deep focus — writing",
                "The second section needs to establish the constraint before the " +
                    "reader sees the workaround, otherwise the fix looks arbitrary.",
                targetHr = 71.0, targetAttention = 0.92,
            ),
        )
    }
}

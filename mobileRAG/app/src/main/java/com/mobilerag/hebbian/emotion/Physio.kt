package com.mobilerag.hebbian.emotion

import com.mobilerag.hebbian.signals.SignalSource
import kotlin.math.tanh

/**
 * Heart-rate → arousal mapping, and a physiology-fused emotion classifier for the
 * second-brain ambient path. Port of `core/physio.py`.
 *
 * Text sentiment is a reasonable valence proxy but a weak arousal proxy — arousal is
 * fundamentally physiological (Kensinger 2004). A BCI's heart-rate channel is a much
 * more direct arousal signal than word choice, so [PhysioFusedEmotionClassifier] keeps
 * valence/dominance from the wrapped text classifier but overrides arousal from a live
 * heart-rate reading.
 */

/**
 * Normalized deviation of heart rate from a resting baseline, mapped to [0, 1] via tanh
 * so large deviations saturate instead of clipping hard. `hrBpm == baselineBpm` → 0.5
 * (matches [VADScore]'s neutral default); null `hrBpm` → 0.5 (neutral, no signal).
 *
 * [sensitivity]: bpm deviation that maps to ~0.88 arousal (tanh(1)).
 */
fun hrToArousal(
    hrBpm: Double?,
    baselineBpm: Double = 68.0,
    sensitivity: Double = 25.0,
): Double {
    if (hrBpm == null) return 0.5
    val deviation = (hrBpm - baselineBpm) / sensitivity
    return (0.5 + 0.5 * tanh(deviation)).coerceIn(0.0, 1.0)
}

/**
 * Drop-in [EmotionClassifier] — valence/dominance come from the wrapped text classifier;
 * arousal is overridden from the physio source's latest heart-rate reading via
 * [hrToArousal].
 */
class PhysioFusedEmotionClassifier(
    private val textClassifier: EmotionClassifier,
    private val physioSource: SignalSource,
    private val baselineBpm: Double = 68.0,
    private val sensitivity: Double = 25.0,
) : EmotionClassifier {

    override fun predict(text: String): VADScore {
        val base = textClassifier.predict(text)
        val physio = physioSource.currentPhysio()
        val arousal = hrToArousal(physio.hrBpm, baselineBpm, sensitivity)

        return VADScore(
            valence = base.valence,
            arousal = arousal,
            dominance = base.dominance,
            confidence = base.confidence,
            rawScores = mapOf(
                "mode" to "physio_fused",
                "text_valence" to base.valence,
                "hr_bpm" to (physio.hrBpm ?: Double.NaN),
                "hr_arousal" to arousal,
            ),
        )
    }
}

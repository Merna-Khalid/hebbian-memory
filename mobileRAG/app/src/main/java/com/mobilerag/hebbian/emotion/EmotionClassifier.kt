package com.mobilerag.hebbian.emotion

import kotlin.math.abs

/**
 * Continuous VAD emotion classification contract, ported from `core/emotion_classifier.py`.
 *
 *     valence   : [-1, +1]  negative ← → positive
 *     arousal   : [ 0,  1]  calm ← → excited
 *     dominance : [ 0,  1]  controlled ← → in-control
 *
 * [StubEmotionClassifier] is the keyword heuristic used until a real model ships; an
 * ONNX-backed classifier (fine-tuned XLM-RoBERTa VAD regressor, per
 * `core/emotion_classifier.py`) can be added later as another [EmotionClassifier]
 * implementation with no call-site changes.
 */

/** VAD regression output. [confidence] is distance from neutral; [rawScores] is debug info. */
data class VADScore(
    val valence: Double,     // [-1, +1]
    val arousal: Double,     // [0, 1]
    val dominance: Double,   // [0, 1]
    val confidence: Double,  // [0, 1]
    val rawScores: Map<String, Any> = emptyMap(),
) {
    companion object {
        /** Neutral reading — also the answer for empty/blank text (emotion_classifier.py:218). */
        val NEUTRAL = VADScore(
            valence = 0.0, arousal = 0.5, dominance = 0.5,
            confidence = 0.0, rawScores = mapOf("mode" to "empty"),
        )
    }
}

interface EmotionClassifier {
    fun predict(text: String): VADScore
}

/**
 * Keyword-list stub — ports `ARME._stub_classifier` (core/arme_v2.py:336-349) into the
 * [EmotionClassifier] shape, plus the low-confidence shrink rule from
 * `core/emotion_classifier.py:206-209`.
 *
 * Each keyword group contributes at most once (the Python `break` semantics): valence
 * accumulates across groups, arousal takes the running max. Dominance has no keyword
 * cues in the Python stub, so it stays at the neutral 0.5.
 */
class StubEmotionClassifier(
    private val minConfidence: Double = 0.15,
) : EmotionClassifier {

    override fun predict(text: String): VADScore {
        if (text.isBlank()) return VADScore.NEUTRAL
        val t = text.lowercase()

        var valence = 0.0
        var arousal = 0.3
        val dominance = 0.5

        for ((words, dv, a) in KEYWORD_CUES) {
            if (words.any { it in t }) {
                valence += dv
                arousal = maxOf(arousal, a)
            }
        }
        valence = valence.coerceIn(-1.0, 1.0)
        arousal = arousal.coerceIn(0.0, 1.0)

        // Mean absolute deviation from neutral (0, 0.5, 0.5) — same formula as
        // emotion_classifier.py's fine-tuned confidence (line 178-182).
        var confidence = (abs(valence) + abs(arousal - 0.5) * 2 + abs(dominance - 0.5) * 2) / 3.0

        var v = valence
        var a = arousal
        var d = dominance
        if (confidence < minConfidence) {
            // Low confidence → shrink VAD 70% toward neutral (emotion_classifier.py:206-209).
            v *= 0.3
            a = 0.5 + (a - 0.5) * 0.3
            d = 0.5 + (d - 0.5) * 0.3
        }

        return VADScore(
            valence = v, arousal = a, dominance = d, confidence = confidence,
            rawScores = mapOf("mode" to "stub", "v" to v, "a" to a, "d" to d),
        )
    }

    private companion object {
        /** (keywords, valence delta, arousal floor) — one entry per loop in the Python stub. */
        val KEYWORD_CUES: List<Triple<List<String>, Double, Double>> = listOf(
            Triple(listOf("excellent", "amazing", "love", "brilliant", "fascinating", "eureka"), 0.8, 0.85),
            Triple(listOf("good", "interesting", "helpful", "nice", "clear", "thanks"), 0.4, 0.5),
            Triple(listOf("terrible", "hate", "useless", "broken", "wrong", "frustrated", "stuck"), -0.8, 0.75),
            Triple(listOf("unclear", "not sure", "hmm", "weird", "doesn't work", "confused"), -0.3, 0.45),
            Triple(listOf("why", "how", "what if", "curious", "wonder", "explain", "?"), 0.2, 0.7),
        )
    }
}

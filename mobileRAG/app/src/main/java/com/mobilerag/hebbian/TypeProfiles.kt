package com.mobilerag.hebbian

import kotlin.math.exp
import kotlin.math.max

/**
 * Per-concept-type memory dynamics (port of core/type_profiles.py).
 *
 * tauStaleS — staleness timescale for the fading/practice queue
 * lamMult   — multiplier on Oja's lambda for Hebbian edge decay
 * Unknown types fall back to [NEUTRAL].
 */
data class TypeProfile(val tauStaleS: Double, val lamMult: Double)

object TypeProfiles {

    const val SECONDS_PER_DAY: Double = 24.0 * 3600.0

    val TYPE_PROFILES: Map<String, TypeProfile> = mapOf(
        // Japanese-tutor domain
        "vocabulary" to TypeProfile(2.0 * SECONDS_PER_DAY, 1.5),
        "grammar" to TypeProfile(10.0 * SECONDS_PER_DAY, 0.7),
        "example" to TypeProfile(2.0 * SECONDS_PER_DAY, 1.3),
        "question" to TypeProfile(2.0 * SECONDS_PER_DAY, 1.2),
        "fact" to TypeProfile(3.0 * SECONDS_PER_DAY, 1.0),
        "entity" to TypeProfile(7.0 * SECONDS_PER_DAY, 0.8),
        "procedure" to TypeProfile(7.0 * SECONDS_PER_DAY, 0.8),
        "cultural" to TypeProfile(14.0 * SECONDS_PER_DAY, 0.6),
        // Second-brain domain
        "idea" to TypeProfile(4.0 * SECONDS_PER_DAY, 1.1),
        "task" to TypeProfile(1.0 * SECONDS_PER_DAY, 1.6),
        "plan" to TypeProfile(5.0 * SECONDS_PER_DAY, 1.0),
        "feeling" to TypeProfile(6.0 * SECONDS_PER_DAY, 0.9),
        // Shared / fallback
        "event" to TypeProfile(1.5 * SECONDS_PER_DAY, 1.5),
    )

    val NEUTRAL = TypeProfile(3.0 * SECONDS_PER_DAY, 1.0)

    /** Decay profile for a concept type; unknown types get neutral values. */
    fun profileFor(conceptType: String): TypeProfile = TYPE_PROFILES[conceptType] ?: NEUTRAL

    /** Exponential staleness in [0, 1) on the type's own timescale. */
    fun staleness(elapsedS: Double, conceptType: String): Double {
        val tau = profileFor(conceptType).tauStaleS
        return 1.0 - exp(-max(elapsedS, 0.0) / tau)
    }

    /** Oja decay rate scaled by the type's erosion multiplier. */
    fun lamFor(conceptType: String, baseLam: Double): Double =
        baseLam * profileFor(conceptType).lamMult
}

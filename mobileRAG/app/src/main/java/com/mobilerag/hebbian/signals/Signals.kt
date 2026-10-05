package com.mobilerag.hebbian.signals

/**
 * Hardware-agnostic contract for the "second brain" ambient input: a stream of decoded
 * utterances (from subvocal EMG, eventually) plus physiological samples (heart rate,
 * attention). Pull-based rather than streaming — [SimulatedSignalSource] implements this
 * now; a real EMG-backed source implements the same two methods later with nothing
 * downstream changing. Port of `core/signals.py`.
 */

/** A decoded utterance. [timestamp] is epoch seconds; [confidence] is decoder confidence in [0, 1]. */
data class UtteranceEvent(
    val text: String,
    val timestamp: Double,
    val confidence: Double,
    val source: String, // "emg" | "sim" | "chat"
)

/** Latest physiological reading. Null fields mean "no signal", not zero. */
data class PhysioSample(
    val timestamp: Double,
    val hrBpm: Double?,      // heart rate, beats per minute
    val attention: Double?,  // [0, 1] — engagement/focus proxy
)

/** Contract every BCI signal adapter implements — simulated or real. */
interface SignalSource {
    /** Return the next decoded utterance if one is ready, else null. */
    fun nextUtterance(): UtteranceEvent?

    /** Return the latest heart-rate / attention reading. */
    fun currentPhysio(): PhysioSample
}

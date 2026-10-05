package com.mobilerag.spikes

import android.content.Context

/** A Phase 0 spike: a self-contained experiment with a pass/fail verdict and measured numbers. */
interface Spike {
    val id: String
    val title: String
    val description: String

    /** Runs the spike, reporting progress via [log]. Returns a result with verdict and metrics. */
    suspend fun run(context: Context, log: (String) -> Unit): SpikeResult
}

data class SpikeResult(
    val passed: Boolean,
    val summary: String,
    val metrics: Map<String, String> = emptyMap(),
    val error: String? = null,
)

package com.mobilerag.hebbian

import android.util.Log
import com.mobilerag.hebbian.cortical.CorticalConsolidator
import com.mobilerag.hebbian.store.HebbianStore

/**
 * When cortical consolidation runs — same public surface and cadence as
 * `core/consolidation_scheduler.py`: every [everyN] ingests, at session end, on demand.
 *
 * The work itself is [CorticalConsolidator] (one per memory space, shared by every engine
 * on that space), which trains the cortical GAT and writes consolidated embeddings back.
 * Unlike Python — which runs synchronously at session end — every run here is launched in
 * the background: on-device a run takes on the order of a minute and must never hold up
 * ingest or the UI.
 *
 * Without a [cortex] (the model-less stub engine and the spikes) this stays a no-op that
 * logs what it would have done.
 */
class ConsolidationScheduler(
    private val store: HebbianStore,
    private val cortex: CorticalConsolidator? = null,
    private val everyN: Int = CorticalConsolidator.EVERY_N,
) {
    /** True when no consolidator is attached (stub engine / spikes). */
    val isStub: Boolean get() = cortex == null

    private var localCount: Int = 0

    val ingestCount: Int get() = cortex?.ingestCount ?: localCount

    /** Epoch seconds of the last completed consolidation (0.0 if never). */
    val lastConsolidationAt: Double get() = (cortex?.lastReport?.at ?: 0L) / 1000.0

    data class Stats(
        val ingestCount: Int,
        val lastConsolidationAt: Double,
        val everyN: Int,
        val nextAt: Int,
        val isStub: Boolean,
    )

    val stats: Stats
        get() = Stats(
            ingestCount = ingestCount,
            lastConsolidationAt = lastConsolidationAt,
            everyN = everyN,
            nextAt = ingestCount + (everyN - ingestCount % everyN),
            isStub = isStub,
        )

    /** Call after every successful ingest; consolidates every [everyN]-th. */
    fun onIngest() {
        if (cortex != null) {
            cortex.onIngest(store, everyN)
        } else {
            localCount += 1
            if (localCount % everyN == 0) logSkipped("scheduled ($localCount ingests)")
        }
    }

    /** Call at session end; Python always consolidates here. */
    fun onSessionEnd() {
        cortex?.launch(store, "session_end") ?: logSkipped("session_end")
    }

    /** Manual trigger. */
    fun consolidateNow() {
        cortex?.launch(store, "manual") ?: logSkipped("manual")
    }

    private fun logSkipped(reason: String) {
        Log.i(TAG, "consolidation not run ($reason) — no cortical consolidator on this engine")
    }

    private companion object {
        const val TAG = "ConsolidationScheduler"
    }
}

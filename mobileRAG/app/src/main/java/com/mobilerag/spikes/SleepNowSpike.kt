package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.hebbian.cortical.CorticalConsolidator
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.profile.SpaceManager
import java.io.File

/**
 * What the nightly SleepWorker does, on demand: two real cortical consolidations of the
 * ACTIVE space's store (the second warm-starts from the saved weights and z_prev). This
 * rewrites that space's concept embeddings and hippocampal causal scores, exactly as sleep
 * would. Checks that the write-back reaches the store and leaves updatedAt alone.
 * (A forced WorkManager job doesn't help here: periodic work run before it's due is skipped.)
 */
class SleepNowSpike : Spike {
    override val id = "sleep_now"
    override val title = "Sleep now (consolidate active space)"
    override val description = "Runs the nightly consolidation twice on the ACTIVE space (rewrites its embeddings + causal scores, as sleep does): write-back reaches the store, updatedAt untouched, warm start from saved state."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult = try {
        val spaceDir = SpaceManager.activeDir(context)
        val store = HebbianStoreFactory.create(context, spaceDir)
        val cortex = CorticalConsolidator.forSpace(spaceDir)
        log("space: ${spaceDir.name}, store: ${store.id}")

        val before = store.listConcepts(10_000).associateBy { it.nodeId }
        log("concepts: ${before.size}")
        val checks = linkedMapOf<String, Boolean>()
        val metrics = linkedMapOf<String, String>()

        val first = cortex.consolidate(store, "diagnostic #1")
        if (first == null) {
            SpikeResult(
                passed = false,
                summary = "Nothing consolidated: graph below ${CorticalConsolidator.MIN_NODES} nodes / " +
                    "${CorticalConsolidator.MIN_EDGES} edges at w ≥ ${CorticalConsolidator.MIN_HEBB_WEIGHT}, or a run was in progress",
            )
        } else {
            log("run 1: ${first.nodes} nodes, ${first.edges} edges, loss %.4f, %d ms".format(first.finalLoss, first.elapsedMs))
            val after = store.listConcepts(10_000).associateBy { it.nodeId }
            val moved = after.values.count { a ->
                val b = before[a.nodeId] ?: return@count false
                b.embedding.size != a.embedding.size || !b.embedding.contentEquals(a.embedding)
            }
            val stamped = after.values.count { a -> before[a.nodeId]?.updatedAt != a.updatedAt }
            val edges = store.getGlobalGraph(minWeight = CorticalConsolidator.MIN_HEBB_WEIGHT).edges
            val withCausal = edges.count { it.causalScore != 0.0 }
            log("embeddings rewritten: $moved, updatedAt changed: $stamped, edges with causal ≠ 0: $withCausal/${edges.size}")
            checks["embeddings written back ($moved ≥ ${first.nodes})"] = moved >= first.nodes
            checks["updatedAt untouched"] = stamped == 0
            checks["causal scores on trained edges"] = withCausal > 0
            val state = File(spaceDir, "cortical_state")
            checks["state saved"] = listOf("weights.bin", "z_prev.bin", "node_order.json").all { File(state, it).isFile }

            val second = cortex.consolidate(store, "diagnostic #2")
            if (second != null) {
                log("run 2 (warm): loss %.4f, %d ms".format(second.finalLoss, second.elapsedMs))
                metrics["run 2"] = "loss %.4f, %d ms".format(second.finalLoss, second.elapsedMs)
            }
            checks["warm-start run completes"] = second != null && second.finalLoss.isFinite()
            metrics["run 1"] = "${first.nodes} nodes, ${first.edges} edges, loss %.4f, %d ms".format(first.finalLoss, first.elapsedMs)

            checks.forEach { (k, v) -> log("${if (v) "PASS" else "FAIL"} $k") }
            val failed = checks.filterValues { !it }.keys
            SpikeResult(
                passed = failed.isEmpty(),
                summary = if (failed.isEmpty()) "Consolidated ${first.nodes} concepts twice; write-back verified"
                else "Failed: ${failed.joinToString()}",
                metrics = metrics,
            )
        }
    } catch (t: Throwable) {
        SpikeResult(
            passed = false,
            summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
            error = t.stackTraceToString().take(2000),
        )
    }
}

package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.hebbian.store.ConceptMerge
import com.mobilerag.hebbian.store.ConceptNode
import com.mobilerag.hebbian.store.HebbianEdge
import com.mobilerag.hebbian.store.HebbianStore
import com.mobilerag.hebbian.store.LadybugHebbianStore
import com.mobilerag.hebbian.store.SqliteHebbianStore
import java.io.File

/**
 * Duplicate merge + backup on both real stores (LadybugDB and SQLite), each in a scratch
 * file under cacheDir — no memory space is touched. Same fixture as the JVM
 * IdentityEndToEndTest.mergeFoldsOldDuplicatesTogether, which only covers the in-memory
 * fake: 食べる / 「食べる」 merge (the more-activated one survives), their edges to 飲む are
 * re-pointed with w = max and co-activations summed, the edge between the two duplicates
 * becomes a self loop and is dropped. The merge must survive close/reopen, and the backup
 * taken before it must still open with the pre-merge graph.
 */
class ConceptMergeSpike : Spike {
    override val id = "concept_merge"
    override val title = "Duplicate merge on both stores"
    override val description = "Merge + backup on scratch LadybugDB and SQLite stores (no space touched): edges re-pointed (w = max, co-activations summed), self loop dropped, survives reopen, backup opens with the pre-merge graph."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val root = File(context.cacheDir, "concept_merge_spike").apply { deleteRecursively(); mkdirs() }
        val checks = linkedMapOf<String, Boolean>()
        return try {
            val kinds = listOf<Pair<String, (String) -> HebbianStore>>(
                "ladybug" to { _ -> LadybugHebbianStore() },
                "sqlite" to { p -> SqliteHebbianStore(context, p) },
            )
            for ((kind, make) in kinds) {
                val path = File(root, "$kind/hebbian.${if (kind == "ladybug") "lbug" else "db"}").absolutePath
                File(path).parentFile!!.mkdirs()
                var store = make(path).also { it.open(path); it.setupSchema() }
                try {
                    seed(store)
                    val plan = ConceptMerge.plan(store.listConcepts(100))
                    log("$kind: plan remap=${plan.remap}")
                    checks["$kind: plan a→b"] = plan.remap == mapOf("a" to "b")

                    val backup = File(root, "$kind-backup")
                    store.backupTo(backup)
                    store.applyMerge(plan)
                    store.close()

                    store = make(path).also { it.open(path); it.setupSchema() }
                    checks["$kind: merged state survives reopen"] = verifyMerged(kind, store, log)
                    store.close()

                    val backupPath = File(backup, File(path).name).absolutePath
                    val old = make(backupPath).also { it.open(backupPath); it.setupSchema() }
                    try {
                        val n = old.listConcepts(100).size
                        val e = old.getGlobalGraph(minWeight = 0.0).edges.size
                        log("$kind backup: $n concepts, $e edges")
                        checks["$kind: backup has pre-merge graph (3 concepts, 4 edges)"] = n == 3 && e == 4
                    } finally {
                        old.close()
                    }
                } catch (t: Throwable) {
                    try { store.close() } catch (_: Throwable) {}
                    throw t
                }
            }
            val failed = checks.filterValues { !it }.keys
            checks.forEach { (k, v) -> log("${if (v) "PASS" else "FAIL"} $k") }
            SpikeResult(
                passed = failed.isEmpty(),
                summary = if (failed.isEmpty()) "Merge + backup verified on LadybugDB and SQLite (${checks.size}/${checks.size})"
                else "Failed: ${failed.joinToString()}",
                metrics = mapOf("checks" to "${checks.size - failed.size}/${checks.size}"),
            )
        } catch (t: Throwable) {
            checks.forEach { (k, v) -> log("${if (v) "PASS" else "FAIL"} $k") }
            SpikeResult(
                passed = false,
                summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                error = t.stackTraceToString().take(2000),
            )
        } finally {
            root.deleteRecursively()
        }
    }

    private suspend fun seed(store: HebbianStore) {
        fun node(id: String, label: String) =
            ConceptNode(id, label, label, label, "vocabulary", "user", null, 0, 0, 0, floatArrayOf(1f, 0f, 0f))
        store.upsertConcept(node("a", "食べる"))
        store.upsertConcept(node("b", "「食べる」"))
        store.upsertConcept(node("c", "飲む"))
        repeat(1) { store.incrementActivation("a") }
        repeat(3) { store.incrementActivation("b") }
        repeat(1) { store.incrementActivation("c") }
        fun edge(s: String, d: String, w: Double, co: Int) = HebbianEdge(s, d, w, 0.0, 0.0, co, 0, "hippocampal")
        store.upsertEdge(edge("a", "c", 2.0, 2))
        store.upsertEdge(edge("b", "c", 1.0, 3))
        store.upsertEdge(edge("c", "a", 1.5, 1))
        store.upsertEdge(edge("a", "b", 1.0, 1))
    }

    private suspend fun verifyMerged(kind: String, store: HebbianStore, log: (String) -> Unit): Boolean {
        val concepts = store.listConcepts(100).associateBy { it.nodeId }
        val edges = store.getGlobalGraph(minWeight = 0.0).edges.associateBy { it.srcId to it.dstId }
        log("$kind merged: concepts=${concepts.mapValues { it.value.activationCount }} " +
            "edges=${edges.values.map { "${it.srcId}→${it.dstId} w=${it.hebbWeight} co=${it.coActivationCount}" }}")
        val bc = edges["b" to "c"]
        val cb = edges["c" to "b"]
        return concepts.keys == setOf("b", "c") &&
            concepts.getValue("b").activationCount == 4 &&
            edges.size == 2 &&
            bc != null && bc.hebbWeight == 2.0 && bc.coActivationCount == 5 &&
            cb != null && cb.hebbWeight == 1.5
    }
}

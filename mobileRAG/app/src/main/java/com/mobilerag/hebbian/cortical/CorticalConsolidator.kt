package com.mobilerag.hebbian.cortical

import android.util.Log
import com.mobilerag.hebbian.store.CausalScoreUpdate
import com.mobilerag.hebbian.store.HebbianStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.Random
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Cortical consolidation for one memory space — port of
 * `core/consolidation_scheduler.py` (_consolidate_sync + save/load):
 *
 *   hippocampal graph (edges ≥ [MIN_HEBB_WEIGHT], top [EDGE_LIMIT]) → normalized embeddings
 *   → z_prev aligned by node id → CorticalGnn.consolidate (30 epochs) → embeddings written
 *   back (the field vector search reads) → causal = cos(z_i, z_j) on the trained edges →
 *   weights, z_prev and node order persisted under `<space>/cortical_state/`.
 *
 * Two deliberate differences from the Python original, both applied there too: the
 * write-back leaves updatedAt alone (it's "last seen" to the practice queue), and causal
 * scores go to the hippocampal edges consolidation trained on (Python targeted 'cortical'
 * edges, which nothing creates, so causal was always 0). One phone-only difference: runs
 * are capped at [MAX_NODES] nodes (see CorticalPrep.prepare).
 *
 * One instance per space ([forSpace]): the tutor and second-brain engines share a store and
 * must not train it twice or overwrite each other's state. Runs on a single low-priority
 * background thread and never overlap (skip-if-running, as in Python); they never block ingest.
 */
class CorticalConsolidator private constructor(private val stateDir: File) {

    data class Report(
        val reason: String,
        val nodes: Int,
        val edges: Int,
        val droppedNodes: Int,
        val finalLoss: Double,
        val elapsedMs: Long,
        val at: Long,
    )

    @Volatile
    var lastReport: Report? = null
        private set

    /** Ingests since the state was created — persisted, drives the every-N trigger. */
    @Volatile
    var ingestCount: Int = 0
        private set

    private val running = AtomicBoolean(false)

    /** Bumped by [clear]; a run that started under an older generation discards its result. */
    @Volatile
    private var generation = 0

    init {
        loadMeta()
    }

    val isRunning: Boolean get() = running.get()

    /** Counts one ingest; every [everyN]-th launches a background run. */
    fun onIngest(store: HebbianStore, everyN: Int) {
        ingestCount++
        saveMeta()
        if (ingestCount % everyN == 0) launch(store, "scheduled ($ingestCount ingests)")
    }

    /** Fire-and-forget run on the consolidation thread. */
    fun launch(store: HebbianStore, reason: String) {
        scope.launch {
            runCatching { consolidate(store, reason) }
                .onFailure { Log.w(TAG, "consolidation ($reason) failed", it) }
        }
    }

    /** Runs one consolidation now (suspending); null when skipped. */
    suspend fun consolidate(store: HebbianStore, reason: String): Report? {
        if (!running.compareAndSet(false, true)) {
            Log.i(TAG, "already running — skipping ($reason)")
            return null
        }
        try {
            return withContext(dispatcher) { run(store, reason, generation) }
        } finally {
            running.set(false)
        }
    }

    private suspend fun run(store: HebbianStore, reason: String, gen: Int): Report? {
        val t0 = System.currentTimeMillis()
        val sg = store.getGlobalGraph(layer = "hippocampal", minWeight = MIN_HEBB_WEIGHT, limit = EDGE_LIMIT)
        val prep = CorticalPrep.prepare(sg, MAX_NODES)
        if (prep == null || prep.ids.size < MIN_NODES || prep.src.size < MIN_EDGES) {
            Log.i(TAG, "skipped ($reason) — graph too small (${prep?.ids?.size ?: 0} nodes, ${prep?.src?.size ?: 0} edges)")
            return null
        }
        if (prep.droppedNodes > 0) Log.i(TAG, "left out ${prep.droppedNodes} nodes (embedding size / node cap)")

        val config = CorticalConfig.production(prep.dim)
        val params = loadParams(config) ?: CorticalParams.glorot(config, Random())
        val zPrev = loadZPrev()?.let { (ids, z) -> CorticalPrep.alignZPrev(ids, z, prep.ids, prep.x) }
        val rng = Random()
        val (z, history) = CorticalGnn(params).consolidate(
            prep.x, prep.src, prep.dst, prep.w, zPrev, RejectionNegativeSampler(rng), rng,
        )
        if (gen != generation) {
            Log.i(TAG, "memory cleared during consolidation — discarding result")
            return null
        }

        // Write back: the cortical embeddings become what vector search compares against.
        val d = z.cols
        store.updateConceptEmbeddings(
            prep.ids.withIndex().associate { (i, id) -> id to FloatArray(d) { k -> z.data[i * d + k].toFloat() } },
        )
        store.updateCausalScores(
            prep.src.indices.map { e ->
                CausalScoreUpdate(prep.ids[prep.src[e]], prep.ids[prep.dst[e]], CorticalPrep.rowCosine(z, prep.src[e], prep.dst[e]))
            },
            layer = "hippocampal",
        )
        saveParams(params)
        saveZPrev(prep.ids, z)

        val report = Report(
            reason = reason, nodes = prep.ids.size, edges = prep.src.size, droppedNodes = prep.droppedNodes,
            finalLoss = history.last().total, elapsedMs = System.currentTimeMillis() - t0,
            at = System.currentTimeMillis(),
        )
        lastReport = report
        saveMeta()
        Log.i(TAG, "consolidated ($reason): ${report.nodes} nodes, ${report.edges} edges, " +
            "loss ${"%.4f".format(report.finalLoss)}, ${report.elapsedMs} ms")
        return report
    }

    /** Clear Hebbian memory: forget all cortical state; an in-flight run discards its result. */
    fun clear() {
        generation++
        stateDir.deleteRecursively()
        ingestCount = 0
        lastReport = null
    }

    // ── Persistence (atomic tmp + rename) ─────────────────────────────

    private fun writeAtomically(name: String, write: (File) -> Unit) {
        stateDir.mkdirs()
        val tmp = File(stateDir, "$name.tmp")
        write(tmp)
        val target = File(stateDir, name)
        if (!tmp.renameTo(target)) {
            target.delete()
            tmp.renameTo(target)
        }
    }

    private fun saveParams(params: CorticalParams) = writeAtomically(WEIGHTS) { f ->
        DataOutputStream(f.outputStream().buffered()).use { params.writeTo(it) }
    }

    private fun loadParams(config: CorticalConfig): CorticalParams? = runCatching {
        File(stateDir, WEIGHTS).takeIf { it.exists() }?.let { f ->
            DataInputStream(f.inputStream().buffered()).use { CorticalParams.readFrom(it, config) }
        }
    }.getOrNull()

    private fun saveZPrev(ids: List<String>, z: Mat) {
        writeAtomically(Z_PREV) { f ->
            DataOutputStream(f.outputStream().buffered()).use { out ->
                out.writeInt(z.rows)
                out.writeInt(z.cols)
                for (v in z.data) out.writeFloat(v.toFloat())
            }
        }
        writeAtomically(NODE_ORDER) { it.writeText(JSONArray(ids).toString()) }
    }

    private fun loadZPrev(): Pair<List<String>, Mat>? = runCatching {
        val zf = File(stateDir, Z_PREV)
        val of = File(stateDir, NODE_ORDER)
        if (!zf.exists() || !of.exists()) return null
        val ids = JSONArray(of.readText()).let { a -> List(a.length()) { a.getString(it) } }
        val z = DataInputStream(zf.inputStream().buffered()).use { input ->
            val rows = input.readInt()
            val cols = input.readInt()
            Mat(rows, cols, DoubleArray(rows * cols) { input.readFloat().toDouble() })
        }
        ids to z
    }.getOrNull()

    private fun saveMeta() = runCatching {
        writeAtomically(META) { f ->
            val json = JSONObject().put("ingest_count", ingestCount)
            lastReport?.let { r ->
                json.put("last", JSONObject()
                    .put("reason", r.reason).put("nodes", r.nodes).put("edges", r.edges)
                    .put("dropped_nodes", r.droppedNodes).put("final_loss", r.finalLoss)
                    .put("elapsed_ms", r.elapsedMs).put("at", r.at))
            }
            f.writeText(json.toString())
        }
    }.onFailure { Log.w(TAG, "meta save failed", it) }

    private fun loadMeta() {
        runCatching {
            val json = JSONObject(File(stateDir, META).readText())
            ingestCount = json.optInt("ingest_count", 0)
            json.optJSONObject("last")?.let { r ->
                lastReport = Report(
                    reason = r.getString("reason"), nodes = r.getInt("nodes"), edges = r.getInt("edges"),
                    droppedNodes = r.optInt("dropped_nodes", 0), finalLoss = r.getDouble("final_loss"),
                    elapsedMs = r.getLong("elapsed_ms"), at = r.getLong("at"),
                )
            }
        }
    }

    companion object {
        private const val TAG = "CorticalConsolidator"

        // consolidation_scheduler.py / tutor_engine.py parity
        const val EVERY_N = 20
        const val MIN_NODES = 5
        const val MIN_EDGES = 3
        const val MIN_HEBB_WEIGHT = 0.5
        const val EDGE_LIMIT = 5000

        /** Phone-only cap (≈ 1 min per run on-device at ~1000 nodes; see CorticalBenchmarkTest). */
        const val MAX_NODES = 1500

        private const val WEIGHTS = "weights.bin"
        private const val Z_PREV = "z_prev.bin"
        private const val NODE_ORDER = "node_order.json"
        private const val META = "meta.json"

        private val dispatcher = Executors.newSingleThreadExecutor { r ->
            Thread(r, "cortical-consolidation").apply { priority = Thread.MIN_PRIORITY; isDaemon = true }
        }.asCoroutineDispatcher()
        private val scope = CoroutineScope(SupervisorJob() + dispatcher)

        private val bySpace = HashMap<String, CorticalConsolidator>()

        /** The consolidator for a memory space (`SpaceManager.activeDir`). */
        fun forSpace(spaceDir: File): CorticalConsolidator = synchronized(bySpace) {
            bySpace.getOrPut(spaceDir.absolutePath) { CorticalConsolidator(File(spaceDir, "cortical_state")) }
        }
    }
}

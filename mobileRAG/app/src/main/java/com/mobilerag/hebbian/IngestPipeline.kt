package com.mobilerag.hebbian

import android.util.Log
import com.mobilerag.hebbian.emotion.EmotionClassifier
import com.mobilerag.hebbian.store.ConceptNode
import com.mobilerag.hebbian.store.HebbWeightUpdate
import com.mobilerag.hebbian.store.HebbianEdge
import com.mobilerag.hebbian.store.HebbianStore
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Full Hebbian RAG ingest pipeline (port of `core/ingest_pipeline.py`).
 *
 * Wires together: embedFn → EmotionClassifier (VAD) → [Arme] → [HippocampalGnn] →
 * [HebbianStore]. One call to [ingest] takes a text chunk and:
 *   1. embeds it and resolves its identity ([ConceptIdentity]: reuse an existing concept
 *   or create one), 2. summarizes it (new nodes), 3. classifies VAD emotion from the user
 *   text, 4. upserts the Concept node (new) or re-activates it (reused), 5. bootstraps
 *   similarity edges (new, or reused without edges), 6. pulls the local
 *   subgraph, 7. computes the ARM-E modulation m_t, 8. propagates through the
 *   hippocampal GNN and applies the ARM-E-modulated Hebbian update (eligibility
 *   gating + anti-Hebbian decay), 9. writes weights + eligibility back,
 *   10. updates ARM-E stats and SessionStats.
 *
 * Not thread-safe (ARM-E EMA state + session counters), same as the Python original.
 */
const val UNVALIDATED_BOOTSTRAP_W = 0.6

/** One [IngestPipeline.ingest] result — mirrors the Python `IngestResult` dataclass. */
data class IngestResult(
    val nodeId: String,
    val label: String,
    val mT: Double,
    val gated: Boolean,
    val quadrant: String,
    val valence: Double,
    val arousal: Double,
    val dominance: Double,
    val meanWBefore: Double,
    val meanWAfter: Double,
    val deltaWMean: Double,
    val nEdges: Int,
    val subgraphSize: Int,
    val elapsedMs: Double,
    /** True when this ingest re-activated an existing concept instead of creating one. */
    val reused: Boolean = false,
)

class IngestPipeline(
    private val store: HebbianStore,
    private val embedFn: suspend (String) -> FloatArray,
    private val summaryFn: (suspend (String) -> String)? = null,
    private val emotionClassifier: EmotionClassifier,
    private val consolidation: ConsolidationScheduler? = null,
    private val featureDim: Int = 768,
    private val eta: Double = 0.05,
    private val lam: Double = 0.008,
    private val wFloor: Double = 0.1,
    private val wCeil: Double = 5.0,
    private val subgraphHops: Int = 2,
    private val subgraphMax: Int = 100,
    private val minHebbWeight: Double = 0.3,
    private val bootstrapK: Int = 5,
    private val bootstrapThreshold: Double = 0.3,
) {
    private val arme = Arme()

    // Session state (Python: _session_id / _session_activations / _node_activation_counts)
    private var currentSessionId: String? = null
    private var sessionActivations: Int = 0
    private val nodeActivationCounts = HashMap<String, Int>()

    // ── Session management ────────────────────────────────────────────

    /** Start a new session; returns its id. Resets counters and the ARM-E EMA. */
    suspend fun startSession(trigger: String = "time_window"): String {
        currentSessionId = store.startSession(trigger)
        sessionActivations = 0
        nodeActivationCounts.clear()
        arme.resetEma()
        Log.d(TAG, "Session started: $currentSessionId (trigger=$trigger)")
        return currentSessionId!!
    }

    /** Close the session; notifies the consolidation scheduler (Python: final consolidation). */
    suspend fun endSession(sessionId: String) {
        store.endSession(sessionId)
        consolidation?.onSessionEnd()
        currentSessionId = null
        Log.d(TAG, "Session closed: $sessionId ($sessionActivations activations)")
    }

    // ── Main ingest entry point ───────────────────────────────────────

    suspend fun ingest(
        text: String,
        label: String,
        conceptType: String = "fact",
        sourceType: String = "document",
        sourceUri: String? = null,
        userText: String? = null,
        sessionId: String? = null,
        nodeId: String? = null,
        attention: Double? = null,
        validated: Boolean = true,
    ): IngestResult {
        val t0 = System.currentTimeMillis()
        val sid = sessionId ?: currentSessionId
        val attn = attention ?: 1.0
        val userTextOrEmpty = userText ?: ""

        // ── 1. Embed ──────────────────────────────────────────────────
        val embedding = embedFn(text)

        // ── 1b. Identity: a concept the graph already holds? ──────────
        // One vector search serves identity and bootstrap. It runs before the node is
        // written, so for a new node it can't contain the node itself (bootstrap below takes
        // the top bootstrapK others — the same set as before). An explicit nodeId skips
        // resolution: the caller has already decided identity.
        val similar = store.vectorSearch(embedding, bootstrapK + 1)
        val existing = if (nodeId != null) null else ConceptIdentity.resolve(
            label = label,
            conceptType = conceptType,
            validated = validated,
            labelMatches = store.findByLabelKey(ConceptIdentity.labelKey(label)),
            vectorHits = similar,
        )
        val reused = existing != null
        val nid = existing?.nodeId ?: nodeId ?: HebbianStore.makeNodeId()

        // ── 3. Classify emotion ───────────────────────────────────────
        val vad = if (userTextOrEmpty.isNotBlank()) emotionClassifier.predict(userTextOrEmpty) else null
        val valence = vad?.valence ?: 0.0
        val arousal = vad?.arousal ?: 0.5
        val dominance = vad?.dominance ?: 0.5

        // ── 2 + 4. Summary + upsert (new nodes only) ──────────────────
        // A reused node keeps its stored embedding — possibly a consolidated cortical one —
        // and its first description; the summary is an LLM call, so skipping it also makes
        // repeats cheaper.
        if (!reused) {
            val summary = if (summaryFn != null) {
                summaryFn.invoke(text)
            } else {
                // Truncate to first 2 sentences as fallback
                val sentences = text.replace("?", ".").replace("!", ".").split(".")
                sentences.take(2).map { it.trim() }.filter { it.isNotEmpty() }.joinToString(". ") + "."
            }
            val now = System.currentTimeMillis()
            store.upsertConcept(
                ConceptNode(
                    nodeId = nid,
                    label = label,
                    textRaw = text,
                    textSummary = summary,
                    conceptType = conceptType,
                    sourceType = sourceType,
                    sourceUri = sourceUri,
                    createdAt = now,
                    updatedAt = now,
                    activationCount = 0,
                    embedding = embedding,
                )
            )
        }
        store.incrementActivation(nid)

        // Track session activation counts for u_t
        sessionActivations += 1
        nodeActivationCounts[nid] = (nodeActivationCounts[nid] ?: 0) + 1

        // ── 4b. Bootstrap edges to similar existing nodes ─────────────
        // A brand-new node has no edges, so the GNN never fires: connect it to similar
        // existing nodes with a starting weight. Bidirectional — Hebbian learning is
        // symmetric at initialization. Unvalidated content starts weaker and must earn
        // strength. A reused node is bootstrapped only if it has no usable edges.
        val bootstrapW = if (validated) 1.0 else UNVALIDATED_BOOTSTRAP_W
        suspend fun bootstrap() {
            for ((neighbor, score) in similar.filter { it.first.nodeId != nid }.take(bootstrapK)) {
                if (score < bootstrapThreshold) continue        // too dissimilar
                for ((srcId, dstId) in listOf(nid to neighbor.nodeId, neighbor.nodeId to nid)) {
                    store.upsertEdge(
                        HebbianEdge(
                            srcId = srcId,
                            dstId = dstId,
                            hebbWeight = bootstrapW,
                            eligibility = 0.0,
                            causalScore = 0.0,
                            coActivationCount = 0,
                            lastUpdated = System.currentTimeMillis(),
                            layer = "hippocampal",
                        )
                    )
                }
            }
        }

        // ── 5. Pull local subgraph ────────────────────────────────────
        suspend fun localSubgraph() = store.getLocalSubgraph(
            seedIds = listOf(nid),
            hops = subgraphHops,
            maxNodes = subgraphMax,
            layer = "hippocampal",
            minWeight = minHebbWeight,
        )
        if (!reused) bootstrap()
        var subgraph = localSubgraph()
        if (reused && subgraph.edges.isEmpty()) {
            bootstrap()
            subgraph = localSubgraph()
        }

        // If subgraph is empty or only the seed node, nothing to update
        if (subgraph.nodes.size < 2 || subgraph.edges.isEmpty()) {
            // Still compute ARM-E for stats, but skip GNN
            val (mT, armeState) = arme.step(
                queryEmb = embedding,
                retrievedEmbs = arrayOf(FloatArray(embedding.size)),
                userText = userTextOrEmpty,
                nodeId = nid,
                nodeActivations = nodeActivationCounts[nid] ?: 1,
                totalActivations = max(sessionActivations, 1),
                valenceOverride = valence,
                arousalOverride = arousal,
                dominanceOverride = dominance,
                attention = attn,
            )
            updateStats(sid, nid, armeState, deltaWMean = 0.0)
            consolidation?.onIngest()
            return IngestResult(
                nodeId = nid, label = label, mT = mT,
                gated = armeState.gated, quadrant = armeState.emotion.quadrant,
                valence = valence, arousal = arousal, dominance = dominance,
                meanWBefore = 0.0, meanWAfter = 0.0, deltaWMean = 0.0,
                nEdges = 0, subgraphSize = 1,
                elapsedMs = (System.currentTimeMillis() - t0).toDouble(),
                reused = reused,
            )
        }

        // ── 6. Build local tensors from subgraph ──────────────────────
        // Map global nodeIds to local indices for the GNN.
        val idToIdx = HashMap<String, Int>(subgraph.nodes.size)
        subgraph.nodes.forEachIndexed { i, n -> idToIdx[n.nodeId] = i }

        // Embeddings normalized to the unit sphere before Hebbian (F.normalize).
        val h = Array(subgraph.nodes.size) { l2Normalize(subgraph.nodes[it].embedding) }

        // Filter to edges whose endpoints are both in the capped node set; built
        // ONCE so edge positions stay aligned (weights, eligibility, write-back).
        val validEdges = subgraph.edges.filter { it.srcId in idToIdx && it.dstId in idToIdx }
        val srcIdx = IntArray(validEdges.size) { idToIdx.getValue(validEdges[it].srcId) }
        val dstIdx = IntArray(validEdges.size) { idToIdx.getValue(validEdges[it].dstId) }
        val weights = DoubleArray(validEdges.size) { validEdges[it].hebbWeight }

        if (srcIdx.isEmpty()) {
            consolidation?.onIngest()
            return IngestResult(
                nodeId = nid, label = label, mT = 1.0,
                gated = true, quadrant = "Q4",
                valence = valence, arousal = arousal, dominance = dominance,
                meanWBefore = 0.0, meanWAfter = 0.0, deltaWMean = 0.0,
                nEdges = 0, subgraphSize = subgraph.nodes.size,
                elapsedMs = (System.currentTimeMillis() - t0).toDouble(),
                reused = reused,
            )
        }

        val meanWBefore = weights.average()

        // ── 7. ARM-E modulation ───────────────────────────────────────
        // Retrieved embeddings for r_t: up to 10 non-seed nodes (normalized rows).
        val retrievedEmbs = subgraph.nodes
            .asSequence()
            .filter { it.nodeId != nid }
            .take(10)
            .map { h[idToIdx.getValue(it.nodeId)] }
            .toList()
            .toTypedArray()

        val (rawMT, armeState) = arme.step(
            queryEmb = embedding,
            retrievedEmbs = retrievedEmbs,
            userText = userTextOrEmpty,
            nodeId = nid,
            nodeActivations = nodeActivationCounts[nid] ?: 1,
            totalActivations = max(sessionActivations, 1),
            valenceOverride = valence,
            arousalOverride = arousal,
            dominanceOverride = dominance,
            attention = attn,
        )

        // Validation gating (Kairos-style): unverified content never gets an
        // amplifying m_t — it can encode at baseline or below. (armeState keeps
        // the uncapped value for stats, as in Python.)
        val mT = if (!validated) min(rawMT, 1.0) else rawMT

        // ── 8. Propagate, then modulated Hebbian update ───────────────
        // Message passing first — co-activation is computed on POST-propagation
        // features, so the GNN actually shapes the weight update.
        val hProp = HippocampalGnn.propagate(h, srcIdx, dstIdx, weights)

        // Eligibility traces are persisted per-edge in the store and passed in,
        // so gating uses the same state we write back afterwards.
        val eligIn = DoubleArray(validEdges.size) { validEdges[it].eligibility }

        // Type-conditioned Oja decay: each edge's λ scales by its source
        // concept's decay profile (see TypeProfiles).
        val nidToType = subgraph.nodes.associate { it.nodeId to it.conceptType }
        val lamEdges = DoubleArray(validEdges.size) {
            TypeProfiles.lamFor(nidToType[validEdges[it].srcId] ?: "fact", lam)
        }

        val hebb = arme.modulatedHebbianUpdate(
            edgeSrc = srcIdx,
            edgeDst = dstIdx,
            edgeWeight = weights,
            hPost = hProp,
            mT = mT,
            eta = eta,
            lamOja = lamEdges,
            wFloor = wFloor,
            wCeil = wCeil,
            eligibility = eligIn,
        )
        val meanWAfter = hebb.stats.meanW
        val deltaWMean = meanWAfter - meanWBefore

        // ── 9. Write updated weights + eligibility back ───────────────
        val weightUpdates = validEdges.mapIndexed { k, e ->
            HebbWeightUpdate(
                srcId = e.srcId,
                dstId = e.dstId,
                weight = hebb.edgeWeight[k],
                eligibility = hebb.eligibility[k],
                layer = "hippocampal",
            )
        }
        if (weightUpdates.isNotEmpty()) store.updateHebbWeights(weightUpdates)

        // ── 10. Update stats ──────────────────────────────────────────
        updateStats(sid, nid, armeState, deltaWMean)

        // Notify scheduler — triggers consolidation every N ingests
        consolidation?.onIngest()

        return IngestResult(
            nodeId = nid,
            label = label,
            mT = mT,
            gated = armeState.gated,
            quadrant = armeState.emotion.quadrant,
            valence = valence,
            arousal = arousal,
            dominance = dominance,
            meanWBefore = meanWBefore,
            meanWAfter = meanWAfter,
            deltaWMean = deltaWMean,
            nEdges = weightUpdates.size,
            subgraphSize = subgraph.nodes.size,
            elapsedMs = (System.currentTimeMillis() - t0).toDouble(),
            reused = reused,
        )
    }

    // ── Helpers ───────────────────────────────────────────────────────

    /** Write ARM-E stats to the node and SessionStats (Python: _update_stats). */
    private suspend fun updateStats(
        sessionId: String?,
        nodeId: String,
        armeState: ARMEState,
        deltaWMean: Double,
    ) {
        store.updateConceptArmeStats(
            nodeId = nodeId,
            mT = armeState.mT,
            dominance = armeState.dominance,
            rT = armeState.rT,
        )
        if (sessionId != null) {
            store.upsertSessionStats(
                sessionId = sessionId,
                nodeId = nodeId,
                mT = armeState.mT,
                dominance = armeState.dominance,
                rT = armeState.rT,
                deltaW = deltaWMean,
            )
            store.incrementSessionActivations(sessionId)
        }
    }

    private companion object {
        const val TAG = "IngestPipeline"

        /** torch F.normalize(dim=1): v / max(||v||₂, 1e-12). */
        fun l2Normalize(v: FloatArray): FloatArray {
            var sum = 0.0
            for (x in v) sum += x.toDouble() * x
            val denom = max(sqrt(sum), 1e-12)
            return FloatArray(v.size) { (v[it] / denom).toFloat() }
        }
    }
}

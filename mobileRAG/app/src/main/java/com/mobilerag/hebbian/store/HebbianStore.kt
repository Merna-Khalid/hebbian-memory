package com.mobilerag.hebbian.store

import com.mobilerag.hebbian.TypeProfiles
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.round
import kotlin.math.sqrt

/**
 * Hebbian graph store — port of `Hebbian Memory/core/graph_store.py` (the single
 * Neo4j interface of the Python system) onto Android. Implementations:
 * [SqliteHebbianStore] (schema-of-record) and [LadybugHebbianStore] (LadybugDB Cypher).
 *
 * Graph model (unchanged from Python):
 *   (:Concept)                        — knowledge nodes, String UUID ids (uuid4)
 *   (:Concept)-[:ASSOCIATED_WITH]->   — Hebbian edges, one per (src, dst, layer)
 *   (:Session)                        — time-windowed or trigger-based sessions
 *   (:Session)-[:AGGREGATES]->(:SessionStats)-[:TRACKED_IN]->(:Concept)
 *
 * Differences from Python: timestamps are epoch millis as Long (Python used
 * time.time() float seconds; they are converted back to seconds only when calling
 * [staleness] and when writing the `ts` field of recent_events entries). The pure
 * algorithms (cosine search, BFS subgraph expansion, spreading activation, Welford
 * stats, reinforcement and scoring rules) live as internal kernels at the bottom of
 * this file so both backends return identical results.
 */

// ── Data classes ──────────────────────────────────────────────────────

/** A Concept node. All fields map 1:1 to stored properties. */
data class ConceptNode(
    val nodeId: String,
    val label: String,
    val textRaw: String,
    val textSummary: String,
    val conceptType: String,        // "fact"|"entity"|"procedure"|"event"|"question"
    val sourceType: String,         // "document"|"web"|"sensor"|"user"|"synthesized"
    val sourceUri: String?,
    val createdAt: Long,            // epoch millis
    val updatedAt: Long,            // epoch millis
    val activationCount: Int,
    val embedding: FloatArray,
    val meanMT: Double = 1.0,
    val meanDominance: Double = 0.5,
    val meanRT: Double = 0.5,
    /** ARM-E event circular buffer (max [HebbianStore.RECENT_EVENTS_MAX]); each entry is one event as a compact JSON object string. */
    val recentEvents: List<String> = emptyList(),
)

/** An ASSOCIATED_WITH relationship, keyed by (srcId, dstId, layer). */
data class HebbianEdge(
    val srcId: String,
    val dstId: String,
    val hebbWeight: Double,
    val eligibility: Double,
    val causalScore: Double,
    val coActivationCount: Int,
    val lastUpdated: Long,          // epoch millis
    val layer: String,              // "hippocampal" | "cortical"
)

/** Local neighborhood returned for GNN processing. */
data class Subgraph(
    val nodes: List<ConceptNode>,
    val edges: List<HebbianEdge>,
)

/** One row of [HebbianStore.updateHebbWeights]: set weight/eligibility on an existing edge. */
data class HebbWeightUpdate(
    val srcId: String,
    val dstId: String,
    val weight: Double,
    val eligibility: Double,
    val layer: String,
)

/** One row of [HebbianStore.updateCausalScores]: set causal_score on an existing cortical edge. */
data class CausalScoreUpdate(
    val srcId: String,
    val dstId: String,
    val score: Double,
)

/** One [HebbianStore.retrieveForLlm] hit: the node plus every score component. */
data class LlmCandidate(
    val node: ConceptNode,
    val score: Double,
    val cosSim: Double,
    val assoc: Double,              // raw avg Hebbian weight from seeds (before normalization)
    val spread: Double,             // normalized spreading activation in [0, 1]
    val causal: Double,
)

/** One [HebbianStore.getFadingConcepts] row — a practice-queue entry. */
data class FadingConcept(
    val concept: ConceptNode,
    val avgW: Double,
    val degree: Int,
    val strength: Double,           // clamp(avgW / wCeil, 0, 1)
    val staleness: Double,          // type-conditioned, from TypeProfiles
    val fadingScore: Double,        // (1 − strength) · staleness · ln(1 + activations)
)

/** One [HebbianStore.getDominanceRegressionData] row (Phase 4 w_d regression input). */
data class DominanceRegressionRow(
    val sessionId: String,
    val conceptType: String,
    val mT: Double,
    val dominance: Double,
    val deltaW: Double,
    val activationCount: Int,
)

// ── Store interface ───────────────────────────────────────────────────

interface HebbianStore {

    val id: String

    suspend fun open(path: String)
    suspend fun close()

    /** Create tables/constraints. Safe to re-run. */
    suspend fun setupSchema()

    // ── Concept CRUD ──────────────────────────────────────────────────

    /**
     * Create or update a Concept node; returns nodeId. On create, createdAt/updatedAt
     * are set to now, activationCount to 0 and ARM-E stats to their defaults (caller
     * values for those fields are ignored, as in Python). On match, only label, texts,
     * types, sourceUri, updatedAt and embedding are written.
     */
    suspend fun upsertConcept(node: ConceptNode): String

    suspend fun getConcept(nodeId: String): ConceptNode?

    /** Increment activationCount and touch updatedAt. Called every time a node is retrieved. */
    suspend fun incrementActivation(nodeId: String)

    /** Concepts whose normalized label ([com.mobilerag.hebbian.ConceptIdentity.labelKey])
     *  equals [labelKey], any type — the identity lookup ingest runs before creating a node. */
    suspend fun findByLabelKey(labelKey: String): List<ConceptNode>

    /**
     * Batch-replace concept embeddings after cortical consolidation. Deliberately leaves
     * updatedAt alone: updatedAt means "last seen" to the practice queue, and a
     * consolidation run is not an exposure (the Python original reset it — fixed in both).
     */
    suspend fun updateConceptEmbeddings(embeddings: Map<String, FloatArray>)

    // ── Vector search ─────────────────────────────────────────────────

    /**
     * Brute-force cosine over an in-memory embedding cache (rag/VectorStore.kt pattern),
     * replacing the Neo4j vector index. With [conceptType] set, over-fetches topK×3 and
     * post-filters by type, matching the Python fallback strategy.
     */
    suspend fun vectorSearch(
        embedding: FloatArray,
        topK: Int,
        conceptType: String? = null,
    ): List<Pair<ConceptNode, Double>>

    // ── Hebbian edge CRUD ─────────────────────────────────────────────

    /**
     * Create the ASSOCIATED_WITH edge for (srcId, dstId, layer) at the given values, or — if
     * it exists — re-assert it without ever weakening it: hebbWeight = max(existing, new),
     * coActivationCount + 1; eligibility and causalScore are kept. lastUpdated is set to now.
     * (It used to overwrite everything: once concepts are reused, re-extracting a relation
     * would knock a trained 4.8 edge back to 1.5 and zero its causal score.)
     */
    suspend fun upsertEdge(edge: HebbianEdge)

    /** Batch set hebb_weight/eligibility on existing edges after a GNN step; bumps coActivationCount and lastUpdated. */
    suspend fun updateHebbWeights(updates: List<HebbWeightUpdate>)

    /**
     * Batch set causal_score on existing edges of [layer] after consolidation; lastUpdated is
     * left alone (not a use of the edge). Consolidation writes to the hippocampal edges it
     * trained on — the Python original matched only 'cortical' edges, which nothing ever
     * creates, so causal was always 0 in both versions.
     */
    suspend fun updateCausalScores(updates: List<CausalScoreUpdate>, layer: String = "hippocampal")

    /**
     * Co-retrieval reinforcement — the temporal Hebbian signal. For every ordered pair
     * (a, b), a ≠ b: create the edge at w = 1.0 if missing, else
     * w = clamp(w + mT·eta − lam·w, wFloor, wCeil); coActivationCount + 1 either way.
     */
    suspend fun reinforceCoRetrieved(
        nodeIds: List<String>,
        mT: Double,
        eta: Double = 0.02,
        lam: Double = 0.008,
        wFloor: Double = 0.1,
        wCeil: Double = 5.0,
        layer: String = "hippocampal",
    )

    // ── Subgraph retrieval for GNN ────────────────────────────────────

    /**
     * Local neighborhood for hippocampal GNN processing. Iterative BFS expansion in
     * Kotlin (replaces apoc.path.subgraphAll): hop by hop over directed edges matching
     * [layer] with hebbWeight ≥ [minWeight]; keeps seeds + endpoints of surviving
     * edges, capped at [maxNodes] by activationCount DESC; edges to evicted nodes are
     * dropped. Matches the semantics of the Python no-APOC fallback query.
     */
    suspend fun getLocalSubgraph(
        seedIds: List<String>,
        hops: Int = 2,
        maxNodes: Int = 100,
        layer: String = "hippocampal",
        minWeight: Double = 0.3,
    ): Subgraph

    /** Large subgraph for cortical GNN consolidation: top-[limit] edges by weight above [minWeight], plus their endpoint nodes. */
    suspend fun getGlobalGraph(
        layer: String = "hippocampal",
        minWeight: Double = 0.5,
        limit: Int = 5000,
    ): Subgraph

    /** Most recently updated concepts, regardless of edges (dashboard view of isolated nodes). */
    suspend fun listConcepts(limit: Int = 200): List<ConceptNode>

    // ── ARM-E stats updates ───────────────────────────────────────────

    /**
     * Welford online mean update of meanMT/meanDominance/meanRT with
     * n = max(activationCount, 1), and appends one event to the recentEvents circular
     * buffer. Called after every ARM-E step for a node.
     */
    suspend fun updateConceptArmeStats(nodeId: String, mT: Double, dominance: Double, rT: Double)

    // ── Session management ────────────────────────────────────────────

    /** Create a new Session; returns its String UUID session id. */
    suspend fun startSession(trigger: String = "time_window"): String

    /** Close a session by writing endedAt. */
    suspend fun endSession(sessionId: String)

    suspend fun incrementSessionActivations(sessionId: String, count: Int = 1)

    /**
     * Welford online-mean upsert of the per-(session, concept) SessionStats row.
     * [deltaW] is the mean Hebbian weight change this step — key for the w_d regression.
     * In LadybugDB this also maintains the AGGREGATES/TRACKED_IN relations.
     */
    suspend fun upsertSessionStats(
        sessionId: String,
        nodeId: String,
        mT: Double,
        dominance: Double,
        rT: Double,
        deltaW: Double,
    )

    // ── LLM retrieval ─────────────────────────────────────────────────

    /**
     * Combined retrieval for LLM context assembly.
     *
     * score = α·cos + β·clamp(assoc/wCeil + δ_spread·spread, 0, 1.5) + γ·causal
     *
     * assoc — avg Hebbian edge weight from [seedNodeIds] to the candidate (1-hop, any
     * layer, as in Python); spread — multi-hop [spreadingActivation] from the same
     * seeds, restricted to the candidate pool. Candidates are vectorSearch(topK×3) plus up
     * to topK nodes reached strongly by spreading activation that vector search missed
     * (withSpreadCandidates). cos is on Neo4j's (1 + cos) / 2 scale.
     */
    suspend fun retrieveForLlm(
        queryEmbedding: FloatArray,
        seedNodeIds: List<String>,
        topK: Int = 5,
        alpha: Double = 0.6,
        beta: Double = 0.3,
        gamma: Double = 0.1,
        wCeil: Double = 5.0,
        useSpreading: Boolean = true,
        deltaSpread: Double = 0.3,
    ): List<LlmCandidate>

    // ── Practice: fading concepts ─────────────────────────────────────

    /**
     * Concepts whose memory is fading — the practice queue. Candidates (activationCount
     * ≥ [minActivations], stalest first, over-fetched ×[candidateMult]) are scored
     * fadingScore = (1 − clamp(avgW/wCeil, 0, 1)) · staleness · ln(1 + activationCount),
     * where avgW/degree cover hippocampal edges in both directions and staleness is
     * type-conditioned (com.mobilerag.hebbian.TypeProfiles). Sorted by fadingScore DESC.
     */
    suspend fun getFadingConcepts(
        limit: Int = 5,
        minActivations: Int = 1,
        wCeil: Double = 5.0,
        candidateMult: Int = 4,
    ): List<FadingConcept>

    /**
     * Multi-hop activation diffusion through the Hebbian graph. Every path (≤ [depth]
     * hops, clamped to 1..3) over [layer]/[minWeight]-filtered edges contributes
     * decay^hops · min(pathWeight, [W_CEIL]); totals are normalized to [0, 1] by the
     * peak. Seeds are excluded. Path enumeration is capped at [maxPaths].
     */
    suspend fun spreadingActivation(
        seedNodeIds: List<String>,
        layer: String = "hippocampal",
        depth: Int = 2,
        decay: Double = 0.45,
        minWeight: Double = 0.3,
        maxPaths: Int = 2000,
    ): Map<String, Double>

    // ── Phase 4: w_d regression data ─────────────────────────────────

    /** SessionStats rows (joined with Session and Concept) for tuning w_dom. */
    suspend fun getDominanceRegressionData(minActivations: Int = 5): List<DominanceRegressionRow>

    /** Wipes all concepts, edges, sessions and session stats. */
    suspend fun clear()

    // ── Maintenance ───────────────────────────────────────────────────

    /**
     * Consistent copy of this store's database into [dir] (created if missing), without
     * handing callers a closed store: SQLite uses VACUUM INTO; LadybugDB closes, copies and
     * reopens the same instance while holding its connection lock.
     */
    suspend fun backupTo(dir: File)

    /**
     * Applies a duplicate-merge [plan] ([ConceptMerge] rules): survivors take the group's
     * combined history, edges and session stats are re-pointed and combined, dropped
     * concepts are deleted. Take a [backupTo] first — this is not reversible.
     */
    suspend fun applyMerge(plan: MergePlan)

    companion object {
        const val RECENT_EVENTS_MAX = 20   // circular buffer size (matches Python)
        const val W_CEIL = 5.0             // Hebbian weight ceiling (matches Python)

        /** String UUID node id (Python uuid4). */
        fun makeNodeId(): String = UUID.randomUUID().toString()
    }
}

// ── Shared pure kernels ───────────────────────────────────────────────
// Both backends delegate to these so SQLite and LadybugDB return identical results.

internal fun round4(x: Double): Double = round(x * 1e4) / 1e4

/** Min normalized spread (peak = 1) for a node outside the vector pool to become a
 *  retrieval candidate — matches graph_store.py SPREAD_CANDIDATE_MIN. */
internal const val SPREAD_CANDIDATE_MIN = 0.3
internal fun round1(x: Double): Double = round(x * 1e1) / 1e1

/** Serializes the recent-events circular buffer as a JSON array of objects (Python-compatible storage format). */
internal fun encodeRecentEvents(events: List<String>): String {
    val arr = JSONArray()
    for (e in events) {
        try {
            arr.put(JSONObject(e))
        } catch (_: Exception) {
            arr.put(e) // tolerate malformed entries rather than dropping the buffer
        }
    }
    return arr.toString()
}

/** Inverse of [encodeRecentEvents]; each event is returned as its compact JSON object string. */
internal fun decodeRecentEvents(raw: String?): List<String> {
    if (raw.isNullOrEmpty()) return emptyList()
    return try {
        val arr = JSONArray(raw)
        List(arr.length()) { arr.get(it).toString() }
    } catch (_: Exception) {
        emptyList()
    }
}

/** Result of one ARM-E stats update (Welford means + new circular buffer). */
internal data class ArmeStats(
    val meanMT: Double,
    val meanDominance: Double,
    val meanRT: Double,
    val recentEvents: List<String>,
)

/** Welford online mean with n = max(activationCount, 1) + circular-buffer append (graph_store.py:605-663). */
internal fun armeStatsUpdate(
    meanMT: Double,
    meanDominance: Double,
    meanRT: Double,
    activationCount: Int,
    recentEvents: List<String>,
    mT: Double,
    dominance: Double,
    rT: Double,
    nowMs: Long,
): ArmeStats {
    val n = max(activationCount, 1)
    val event = JSONObject()
        .put("m_t", round4(mT))
        .put("dominance", round4(dominance))
        .put("r_t", round4(rT))
        .put("ts", round1(nowMs / 1000.0)) // Python stores ts in epoch seconds
        .toString()
    return ArmeStats(
        meanMT = meanMT + (mT - meanMT) / n,
        meanDominance = meanDominance + (dominance - meanDominance) / n,
        meanRT = meanRT + (rT - meanRT) / n,
        recentEvents = (recentEvents + event).takeLast(HebbianStore.RECENT_EVENTS_MAX),
    )
}

/**
 * Neo4j's cosine score, which the Python system's thresholds and retrieval weights were tuned
 * on: (1 + cos) / 2, in [0, 1] (Cypher manual, vector indexes). Returning raw cosine here
 * doubled the spread of the α·cos retrieval term relative to the Hebbian terms, and made the
 * ingest bootstrap threshold (0.3) far stricter than on the PC (where 0.3 ≙ raw cos −0.4).
 * Null when either vector is empty, zero, or the dimensions differ.
 */
internal fun neo4jCosineScore(query: FloatArray, emb: FloatArray): Double? {
    if (query.isEmpty() || emb.size != query.size) return null
    var dot = 0.0
    var qNorm = 0.0
    var eNorm = 0.0
    for (i in emb.indices) {
        dot += emb[i].toDouble() * query[i]
        qNorm += query[i].toDouble() * query[i]
        eNorm += emb[i].toDouble() * emb[i]
    }
    if (qNorm == 0.0 || eNorm == 0.0) return null
    return (1.0 + dot / (sqrt(qNorm) * sqrt(eNorm))) / 2.0
}

/** Brute-force cosine search with topK×3 over-fetch + type post-filter (graph_store.py:239-278).
 *  Scores are on Neo4j's [0, 1] scale — see [neo4jCosineScore]. */
internal fun cosineSearch(
    concepts: List<ConceptNode>,
    query: FloatArray,
    topK: Int,
    conceptType: String?,
): List<Pair<ConceptNode, Double>> {
    if (concepts.isEmpty() || query.isEmpty() || topK <= 0) return emptyList()
    val fetchK = if (conceptType != null) topK * 3 else topK
    val scored = concepts.mapNotNull { c ->
        neo4jCosineScore(query, c.embedding)?.let { c to it }
    }.sortedByDescending { it.second }.take(fetchK)
    val filtered = if (conceptType != null) scored.filter { it.first.conceptType == conceptType } else scored
    return filtered.take(topK)
}

/**
 * Candidate pool for retrieveForLlm: the vector hits plus up to [extra] nodes that spreading
 * activation from the session seeds reaches strongly (≥ [minSpread] of the peak) but vector
 * search missed. The original kept spread only for nodes already in the vector pool, so the
 * graph could rerank but never recall an association the embedding didn't already surface —
 * contradicting the retrieve_for_llm docstring. Extras are scored by the same formula as
 * everything else; they win only when their association outweighs their lower cosine.
 */
internal fun withSpreadCandidates(
    hits: List<Pair<ConceptNode, Double>>,
    spread: Map<String, Double>,
    query: FloatArray,
    extra: Int,
    minSpread: Double,
    nodeById: (String) -> ConceptNode?,
): List<Pair<ConceptNode, Double>> {
    if (extra <= 0 || spread.isEmpty()) return hits
    val have = hits.mapTo(HashSet()) { it.first.nodeId }
    val added = spread.entries
        .filter { it.key !in have && it.value >= minSpread }
        .sortedByDescending { it.value }
        .asSequence()
        .mapNotNull { (nid, _) -> nodeById(nid)?.let { n -> neo4jCosineScore(query, n.embedding)?.let { n to it } } }
        .take(extra)
        .toList()
    return hits + added
}

/** w' = clamp(w + mT·eta − lam·w) — the co-retrieval reinforcement rule (graph_store.py:394-400). */
internal fun reinforcedWeight(
    current: Double,
    mT: Double,
    eta: Double,
    lam: Double,
    wFloor: Double,
    wCeil: Double,
): Double = (current + mT * eta - lam * current).coerceIn(wFloor, wCeil)

/**
 * Iterative BFS expansion replacing apoc.path.subgraphAll (graph_store.py:416-533).
 * [nodeById] and [outEdges] are backend lookups; [outEdges] may return unfiltered
 * edges — layer/weight filtering happens here. Edges from a frontier node to an
 * already-visited node are kept (they belong to the subgraph), as in the APOC version.
 */
internal fun localSubgraphKernel(
    seedIds: List<String>,
    hops: Int,
    maxNodes: Int,
    layer: String,
    minWeight: Double,
    nodeById: (String) -> ConceptNode?,
    outEdges: (String) -> List<HebbianEdge>,
): Subgraph {
    val seedSet = seedIds.toSet()
    val visited = seedSet.toMutableSet()
    val keptEdges = LinkedHashSet<HebbianEdge>()
    var frontier = seedSet.toList()
    repeat(hops.coerceAtLeast(0)) {
        val next = mutableListOf<String>()
        for (id in frontier) {
            for (e in outEdges(id)) {
                if (e.layer != layer || e.hebbWeight < minWeight) continue
                keptEdges += e
                if (visited.add(e.dstId)) next += e.dstId
            }
        }
        frontier = next
    }
    // Keep seeds + endpoints of surviving edges, cap maxNodes by activationCount DESC,
    // then drop edges whose endpoints were evicted (graph_store.py:487-517).
    val endpointIds = keptEdges.flatMapTo(mutableSetOf()) { listOf(it.srcId, it.dstId) }
    val nodes = (seedSet + endpointIds)
        .mapNotNull(nodeById)
        .sortedByDescending { it.activationCount }
        .take(maxNodes)
    val keptIds = nodes.mapTo(HashSet()) { it.nodeId }
    val edges = keptEdges.filter { it.srcId in keptIds && it.dstId in keptIds }
    return Subgraph(nodes, edges)
}

/**
 * Path enumeration for spreading activation (graph_store.py:918-978). Iterative DFS
 * over layer/weight-filtered edges; each enumerated path (≤ maxPaths) contributes
 * decay^hops · min(pathWeight, W_CEIL); results normalized by peak; seeds excluded.
 */
internal fun spreadingActivationKernel(
    seedIds: List<String>,
    layer: String,
    depth: Int,
    decay: Double,
    minWeight: Double,
    maxPaths: Int,
    outEdges: (String) -> List<HebbianEdge>,
): Map<String, Double> {
    val d = depth.coerceIn(1, 3)
    val seedSet = seedIds.toSet()
    val activations = LinkedHashMap<String, Double>()
    // Stack entries are (nodeId, hops, pathWeight) — one entry per enumerated path.
    val stack = ArrayDeque<Triple<String, Int, Double>>()
    fun expand(from: String, hops: Int, pathWeight: Double) {
        for (e in outEdges(from).sortedBy { it.dstId }) {
            if (e.layer != layer || e.hebbWeight < minWeight) continue
            stack.addLast(Triple(e.dstId, hops + 1, pathWeight * e.hebbWeight))
        }
    }
    for (seed in seedIds) expand(seed, 0, 1.0)
    var paths = 0
    while (stack.isNotEmpty() && paths < maxPaths) {
        val (nid, hops, pathWeight) = stack.removeLast()
        paths++
        if (nid !in seedSet) {
            activations[nid] = (activations[nid] ?: 0.0) +
                decay.pow(hops) * min(pathWeight, HebbianStore.W_CEIL)
        }
        if (hops < d) expand(nid, hops, pathWeight)
    }
    val peak = activations.values.maxOrNull() ?: return emptyMap()
    if (peak <= 0.0) return emptyMap()
    return activations.mapValues { it.value / peak }
}

/** One fading-candidate row before scoring: node plus its hippocampal avg weight and degree. */
internal data class FadingRaw(val node: ConceptNode, val avgW: Double, val degree: Int)

/** fadingScore = (1 − strength) · staleness · ln(1 + activations), sorted DESC (graph_store.py:894-916). */
internal fun rankFadingConcepts(
    raw: List<FadingRaw>,
    nowMs: Long,
    limit: Int,
    wCeil: Double,
): List<FadingConcept> = raw.map { (node, avgW, degree) ->
    val strength = (avgW / wCeil).coerceIn(0.0, 1.0)
    val stale = TypeProfiles.staleness((nowMs - node.updatedAt) / 1000.0, node.conceptType)
    FadingConcept(
        concept = node,
        avgW = avgW,
        degree = degree,
        strength = strength,
        staleness = stale,
        fadingScore = (1.0 - strength) * stale * ln(1.0 + node.activationCount),
    )
}.sortedByDescending { it.fadingScore }.take(limit)

/** Final scoring pass of retrieveForLlm (graph_store.py:815-846). Score components are kept unrounded. */
internal fun combineLlmScores(
    hits: List<Pair<ConceptNode, Double>>,
    assocScores: Map<String, Double>,
    causalScores: Map<String, Double>,
    spreadScores: Map<String, Double>,
    topK: Int,
    alpha: Double,
    beta: Double,
    gamma: Double,
    wCeil: Double,
    deltaSpread: Double,
): List<LlmCandidate> = hits.map { (node, cos) ->
    val assoc = assocScores[node.nodeId] ?: 0.0
    val spread = spreadScores[node.nodeId] ?: 0.0
    val causal = causalScores[node.nodeId] ?: 0.0
    val assocNorm = (assoc / wCeil + deltaSpread * spread).coerceIn(0.0, 1.5)
    LlmCandidate(
        node = node,
        score = alpha * cos + beta * assocNorm + gamma * causal,
        cosSim = cos,
        assoc = assoc,
        spread = spread,
        causal = causal,
    )
}.sortedByDescending { it.score }.take(topK)

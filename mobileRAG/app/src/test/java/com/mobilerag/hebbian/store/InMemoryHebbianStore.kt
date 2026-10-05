package com.mobilerag.hebbian.store

import com.mobilerag.hebbian.ConceptIdentity
import java.io.File
import java.util.UUID
import kotlin.math.max

/**
 * Test-only [HebbianStore] held in maps, with SQLite-store semantics and the same shared
 * pure kernels both real stores delegate to (cosineSearch, localSubgraphKernel,
 * spreadingActivationKernel, rankFadingConcepts, combineLlmScores, armeStatsUpdate, …).
 * Lets the real IngestPipeline / TutorEngine / merge logic run on the JVM.
 */
class InMemoryHebbianStore : HebbianStore {
    override val id = "in-memory-test"

    val concepts = LinkedHashMap<String, ConceptNode>()
    val edges = LinkedHashMap<Triple<String, String, String>, HebbianEdge>()
    val sessionStats = LinkedHashMap<Pair<String, String>, SessionStatRow>()
    private val sessions = HashMap<String, Int>()

    private fun now() = System.currentTimeMillis()

    override suspend fun open(path: String) = Unit
    override suspend fun close() = Unit
    override suspend fun setupSchema() = Unit

    override suspend fun upsertConcept(node: ConceptNode): String {
        val cur = concepts[node.nodeId]
        concepts[node.nodeId] = if (cur == null) {
            node.copy(createdAt = now(), updatedAt = now(), activationCount = 0, meanMT = 1.0,
                meanDominance = 0.5, meanRT = 0.5, recentEvents = emptyList())
        } else {
            cur.copy(label = node.label, textRaw = node.textRaw, textSummary = node.textSummary,
                conceptType = node.conceptType, sourceType = node.sourceType, sourceUri = node.sourceUri,
                updatedAt = now(), embedding = node.embedding)
        }
        return node.nodeId
    }

    override suspend fun getConcept(nodeId: String) = concepts[nodeId]

    override suspend fun incrementActivation(nodeId: String) {
        concepts[nodeId]?.let { concepts[nodeId] = it.copy(activationCount = it.activationCount + 1, updatedAt = now()) }
    }

    override suspend fun findByLabelKey(labelKey: String) =
        if (labelKey.isEmpty()) emptyList() else concepts.values.filter { ConceptIdentity.labelKey(it.label) == labelKey }

    override suspend fun updateConceptEmbeddings(embeddings: Map<String, FloatArray>) {
        for ((id, e) in embeddings) concepts[id]?.let { concepts[id] = it.copy(embedding = e) }
    }

    override suspend fun vectorSearch(embedding: FloatArray, topK: Int, conceptType: String?) =
        cosineSearch(concepts.values.toList(), embedding, topK, conceptType)

    override suspend fun upsertEdge(edge: HebbianEdge) {
        val key = Triple(edge.srcId, edge.dstId, edge.layer)
        val cur = edges[key]
        edges[key] = if (cur == null) edge.copy(lastUpdated = now())
        else cur.copy(hebbWeight = max(cur.hebbWeight, edge.hebbWeight), coActivationCount = cur.coActivationCount + 1, lastUpdated = now())
    }

    override suspend fun updateHebbWeights(updates: List<HebbWeightUpdate>) {
        for (u in updates) {
            val key = Triple(u.srcId, u.dstId, u.layer)
            edges[key]?.let {
                edges[key] = it.copy(hebbWeight = u.weight, eligibility = u.eligibility,
                    coActivationCount = it.coActivationCount + 1, lastUpdated = now())
            }
        }
    }

    override suspend fun updateCausalScores(updates: List<CausalScoreUpdate>, layer: String) {
        for (u in updates) {
            val key = Triple(u.srcId, u.dstId, layer)
            edges[key]?.let { edges[key] = it.copy(causalScore = u.score) }
        }
    }

    override suspend fun reinforceCoRetrieved(
        nodeIds: List<String>, mT: Double, eta: Double, lam: Double, wFloor: Double, wCeil: Double, layer: String,
    ) {
        for (a in nodeIds) for (b in nodeIds) {
            if (a == b) continue
            val key = Triple(a, b, layer)
            val cur = edges[key]
            edges[key] = cur?.copy(
                hebbWeight = reinforcedWeight(cur.hebbWeight, mT, eta, lam, wFloor, wCeil),
                coActivationCount = cur.coActivationCount + 1, lastUpdated = now(),
            ) ?: HebbianEdge(a, b, 1.0, 0.0, 0.0, 1, now(), layer)
        }
    }

    private fun outEdges(nodeId: String): List<HebbianEdge> =
        edges.values.filter { it.srcId == nodeId }.sortedWith(compareBy({ it.dstId }, { it.layer }))

    override suspend fun getLocalSubgraph(seedIds: List<String>, hops: Int, maxNodes: Int, layer: String, minWeight: Double) =
        localSubgraphKernel(seedIds, hops, maxNodes, layer, minWeight, { concepts[it] }, ::outEdges)

    override suspend fun getGlobalGraph(layer: String, minWeight: Double, limit: Int): Subgraph {
        val es = edges.values.filter { it.layer == layer && it.hebbWeight >= minWeight }
            .sortedByDescending { it.hebbWeight }.take(limit)
        val ids = LinkedHashSet<String>().apply { es.forEach { add(it.srcId); add(it.dstId) } }
        return Subgraph(ids.mapNotNull { concepts[it] }, es)
    }

    override suspend fun listConcepts(limit: Int) = concepts.values.sortedByDescending { it.updatedAt }.take(limit)

    override suspend fun updateConceptArmeStats(nodeId: String, mT: Double, dominance: Double, rT: Double) {
        val c = concepts[nodeId] ?: return
        val s = armeStatsUpdate(c.meanMT, c.meanDominance, c.meanRT, c.activationCount, c.recentEvents, mT, dominance, rT, now())
        concepts[nodeId] = c.copy(meanMT = s.meanMT, meanDominance = s.meanDominance, meanRT = s.meanRT, recentEvents = s.recentEvents)
    }

    override suspend fun startSession(trigger: String) = UUID.randomUUID().toString().also { sessions[it] = 0 }
    override suspend fun endSession(sessionId: String) = Unit
    override suspend fun incrementSessionActivations(sessionId: String, count: Int) {
        sessions[sessionId] = (sessions[sessionId] ?: 0) + count
    }

    override suspend fun upsertSessionStats(sessionId: String, nodeId: String, mT: Double, dominance: Double, rT: Double, deltaW: Double) {
        val key = sessionId to nodeId
        val cur = sessionStats[key]
        sessionStats[key] = if (cur == null) {
            SessionStatRow(null, sessionId, nodeId, mT, dominance, rT, 1, deltaW)
        } else {
            val n = cur.activationCount + 1
            cur.copy(meanMT = cur.meanMT + (mT - cur.meanMT) / n, meanDominance = cur.meanDominance + (dominance - cur.meanDominance) / n,
                meanRT = cur.meanRT + (rT - cur.meanRT) / n, deltaWMean = cur.deltaWMean + (deltaW - cur.deltaWMean) / n, activationCount = n)
        }
    }

    override suspend fun retrieveForLlm(
        queryEmbedding: FloatArray, seedNodeIds: List<String>, topK: Int, alpha: Double, beta: Double,
        gamma: Double, wCeil: Double, useSpreading: Boolean, deltaSpread: Double,
    ): List<LlmCandidate> {
        val rawSpread = if (useSpreading && seedNodeIds.isNotEmpty()) {
            spreadingActivationKernel(seedNodeIds, "hippocampal", 2, 0.45, 0.3, 2000, ::outEdges)
        } else emptyMap()
        val hits = withSpreadCandidates(
            cosineSearch(concepts.values.toList(), queryEmbedding, topK * 3, null),
            rawSpread, queryEmbedding, topK, SPREAD_CANDIDATE_MIN, { concepts[it] },
        )
        val cands = hits.mapTo(HashSet()) { it.first.nodeId }
        val fromSeeds = edges.values.filter { it.srcId in seedNodeIds && it.dstId in cands }.groupBy { it.dstId }
        return combineLlmScores(
            hits,
            fromSeeds.mapValues { (_, es) -> es.map { it.hebbWeight }.average() },
            fromSeeds.mapValues { (_, es) -> es.map { it.causalScore }.average() },
            rawSpread.filterKeys { it in cands },
            topK, alpha, beta, gamma, wCeil, deltaSpread,
        )
    }

    override suspend fun getFadingConcepts(limit: Int, minActivations: Int, wCeil: Double, candidateMult: Int): List<FadingConcept> {
        val raw = concepts.values.filter { it.activationCount >= minActivations }.sortedBy { it.updatedAt }
            .take(limit * candidateMult).map { c ->
                val es = edges.values.filter { it.layer == "hippocampal" && (it.srcId == c.nodeId || it.dstId == c.nodeId) }
                FadingRaw(c, if (es.isEmpty()) 0.0 else es.map { it.hebbWeight }.average(), es.size)
            }
        return rankFadingConcepts(raw, now(), limit, wCeil)
    }

    override suspend fun spreadingActivation(
        seedNodeIds: List<String>, layer: String, depth: Int, decay: Double, minWeight: Double, maxPaths: Int,
    ) = spreadingActivationKernel(seedNodeIds, layer, depth, decay, minWeight, maxPaths, ::outEdges)

    override suspend fun getDominanceRegressionData(minActivations: Int) = emptyList<DominanceRegressionRow>()

    override suspend fun clear() {
        concepts.clear(); edges.clear(); sessionStats.clear(); sessions.clear()
    }

    override suspend fun backupTo(dir: File) = Unit

    /** Same flow as the SQLite store, over the maps. */
    override suspend fun applyMerge(plan: MergePlan) {
        val touched = plan.touchedIds
        val hit = edges.values.filter { it.srcId in touched || it.dstId in touched }
        hit.forEach { edges.remove(Triple(it.srcId, it.dstId, it.layer)) }
        ConceptMerge.mergeEdges(hit, plan.remap).forEach { edges[Triple(it.srcId, it.dstId, it.layer)] = it }
        val rows = sessionStats.values.filter { it.nodeId in touched }
        rows.forEach { sessionStats.remove(it.sessionId to it.nodeId) }
        ConceptMerge.mergeSessionStats(rows, plan.remap).forEach { sessionStats[it.sessionId to it.nodeId] = it }
        for (g in plan.groups) concepts[g.keep.nodeId] = ConceptMerge.mergedNode(g)
        plan.remap.keys.forEach { concepts.remove(it) }
    }
}

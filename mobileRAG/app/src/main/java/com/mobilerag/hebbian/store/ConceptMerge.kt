package com.mobilerag.hebbian.store

import com.mobilerag.hebbian.ConceptIdentity
import org.json.JSONObject
import kotlin.math.max

/**
 * Opt-in merge of concepts that were duplicated before concept identity existed. Pure
 * planning + combining rules, shared by both stores (which only do the I/O) and mirrored by
 * `Hebbian Memory/tools/merge_duplicates.py`.
 *
 * Groups: same [ConceptIdentity.labelKey] + same type, events excluded (the ingest rule —
 * the embedding rule isn't applied retroactively; it's for borderline cases the user sees
 * as they happen). Survivor: most activations, ties → earliest created.
 */
data class MergeGroup(val keep: ConceptNode, val drops: List<ConceptNode>)

data class MergePlan(val groups: List<MergeGroup>) {
    /** dropped nodeId → surviving nodeId */
    val remap: Map<String, String> = groups.flatMap { g -> g.drops.map { it.nodeId to g.keep.nodeId } }.toMap()

    /** Every node a group touches (survivors + dropped). */
    val touchedIds: Set<String> = groups.flatMapTo(LinkedHashSet()) { g -> listOf(g.keep.nodeId) + g.drops.map { it.nodeId } }

    val droppedCount: Int get() = remap.size
}

/** One per-(session, concept) stats row; [statId] is LadybugDB's key (null for SQLite). */
data class SessionStatRow(
    val statId: String?,
    val sessionId: String,
    val nodeId: String,
    val meanMT: Double,
    val meanDominance: Double,
    val meanRT: Double,
    val activationCount: Int,
    val deltaWMean: Double,
)

object ConceptMerge {

    fun plan(concepts: List<ConceptNode>): MergePlan {
        val groups = concepts
            .filter { it.conceptType != ConceptIdentity.EVENT_TYPE }
            .groupBy { ConceptIdentity.labelKey(it.label) to it.conceptType }
            .filter { (key, members) -> key.first.isNotEmpty() && members.size > 1 }
            .values
            .map { members ->
                val keep = members.maxWith(compareBy<ConceptNode> { it.activationCount }.thenByDescending { it.createdAt })
                MergeGroup(keep, members.filter { it !== keep })
            }
            .sortedBy { it.keep.nodeId }
        return MergePlan(groups)
    }

    /** The survivor with the group's combined history; label, texts and embedding stay the survivor's. */
    fun mergedNode(g: MergeGroup): ConceptNode {
        val all = listOf(g.keep) + g.drops
        val weights = all.map { max(it.activationCount, 1).toDouble() }
        val total = weights.sum()
        fun weighted(f: (ConceptNode) -> Double) = all.indices.sumOf { f(all[it]) * weights[it] } / total
        val events = all.flatMap { it.recentEvents }
            .sortedBy { runCatching { JSONObject(it).optDouble("ts", 0.0) }.getOrDefault(0.0) }
            .takeLast(HebbianStore.RECENT_EVENTS_MAX)
        return g.keep.copy(
            activationCount = all.sumOf { it.activationCount },
            createdAt = all.minOf { it.createdAt },
            updatedAt = all.maxOf { it.updatedAt },
            meanMT = weighted { it.meanMT },
            meanDominance = weighted { it.meanDominance },
            meanRT = weighted { it.meanRT },
            recentEvents = events,
        )
    }

    /**
     * Re-point [edges] (every edge touching a merged node) through [remap]; drop self loops;
     * combine collisions on (src, dst, layer): weight, eligibility and causal take the max,
     * co-activations add, lastUpdated takes the latest.
     */
    fun mergeEdges(edges: List<HebbianEdge>, remap: Map<String, String>): List<HebbianEdge> {
        val out = LinkedHashMap<Triple<String, String, String>, HebbianEdge>()
        for (e in edges) {
            val s = remap[e.srcId] ?: e.srcId
            val d = remap[e.dstId] ?: e.dstId
            if (s == d) continue
            val key = Triple(s, d, e.layer)
            val cur = out[key]
            out[key] = if (cur == null) {
                e.copy(srcId = s, dstId = d)
            } else {
                cur.copy(
                    hebbWeight = max(cur.hebbWeight, e.hebbWeight),
                    eligibility = max(cur.eligibility, e.eligibility),
                    causalScore = max(cur.causalScore, e.causalScore),
                    coActivationCount = cur.coActivationCount + e.coActivationCount,
                    lastUpdated = max(cur.lastUpdated, e.lastUpdated),
                )
            }
        }
        return out.values.toList()
    }

    /** Re-point stats rows; on a (session, concept) collision keep the row with more activations. */
    fun mergeSessionStats(rows: List<SessionStatRow>, remap: Map<String, String>): List<SessionStatRow> =
        rows.map { r -> remap[r.nodeId]?.let { r.copy(nodeId = it) } ?: r }
            .groupBy { it.sessionId to it.nodeId }
            .values
            .map { same -> same.maxBy { it.activationCount } }
}

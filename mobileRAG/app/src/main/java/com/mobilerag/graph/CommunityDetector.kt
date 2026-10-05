package com.mobilerag.graph

import com.mobilerag.core.GraphStore

/**
 * Deterministic weighted label propagation over the entity graph.
 *
 * The graph is treated as undirected; parallel edges between a pair (one per co-occurring
 * chunk) are summed into a single neighbor weight. Nodes are processed in a fixed order
 * (ascending id), label ties break toward the lowest community label, and sweeps stop when
 * a full pass changes nothing (or after [MAX_ITERATIONS]). Communities smaller than
 * [MIN_COMMUNITY_SIZE] are merged into the neighboring community with the strongest total
 * edge weight; isolated leftovers are dropped.
 */
class CommunityDetector {

    data class Community(val memberIds: List<Long>, val internalEdgeWeight: Double)

    fun detect(entities: List<GraphStore.Entity>, edges: List<GraphStore.Edge>): List<Community> {
        if (entities.isEmpty()) return emptyList()
        val ids = entities.map { it.id }.sorted()
        val index = ids.withIndex().associate { (i, id) -> id to i }
        val n = ids.size

        // adjacency[i][j] = summed weight between node i and node j (undirected)
        val adjacency = Array(n) { HashMap<Int, Double>() }
        for (e in edges) {
            val a = index[e.fromId] ?: continue
            val b = index[e.toId] ?: continue
            if (a == b) continue
            adjacency[a].merge(b, e.weight, Double::plus)
            adjacency[b].merge(a, e.weight, Double::plus)
        }

        // Synchronous-ish sweeps: in-order updates, majority neighbor label wins,
        // ties (and empty neighborhoods) keep the current label, lowest label on tie.
        val labels = IntArray(n) { it }
        repeat(MAX_ITERATIONS) {
            var changed = false
            for (i in 0 until n) {
                val scores = HashMap<Int, Double>()
                for ((j, w) in adjacency[i]) scores.merge(labels[j], w, Double::plus)
                val best = scores.entries.maxWithOrNull(compareBy({ it.value }, { -it.key }))
                if (best != null && best.key != labels[i]) {
                    labels[i] = best.key
                    changed = true
                }
            }
            if (!changed) return@repeat
        }

        val groups = labels.indices.groupBy { labels[it] }.values
            .map { members -> members.map { ids[it] }.sorted() }
            .sortedBy { it.first() }
            .toMutableList()

        // Merge undersized communities into their strongest-weight neighbor community.
        var i = 0
        while (i < groups.size) {
            val members = groups[i]
            if (members.size >= MIN_COMMUNITY_SIZE) { i++; continue }
            val memberSet = members.toSet()
            val pull = HashMap<Int, Double>() // target group index -> total weight
            for (id in members) {
                for ((j, w) in adjacency[index.getValue(id)]) {
                    val neighborId = ids[j]
                    if (neighborId in memberSet) continue
                    val target = groups.indices.firstOrNull { g -> g != i && neighborId in groups[g] } ?: continue
                    pull.merge(target, w, Double::plus)
                }
            }
            val best = pull.entries.maxWithOrNull(compareBy({ it.value }, { -it.key }))
            if (best == null) {
                groups.removeAt(i) // isolated leftovers are dropped
            } else {
                val merged = (groups[best.key] + members).sorted()
                groups[best.key] = merged
                groups.removeAt(i)
            }
        }

        return groups.map { members ->
            val memberSet = members.toSet()
            var internal = 0.0
            for (id in members) {
                for ((j, w) in adjacency[index.getValue(id)]) {
                    if (ids[j] in memberSet && ids[j] > id) internal += w // count each pair once
                }
            }
            Community(members, internal)
        }
    }

    private companion object {
        const val MAX_ITERATIONS = 20
        const val MIN_COMMUNITY_SIZE = 3
    }
}

package com.mobilerag.hebbian.cortical

import com.mobilerag.hebbian.store.ConceptNode
import com.mobilerag.hebbian.store.Subgraph
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Pure data preparation for a consolidation run (consolidation_scheduler.py steps 2–3),
 * kept free of Android so it's unit-testable.
 */
internal object CorticalPrep {

    class Prepared(
        val ids: List<String>,
        /** L2-normalized node embeddings [N, D] (F.normalize). */
        val x: Mat,
        val src: IntArray,
        val dst: IntArray,
        val w: DoubleArray,
        /** Nodes left out: wrong/missing embedding size, or beyond the node cap. */
        val droppedNodes: Int,
    ) {
        val dim: Int get() = x.cols
    }

    /**
     * Graph → training tensors. Nodes whose embedding size differs from the majority are
     * dropped (Python would fail on the ragged tensor). Above [maxNodes], only the nodes with
     * the largest summed incident edge weight are kept — a phone-only safety cap on memory
     * and time; the Python original has none. Returns null when no node has an embedding.
     */
    fun prepare(sg: Subgraph, maxNodes: Int): Prepared? {
        val dim = sg.nodes.map { it.embedding.size }.filter { it > 0 }
            .groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: return null
        var nodes = sg.nodes.filter { it.embedding.size == dim }
        var keep = nodes.mapTo(HashSet()) { it.nodeId }
        var edges = sg.edges.filter { it.srcId in keep && it.dstId in keep }

        if (nodes.size > maxNodes) {
            val strength = HashMap<String, Double>()
            for (e in edges) {
                strength[e.srcId] = (strength[e.srcId] ?: 0.0) + e.hebbWeight
                strength[e.dstId] = (strength[e.dstId] ?: 0.0) + e.hebbWeight
            }
            keep = nodes.sortedWith(compareByDescending<ConceptNode> { strength[it.nodeId] ?: 0.0 }
                .thenBy { it.nodeId })
                .take(maxNodes).mapTo(HashSet()) { it.nodeId }
            nodes = nodes.filter { it.nodeId in keep } // original order kept
            edges = edges.filter { it.srcId in keep && it.dstId in keep }
        }

        val ids = nodes.map { it.nodeId }
        val index = ids.withIndex().associate { (i, id) -> id to i }
        val x = Mat(nodes.size, dim)
        for ((i, node) in nodes.withIndex()) {
            var norm = 0.0
            for (v in node.embedding) norm += v.toDouble() * v
            val denom = max(sqrt(norm), 1e-12)
            for (k in 0 until dim) x.data[i * dim + k] = node.embedding[k] / denom
        }
        return Prepared(
            ids = ids,
            x = x,
            src = IntArray(edges.size) { index.getValue(edges[it].srcId) },
            dst = IntArray(edges.size) { index.getValue(edges[it].dstId) },
            w = DoubleArray(edges.size) { edges[it].hebbWeight },
            droppedNodes = sg.nodes.size - nodes.size,
        )
    }

    /**
     * z_prev aligned to this run's node order (consolidation_scheduler.py step 3): a node
     * seen last time keeps its previous cortical embedding; a new node is anchored to its
     * current (normalized) embedding. Null when there is no previous state or its width
     * differs — then consolidate() anchors everything to x, as Python's first run does.
     */
    fun alignZPrev(prevIds: List<String>, prev: Mat, ids: List<String>, x: Mat): Mat? {
        if (prevIds.isEmpty() || prev.cols != x.cols || prev.rows != prevIds.size) return null
        val prevRow = prevIds.withIndex().associate { (i, id) -> id to i }
        val d = x.cols
        val out = Mat(ids.size, d)
        for ((i, id) in ids.withIndex()) {
            val r = prevRow[id]
            if (r != null) System.arraycopy(prev.data, r * d, out.data, i * d, d)
            else System.arraycopy(x.data, i * d, out.data, i * d, d)
        }
        return out
    }

    /** F.cosine_similarity of two rows of z (eps 1e-8) — the causal score per edge. */
    fun rowCosine(z: Mat, i: Int, j: Int): Double {
        val d = z.cols
        var ab = 0.0
        var aa = 0.0
        var bb = 0.0
        for (k in 0 until d) {
            val a = z.data[i * d + k]
            val b = z.data[j * d + k]
            ab += a * b
            aa += a * a
            bb += b * b
        }
        return ab / (max(sqrt(aa), Tape.COS_EPS) * max(sqrt(bb), Tape.COS_EPS))
    }
}

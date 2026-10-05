package com.mobilerag.hebbian

import kotlin.math.max
import kotlin.math.tanh

/**
 * Message-passing path of the hippocampal GNN (port of the propagate() path
 * in core/hippocampal_gnn_pyg.py, single HebbianConv layer).
 *
 * Per destination node j: in_deg[j] = sum of incoming edge weights, clamped
 * to min 1.0. Message along edge (i -> j): (w_ij / in_deg[j]) * x_i.
 * Aggregation is sum; output is tanh(agg). The Hebbian weight update is NOT
 * here — it lives in [Arme.modulatedHebbianUpdate].
 */
object HippocampalGnn {

    fun propagate(
        x: Array<FloatArray>,
        edgeSrc: IntArray,
        edgeDst: IntArray,
        edgeWeight: DoubleArray,
    ): Array<FloatArray> {
        val n = x.size
        if (n == 0) return emptyArray()
        val d = x[0].size
        val e = edgeSrc.size
        require(edgeDst.size == e && edgeWeight.size == e) { "edge arrays must have equal length" }

        val inDeg = DoubleArray(n)
        for (i in 0 until e) inDeg[edgeDst[i]] += edgeWeight[i]
        for (j in 0 until n) inDeg[j] = max(inDeg[j], 1.0)

        val out = Array(n) { FloatArray(d) }
        for (i in 0 until e) {
            val scale = edgeWeight[i] / inDeg[edgeDst[i]]
            val xs = x[edgeSrc[i]]
            val o = out[edgeDst[i]]
            for (k in 0 until d) o[k] = (o[k] + scale * xs[k]).toFloat()
        }
        for (j in 0 until n) {
            val o = out[j]
            for (k in 0 until d) o[k] = tanh(o[k].toDouble()).toFloat()
        }
        return out
    }
}

package com.mobilerag.hebbian.cortical

import java.io.DataInputStream
import java.io.DataOutputStream
import java.util.Random
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Cortical consolidation network — port of `core/cortical_gnn.py` (CorticalGNN + consolidate),
 * verified against the real PyTorch/PyG code by CorticalGnnGoldenTest
 * (docs/cortical_golden.py → docs/cortical-golden-values.json).
 *
 * Two PyG-2.8 GATConv layers: GAT(heads, concat) → ELU → dropout → GAT(1 head, mean).
 * GATConv semantics reproduced exactly: one bias-free `lin`; att_src/att_dst per head;
 * existing self loops removed and one per node added; LeakyReLU(0.2) on
 * a_src[j] + a_dst[i]; softmax over each destination's incoming edges; attention dropout
 * while training; heads concatenated or averaged; + bias. Edge weights do NOT enter the
 * attention (as in Python) — they only supervise the loss:
 *
 *   L = recon + λ_causal·causal + λ_consist·consist
 *   recon  = mean(w_norm · BCE(z_i·z_j, 1)) over edges + mean(BCE(z_i·z_j, 0)) over negatives
 *   causal = mean((cos(z_i, z_j) − (2·w_norm − 1))²)
 *   consist = mean((z − z_prev)²)
 */
data class CorticalConfig(
    val inDim: Int,
    val hidden: Int,
    val heads: Int,
    val outDim: Int,
    val dropout: Double = 0.1,
    val lambdaCausal: Double = 0.3,
    val lambdaConsist: Double = 0.5,
    val epochs: Int = 30,
    val lr: Double = 5e-4,
) {
    companion object {
        /** Parity with the PC build (tutor_engine.py: hidden 256 × 4 heads, 30 epochs,
         *  lr 5e-4) at the phone's embedding size. */
        fun production(embeddingDim: Int) = CorticalConfig(
            inDim = embeddingDim, hidden = 256, heads = 4, outDim = embeddingDim,
        )
    }
}

/** The eight parameter tensors, named exactly as the PyTorch state_dict. */
class CorticalParams(val config: CorticalConfig) {
    private val hc = config.hidden * config.heads
    val gat1AttSrc = Mat(1, hc)
    val gat1AttDst = Mat(1, hc)
    val gat1Bias = Mat(1, hc)
    val gat1Lin = Mat(hc, config.inDim)
    val gat2AttSrc = Mat(1, config.outDim)
    val gat2AttDst = Mat(1, config.outDim)
    val gat2Bias = Mat(1, config.outDim)
    val gat2Lin = Mat(config.outDim, hc)

    val named: LinkedHashMap<String, Mat> = linkedMapOf(
        "gat1.att_src" to gat1AttSrc,
        "gat1.att_dst" to gat1AttDst,
        "gat1.bias" to gat1Bias,
        "gat1.lin.weight" to gat1Lin,
        "gat2.att_src" to gat2AttSrc,
        "gat2.att_dst" to gat2AttDst,
        "gat2.bias" to gat2Bias,
        "gat2.lin.weight" to gat2Lin,
    )

    val count: Int get() = named.values.sumOf { it.data.size }

    fun writeTo(out: DataOutputStream) {
        out.writeInt(MAGIC)
        out.writeInt(config.inDim)
        out.writeInt(config.hidden)
        out.writeInt(config.heads)
        out.writeInt(config.outDim)
        for (m in named.values) for (v in m.data) out.writeDouble(v)
    }

    companion object {
        private const val MAGIC = 0x434f5254 // "CORT"

        /** PyG initialisation: glorot-uniform lin and attention vectors, zero bias. For an
         *  attention tensor [1, H, C] PyG's glorot uses fan = H + C. */
        fun glorot(config: CorticalConfig, rng: Random): CorticalParams = CorticalParams(config).apply {
            fun uniform(m: Mat, fan: Int) {
                val a = sqrt(6.0 / fan)
                for (i in m.data.indices) m.data[i] = (rng.nextDouble() * 2.0 - 1.0) * a
            }
            uniform(gat1Lin, gat1Lin.rows + gat1Lin.cols)
            uniform(gat1AttSrc, config.heads + config.hidden)
            uniform(gat1AttDst, config.heads + config.hidden)
            uniform(gat2Lin, gat2Lin.rows + gat2Lin.cols)
            uniform(gat2AttSrc, 1 + config.outDim)
            uniform(gat2AttDst, 1 + config.outDim)
        }

        /** Reads parameters written by [writeTo]; null when the file was written for a
         *  different architecture (e.g. another embedding size). */
        fun readFrom(input: DataInputStream, expected: CorticalConfig): CorticalParams? {
            if (input.readInt() != MAGIC) return null
            val dims = IntArray(4) { input.readInt() }
            if (dims[0] != expected.inDim || dims[1] != expected.hidden ||
                dims[2] != expected.heads || dims[3] != expected.outDim
            ) return null
            return CorticalParams(expected).apply {
                for (m in named.values) for (i in m.data.indices) m.data[i] = input.readDouble()
            }
        }
    }
}

data class LossParts(val total: Double, val recon: Double, val causal: Double, val consist: Double)

/** Draws the negative (non-edge) pairs for one epoch's reconstruction loss. */
fun interface NegativeSampler {
    fun sample(epoch: Int, src: IntArray, dst: IntArray, n: Int, numNeg: Int): Pair<IntArray, IntArray>
}

/** Port of cortical_gnn._sample_negative_edges: rejection-sample random ordered pairs
 *  i ≠ j not in the graph, up to numNeg·20 attempts; if none, the reversed edges. */
class RejectionNegativeSampler(private val rng: Random) : NegativeSampler {
    override fun sample(epoch: Int, src: IntArray, dst: IntArray, n: Int, numNeg: Int): Pair<IntArray, IntArray> {
        val existing = HashSet<Long>(src.size * 2)
        for (e in src.indices) existing += pairKey(src[e], dst[e])
        val ns = ArrayList<Int>(numNeg)
        val nd = ArrayList<Int>(numNeg)
        var attempts = 0
        while (ns.size < numNeg && attempts < numNeg * 20) {
            val i = rng.nextInt(n)
            val j = rng.nextInt(n)
            if (i != j && pairKey(i, j) !in existing) {
                ns += i
                nd += j
            }
            attempts++
        }
        if (ns.isEmpty()) {
            val k = min(numNeg, src.size)
            return IntArray(k) { dst[it] } to IntArray(k) { src[it] }
        }
        return ns.toIntArray() to nd.toIntArray()
    }

    private fun pairKey(i: Int, j: Int): Long = (i.toLong() shl 32) or (j.toLong() and 0xffffffffL)
}

/** torch.optim.Adam with default β₁ 0.9, β₂ 0.999, ε 1e-8, no weight decay. */
class Adam(private val params: List<Mat>, private val lr: Double) {
    private val m = params.map { DoubleArray(it.data.size) }
    private val v = params.map { DoubleArray(it.data.size) }
    private var t = 0

    fun step(grads: List<Mat?>) {
        t++
        val bc1 = 1.0 - B1.pow(t)
        val bc2 = 1.0 - B2.pow(t)
        for ((k, p) in params.withIndex()) {
            val g = grads[k]?.data ?: continue
            val mk = m[k]
            val vk = v[k]
            for (i in p.data.indices) {
                mk[i] = B1 * mk[i] + (1 - B1) * g[i]
                vk[i] = B2 * vk[i] + (1 - B2) * g[i] * g[i]
                p.data[i] -= (lr / bc1) * mk[i] / (sqrt(vk[i]) / sqrt(bc2) + EPS)
            }
        }
    }

    private companion object {
        const val B1 = 0.9
        const val B2 = 0.999
        const val EPS = 1e-8
    }
}

class CorticalGnn(val params: CorticalParams) {

    private val cfg = params.config

    /** Message-passing edges: input edges minus self loops, plus one self loop per node. */
    private class MsgGraph(val src: IntArray, val dst: IntArray, val n: Int)

    private fun msgGraph(src: IntArray, dst: IntArray, n: Int): MsgGraph {
        val keep = src.indices.filter { src[it] != dst[it] }
        return MsgGraph(
            IntArray(keep.size + n) { if (it < keep.size) src[keep[it]] else it - keep.size },
            IntArray(keep.size + n) { if (it < keep.size) dst[keep[it]] else it - keep.size },
            n,
        )
    }

    private class Leaves(tape: Tape, p: CorticalParams, grad: Boolean) {
        val all = p.named.mapValues { (_, m) -> tape.leaf(m, grad) }
        val g1Src = all.getValue("gat1.att_src")
        val g1Dst = all.getValue("gat1.att_dst")
        val g1Bias = all.getValue("gat1.bias")
        val g1Lin = all.getValue("gat1.lin.weight")
        val g2Src = all.getValue("gat2.att_src")
        val g2Dst = all.getValue("gat2.att_dst")
        val g2Bias = all.getValue("gat2.bias")
        val g2Lin = all.getValue("gat2.lin.weight")
    }

    private fun gat(
        tape: Tape, x: Var, lin: Var, attSrc: Var, attDst: Var, bias: Var,
        heads: Int, concat: Boolean, g: MsgGraph, training: Boolean, rng: Random,
    ): Var {
        val h = tape.linear(x, lin)
        val aSrc = tape.headDot(h, attSrc, heads)
        val aDst = tape.headDot(h, attDst, heads)
        val logits = tape.leakyRelu(tape.add(tape.gatherRows(aSrc, g.src), tape.gatherRows(aDst, g.dst)), 0.2)
        var alpha = tape.segmentSoftmax(logits, g.dst, g.n)
        if (training) alpha = tape.dropout(alpha, cfg.dropout, rng)
        // Fused gather → weight → scatter (+ head mean) + bias: never materializes [E, H·C].
        return tape.attentionAggregate(h, alpha, g.src, g.dst, g.n, heads, mean = !concat, bias = bias)
    }

    private fun encode(tape: Tape, x: Var, p: Leaves, g: MsgGraph, training: Boolean, rng: Random): Var {
        val h = gat(tape, x, p.g1Lin, p.g1Src, p.g1Dst, p.g1Bias, cfg.heads, true, g, training, rng)
        val act = tape.eluDropout(h, cfg.dropout, rng, training)
        return gat(tape, act, p.g2Lin, p.g2Src, p.g2Dst, p.g2Bias, 1, false, g, training, rng)
    }

    /** Inference (eval mode, no dropout): x [N, inDim] → z [N, outDim]. */
    fun encode(x: Mat, src: IntArray, dst: IntArray): Mat {
        val tape = Tape()
        return encode(tape, tape.leaf(x), Leaves(tape, params, grad = false), msgGraph(src, dst, x.rows),
            training = false, rng = Random(0)).value
    }

    private fun loss(
        tape: Tape, z: Var, src: IntArray, dst: IntArray, w: DoubleArray, zPrev: Mat,
        negSrc: IntArray, negDst: IntArray,
    ): Pair<Var, LossParts> {
        val wMax = w.maxOrNull() ?: 0.0
        val wNorm = DoubleArray(w.size) { w[it] / (wMax + 1e-8) }

        // Pair ops read z directly — no [E, D] gathers (see Tape fused ops).
        val posLoss = tape.mean(tape.mulConst(tape.bceWithLogits(tape.pairDot(z, src, dst), 1.0), wNorm))
        val negLoss = tape.mean(tape.bceWithLogits(tape.pairDot(z, negSrc, negDst), 0.0))
        val recon = tape.add(posLoss, negLoss)

        val expected = DoubleArray(w.size) { max(-1.0, min(1.0, wNorm[it] * 2 - 1)) }
        val causal = tape.mseConst(tape.pairCosine(z, src, dst), expected)
        val consist = tape.mseConst(z, zPrev.data)

        val total = tape.add(
            tape.add(recon, tape.scale(causal, cfg.lambdaCausal)),
            tape.scale(consist, cfg.lambdaConsist),
        )
        val parts = LossParts(
            total = total.value.data[0],
            recon = recon.value.data[0],
            causal = causal.value.data[0],
            consist = consist.value.data[0],
        )
        return total to parts
    }

    /** One forward + backward pass without updating (for golden-value checks). */
    fun lossAndGradients(
        x: Mat, src: IntArray, dst: IntArray, w: DoubleArray, zPrev: Mat?,
        negSrc: IntArray, negDst: IntArray, rng: Random = Random(0),
    ): Triple<Mat, LossParts, Map<String, Mat>> {
        val tape = Tape()
        val leaves = Leaves(tape, params, grad = true)
        val z = encode(tape, tape.leaf(x), leaves, msgGraph(src, dst, x.rows), training = true, rng = rng)
        val (total, parts) = loss(tape, z, src, dst, w, zPrev ?: x, negSrc, negDst)
        tape.backward(total)
        return Triple(z.value, parts, leaves.all.mapValues { (_, v) -> v.grad ?: Mat(v.rows, v.cols) })
    }

    /**
     * One consolidation session (cortical_gnn.consolidate): a fresh Adam, [CorticalConfig.epochs]
     * steps in training mode, then an eval-mode encode. [zPrev] null anchors to [x], as in
     * Python's first consolidation. Mutates [params]; returns the final z and loss history.
     */
    fun consolidate(
        x: Mat,
        src: IntArray,
        dst: IntArray,
        w: DoubleArray,
        zPrev: Mat?,
        sampler: NegativeSampler,
        rng: Random,
        epochs: Int = cfg.epochs,
        lr: Double = cfg.lr,
        onEpoch: (Int, LossParts) -> Unit = { _, _ -> },
    ): Pair<Mat, List<LossParts>> {
        val anchor = zPrev ?: x.copy()
        val g = msgGraph(src, dst, x.rows)
        val order = params.named.values.toList()
        val adam = Adam(order, lr)
        val history = ArrayList<LossParts>(epochs)
        for (epoch in 0 until epochs) {
            val tape = Tape()
            val leaves = Leaves(tape, params, grad = true)
            val z = encode(tape, tape.leaf(x), leaves, g, training = true, rng = rng)
            val (negSrc, negDst) = sampler.sample(epoch, src, dst, x.rows, src.size)
            val (total, parts) = loss(tape, z, src, dst, w, anchor, negSrc, negDst)
            tape.backward(total)
            adam.step(leaves.all.values.map { it.grad })
            history += parts
            onEpoch(epoch, parts)
        }
        return encode(x, src, dst) to history
    }
}

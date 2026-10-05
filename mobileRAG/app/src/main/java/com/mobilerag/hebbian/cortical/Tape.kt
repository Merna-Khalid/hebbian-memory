package com.mobilerag.hebbian.cortical

import java.util.Random
import java.util.stream.IntStream
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Minimal reverse-mode autodiff for the cortical GAT (the phone has no PyTorch, and ONNX
 * Runtime's on-device training was deprecated after 1.19.2). Only the ops CorticalGnn needs;
 * each op's backward is finite-difference tested on its own (CorticalTapeTest).
 *
 * Values are row-major Double matrices. [Var]s created from leaves with requiresGrad=true
 * (parameters) propagate the flag; [Tape.backward] walks recorded ops in reverse.
 * Not thread-safe; the heavy loops inside [Tape.linear] run in parallel over rows.
 */
class Mat(val rows: Int, val cols: Int, val data: DoubleArray = DoubleArray(rows * cols)) {
    init {
        require(data.size == rows * cols) { "data size ${data.size} != $rows x $cols" }
    }

    operator fun get(r: Int, c: Int): Double = data[r * cols + c]
    fun copy(): Mat = Mat(rows, cols, data.copyOf())
}

class Var internal constructor(val value: Mat, val requiresGrad: Boolean) {
    var grad: Mat? = null
        internal set
    internal var backwardFn: (() -> Unit)? = null

    internal fun gradBuf(): Mat = grad ?: Mat(value.rows, value.cols).also { grad = it }
    val rows: Int get() = value.rows
    val cols: Int get() = value.cols
}

class Tape {
    private val recorded = ArrayList<Var>()

    fun leaf(m: Mat, requiresGrad: Boolean = false): Var = Var(m, requiresGrad)

    private inline fun record(value: Mat, inputs: Array<out Var>, crossinline backward: (out: Var) -> Unit): Var {
        val v = Var(value, inputs.any { it.requiresGrad })
        if (v.requiresGrad) {
            v.backwardFn = { backward(v) }
            recorded += v
        }
        return v
    }

    /** Seeds d(loss)/d(loss) = 1 and back-propagates through every recorded op. */
    fun backward(loss: Var) {
        require(loss.rows == 1 && loss.cols == 1) { "loss must be a scalar" }
        loss.gradBuf().data[0] = 1.0
        for (i in recorded.indices.reversed()) {
            val v = recorded[i]
            if (v.grad != null) v.backwardFn?.invoke()
        }
    }

    // ── Dense ─────────────────────────────────────────────────────────

    /** y = x·Wᵀ (torch Linear without bias): x [N, in], w [out, in] → [N, out]. */
    fun linear(x: Var, w: Var): Var {
        require(x.cols == w.cols) { "linear: x.cols ${x.cols} != w.cols ${w.cols}" }
        val n = x.rows
        val inDim = x.cols
        val outDim = w.rows
        val xd = x.value.data
        val wd = w.value.data
        val y = Mat(n, outDim)
        val yd = y.data
        parallelRows(n) { i ->
            val xo = i * inDim
            for (o in 0 until outDim) {
                val wo = o * inDim
                var s = 0.0
                for (k in 0 until inDim) s += xd[xo + k] * wd[wo + k]
                yd[i * outDim + o] = s
            }
        }
        return record(y, arrayOf(x, w)) { out ->
            val g = out.grad!!.data
            if (x.requiresGrad) {
                val dx = x.gradBuf().data
                parallelRows(n) { i ->
                    val xo = i * inDim
                    for (o in 0 until outDim) {
                        val go = g[i * outDim + o]
                        if (go == 0.0) continue
                        val wo = o * inDim
                        for (k in 0 until inDim) dx[xo + k] += go * wd[wo + k]
                    }
                }
            }
            if (w.requiresGrad) {
                val dw = w.gradBuf().data
                parallelRows(outDim) { o ->
                    val wo = o * inDim
                    for (i in 0 until n) {
                        val go = g[i * outDim + o]
                        if (go == 0.0) continue
                        val xo = i * inDim
                        for (k in 0 until inDim) dw[wo + k] += go * xd[xo + k]
                    }
                }
            }
        }
    }

    /** Per-head dot with an attention vector: x [N, H·C], att [1, H·C] → [N, H]. */
    fun headDot(x: Var, att: Var, heads: Int): Var {
        val c = x.cols / heads
        require(att.cols == x.cols && att.rows == 1)
        val n = x.rows
        val xd = x.value.data
        val ad = att.value.data
        val y = Mat(n, heads)
        for (i in 0 until n) for (h in 0 until heads) {
            var s = 0.0
            for (k in 0 until c) s += xd[i * x.cols + h * c + k] * ad[h * c + k]
            y.data[i * heads + h] = s
        }
        return record(y, arrayOf(x, att)) { out ->
            val g = out.grad!!.data
            if (x.requiresGrad) {
                val dx = x.gradBuf().data
                for (i in 0 until n) for (h in 0 until heads) {
                    val gh = g[i * heads + h]
                    for (k in 0 until c) dx[i * x.cols + h * c + k] += gh * ad[h * c + k]
                }
            }
            if (att.requiresGrad) {
                val da = att.gradBuf().data
                for (i in 0 until n) for (h in 0 until heads) {
                    val gh = g[i * heads + h]
                    for (k in 0 until c) da[h * c + k] += gh * xd[i * x.cols + h * c + k]
                }
            }
        }
    }

    // ── Graph plumbing ────────────────────────────────────────────────

    /** out[e] = x[idx[e]]: x [N, K] → [E, K]. */
    fun gatherRows(x: Var, idx: IntArray): Var {
        val k = x.cols
        val y = Mat(idx.size, k)
        for (e in idx.indices) System.arraycopy(x.value.data, idx[e] * k, y.data, e * k, k)
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (e in idx.indices) {
                val src = e * k
                val dst = idx[e] * k
                for (j in 0 until k) dx[dst + j] += g[src + j]
            }
        }
    }

    /** out[idx[e]] += x[e]: x [E, K] → [n, K]. */
    fun scatterAddRows(x: Var, idx: IntArray, n: Int): Var {
        val k = x.cols
        val y = Mat(n, k)
        for (e in idx.indices) {
            val src = e * k
            val dst = idx[e] * k
            for (j in 0 until k) y.data[dst + j] += x.value.data[src + j]
        }
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (e in idx.indices) {
                val src = e * k
                val dst = idx[e] * k
                for (j in 0 until k) dx[src + j] += g[dst + j]
            }
        }
    }

    /**
     * Softmax over entries sharing a segment id, per column (PyG `softmax(src, index)`):
     * exp(x − max_seg) / (Σ_seg + 1e-16). logits [E, H], seg [E] in 0 until nSeg.
     */
    fun segmentSoftmax(logits: Var, seg: IntArray, nSeg: Int): Var {
        val e = logits.rows
        val h = logits.cols
        val x = logits.value.data
        val segMax = DoubleArray(nSeg * h) { Double.NEGATIVE_INFINITY }
        for (i in 0 until e) for (c in 0 until h) {
            val s = seg[i] * h + c
            if (x[i * h + c] > segMax[s]) segMax[s] = x[i * h + c]
        }
        val y = Mat(e, h)
        val segSum = DoubleArray(nSeg * h)
        for (i in 0 until e) for (c in 0 until h) {
            val v = exp(x[i * h + c] - segMax[seg[i] * h + c])
            y.data[i * h + c] = v
            segSum[seg[i] * h + c] += v
        }
        for (i in 0 until e) for (c in 0 until h) y.data[i * h + c] /= (segSum[seg[i] * h + c] + 1e-16)
        return record(y, arrayOf(logits)) { out ->
            val g = out.grad!!.data
            val a = out.value.data
            val dot = DoubleArray(nSeg * h)
            for (i in 0 until e) for (c in 0 until h) dot[seg[i] * h + c] += a[i * h + c] * g[i * h + c]
            val dx = logits.gradBuf().data
            for (i in 0 until e) for (c in 0 until h) {
                dx[i * h + c] += a[i * h + c] * (g[i * h + c] - dot[seg[i] * h + c])
            }
        }
    }

    /** out[e, h·C + c] = w[e, h] · x[e, h·C + c]: x [E, H·C], w [E, H]. */
    fun headScaleRows(x: Var, w: Var, heads: Int): Var {
        val c = x.cols / heads
        require(w.rows == x.rows && w.cols == heads)
        val e = x.rows
        val xd = x.value.data
        val wd = w.value.data
        val y = Mat(e, x.cols)
        for (i in 0 until e) for (h in 0 until heads) {
            val wh = wd[i * heads + h]
            for (k in 0 until c) y.data[i * x.cols + h * c + k] = wh * xd[i * x.cols + h * c + k]
        }
        return record(y, arrayOf(x, w)) { out ->
            val g = out.grad!!.data
            if (x.requiresGrad) {
                val dx = x.gradBuf().data
                for (i in 0 until e) for (h in 0 until heads) {
                    val wh = wd[i * heads + h]
                    for (k in 0 until c) dx[i * x.cols + h * c + k] += wh * g[i * x.cols + h * c + k]
                }
            }
            if (w.requiresGrad) {
                val dw = w.gradBuf().data
                for (i in 0 until e) for (h in 0 until heads) {
                    var s = 0.0
                    for (k in 0 until c) s += g[i * x.cols + h * c + k] * xd[i * x.cols + h * c + k]
                    dw[i * heads + h] += s
                }
            }
        }
    }

    /** Mean over heads: x [N, H·C] → [N, C]. */
    fun headMean(x: Var, heads: Int): Var {
        val c = x.cols / heads
        val n = x.rows
        val y = Mat(n, c)
        for (i in 0 until n) for (h in 0 until heads) for (k in 0 until c) {
            y.data[i * c + k] += x.value.data[i * x.cols + h * c + k] / heads
        }
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (i in 0 until n) for (h in 0 until heads) for (k in 0 until c) {
                dx[i * x.cols + h * c + k] += g[i * c + k] / heads
            }
        }
    }

    /** x [N, K] + b [1, K] broadcast over rows. */
    fun addBias(x: Var, b: Var): Var {
        require(b.rows == 1 && b.cols == x.cols)
        val y = x.value.copy()
        for (i in 0 until x.rows) for (k in 0 until x.cols) y.data[i * x.cols + k] += b.value.data[k]
        return record(y, arrayOf(x, b)) { out ->
            val g = out.grad!!.data
            if (x.requiresGrad) {
                val dx = x.gradBuf().data
                for (j in g.indices) dx[j] += g[j]
            }
            if (b.requiresGrad) {
                val db = b.gradBuf().data
                for (i in 0 until x.rows) for (k in 0 until x.cols) db[k] += g[i * x.cols + k]
            }
        }
    }

    // ── Elementwise ───────────────────────────────────────────────────

    fun add(a: Var, b: Var): Var {
        require(a.rows == b.rows && a.cols == b.cols)
        val y = Mat(a.rows, a.cols, DoubleArray(a.value.data.size) { a.value.data[it] + b.value.data[it] })
        return record(y, arrayOf(a, b)) { out ->
            val g = out.grad!!.data
            if (a.requiresGrad) a.gradBuf().data.let { d -> for (j in g.indices) d[j] += g[j] }
            if (b.requiresGrad) b.gradBuf().data.let { d -> for (j in g.indices) d[j] += g[j] }
        }
    }

    fun scale(x: Var, s: Double): Var {
        val y = Mat(x.rows, x.cols, DoubleArray(x.value.data.size) { x.value.data[it] * s })
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (j in g.indices) dx[j] += g[j] * s
        }
    }

    /** Elementwise product with a constant of the same shape. */
    fun mulConst(x: Var, c: DoubleArray): Var {
        require(c.size == x.value.data.size)
        val y = Mat(x.rows, x.cols, DoubleArray(c.size) { x.value.data[it] * c[it] })
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (j in g.indices) dx[j] += g[j] * c[j]
        }
    }

    fun leakyRelu(x: Var, slope: Double): Var {
        val xd = x.value.data
        val y = Mat(x.rows, x.cols, DoubleArray(xd.size) { if (xd[it] > 0) xd[it] else slope * xd[it] })
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (j in g.indices) dx[j] += g[j] * (if (xd[j] > 0) 1.0 else slope)
        }
    }

    /** ELU with α = 1 (torch default). */
    fun elu(x: Var): Var {
        val xd = x.value.data
        val y = Mat(x.rows, x.cols, DoubleArray(xd.size) { if (xd[it] > 0) xd[it] else exp(xd[it]) - 1.0 })
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (j in g.indices) dx[j] += g[j] * (if (xd[j] > 0) 1.0 else exp(xd[j]))
        }
    }

    /** Inverted dropout (torch semantics): zero with prob p, scale survivors by 1/(1−p). */
    fun dropout(x: Var, p: Double, rng: Random): Var {
        if (p <= 0.0) return x
        val keep = 1.0 / (1.0 - p)
        val mask = DoubleArray(x.value.data.size) { if (rng.nextDouble() < p) 0.0 else keep }
        return mulConst(x, mask)
    }

    // ── Fused ops (memory) ────────────────────────────────────────────
    // The composed forms materialize [E, H·C] / [E, D] matrices (E ≈ edges + nodes, up to
    // ~6k; H·C = 1024 in production) — ~200 MB of Doubles per epoch, too much for an app
    // heap. These compute the same values straight from the node matrices. Each is checked
    // against its composed equivalent and by finite differences (CorticalTapeTest).

    /**
     * GAT message passing + bias in one op — equals
     * addBias(scatterAddRows(headScaleRows(gatherRows(h, src), alpha), dst, n) [→ headMean], bias):
     * out[i] = bias + Σ_{e: dst[e]=i} alpha[e, h] · h[src[e], h·C..], concatenated over heads,
     * or averaged over heads when [mean]. h [N, H·C], alpha [E, H], bias [1, H·C or C].
     */
    fun attentionAggregate(h: Var, alpha: Var, src: IntArray, dst: IntArray, n: Int, heads: Int, mean: Boolean, bias: Var): Var {
        val hc = h.cols
        val c = hc / heads
        val outCols = if (mean) c else hc
        require(alpha.rows == src.size && alpha.cols == heads && bias.cols == outCols)
        val hd = h.value.data
        val ad = alpha.value.data
        val y = Mat(n, outCols)
        val yd = y.data
        val scale = if (mean) 1.0 / heads else 1.0
        for (e in src.indices) {
            val s = src[e] * hc
            val d = dst[e] * outCols
            for (hh in 0 until heads) {
                val a = ad[e * heads + hh] * scale
                val off = if (mean) 0 else hh * c
                for (k in 0 until c) yd[d + off + k] += a * hd[s + hh * c + k]
            }
        }
        for (i in 0 until n) for (k in 0 until outCols) yd[i * outCols + k] += bias.value.data[k]
        return record(y, arrayOf(h, alpha, bias)) { out ->
            val g = out.grad!!.data
            val dh = if (h.requiresGrad) h.gradBuf().data else null
            val da = if (alpha.requiresGrad) alpha.gradBuf().data else null
            for (e in src.indices) {
                val s = src[e] * hc
                val d = dst[e] * outCols
                for (hh in 0 until heads) {
                    val off = if (mean) 0 else hh * c
                    val a = ad[e * heads + hh] * scale
                    var dot = 0.0
                    for (k in 0 until c) {
                        val gk = g[d + off + k]
                        dot += gk * hd[s + hh * c + k]
                        if (dh != null) dh[s + hh * c + k] += a * gk
                    }
                    if (da != null) da[e * heads + hh] += dot * scale
                }
            }
            if (bias.requiresGrad) {
                val db = bias.gradBuf().data
                for (i in 0 until n) for (k in 0 until outCols) db[k] += g[i * outCols + k]
            }
        }
    }

    /** elu(x) followed by inverted dropout (p, training) — mask kept as bytes, not Doubles. */
    fun eluDropout(x: Var, p: Double, rng: Random, training: Boolean): Var {
        val xd = x.value.data
        val drop = training && p > 0.0
        val keep = if (drop) 1.0 / (1.0 - p) else 1.0
        val mask = if (drop) BooleanArray(xd.size) { rng.nextDouble() >= p } else null
        val y = Mat(x.rows, x.cols, DoubleArray(xd.size) {
            val v = if (xd[it] > 0) xd[it] else exp(xd[it]) - 1.0
            if (mask == null) v else if (mask[it]) v * keep else 0.0
        })
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data
            val dx = x.gradBuf().data
            for (j in g.indices) {
                if (mask != null && !mask[j]) continue
                dx[j] += g[j] * keep * (if (xd[j] > 0) 1.0 else exp(xd[j]))
            }
        }
    }

    /** Per-pair dot of rows of z: out[e] = z[a[e]] · z[b[e]] → [E, 1]. */
    fun pairDot(z: Var, a: IntArray, b: IntArray): Var {
        require(a.size == b.size)
        val k = z.cols
        val zd = z.value.data
        val y = Mat(a.size, 1)
        for (e in a.indices) {
            var s = 0.0
            val ai = a[e] * k
            val bi = b[e] * k
            for (j in 0 until k) s += zd[ai + j] * zd[bi + j]
            y.data[e] = s
        }
        return record(y, arrayOf(z)) { out ->
            val g = out.grad!!.data
            val dz = z.gradBuf().data
            for (e in a.indices) {
                val ai = a[e] * k
                val bi = b[e] * k
                for (j in 0 until k) {
                    dz[ai + j] += g[e] * zd[bi + j]
                    dz[bi + j] += g[e] * zd[ai + j]
                }
            }
        }
    }

    /** Per-pair cosine of rows of z (torch F.cosine_similarity, eps 1e-8) → [E, 1]. */
    fun pairCosine(z: Var, a: IntArray, b: IntArray): Var {
        require(a.size == b.size)
        val k = z.cols
        val zd = z.value.data
        val norm = DoubleArray(z.rows) { i ->
            var s = 0.0
            for (j in 0 until k) s += zd[i * k + j] * zd[i * k + j]
            max(sqrt(s), COS_EPS)
        }
        val y = Mat(a.size, 1)
        for (e in a.indices) {
            var s = 0.0
            for (j in 0 until k) s += zd[a[e] * k + j] * zd[b[e] * k + j]
            y.data[e] = s / (norm[a[e]] * norm[b[e]])
        }
        return record(y, arrayOf(z)) { out ->
            val g = out.grad!!.data
            val dz = z.gradBuf().data
            for (e in a.indices) {
                val ai = a[e] * k
                val bi = b[e] * k
                val na = norm[a[e]]
                val nb = norm[b[e]]
                val cos = y.data[e]
                for (j in 0 until k) {
                    dz[ai + j] += g[e] * (zd[bi + j] / (na * nb) - cos * zd[ai + j] / (na * na))
                    dz[bi + j] += g[e] * (zd[ai + j] / (na * nb) - cos * zd[bi + j] / (nb * nb))
                }
            }
        }
    }

    // ── Rows / reductions / losses ────────────────────────────────────

    /** Row-wise dot product: a, b [E, K] → [E, 1]. */
    fun rowDot(a: Var, b: Var): Var {
        require(a.rows == b.rows && a.cols == b.cols)
        val k = a.cols
        val ad = a.value.data
        val bd = b.value.data
        val y = Mat(a.rows, 1)
        for (i in 0 until a.rows) {
            var s = 0.0
            for (j in 0 until k) s += ad[i * k + j] * bd[i * k + j]
            y.data[i] = s
        }
        return record(y, arrayOf(a, b)) { out ->
            val g = out.grad!!.data
            if (a.requiresGrad) a.gradBuf().data.let { d ->
                for (i in 0 until a.rows) for (j in 0 until k) d[i * k + j] += g[i] * bd[i * k + j]
            }
            if (b.requiresGrad) b.gradBuf().data.let { d ->
                for (i in 0 until a.rows) for (j in 0 until k) d[i * k + j] += g[i] * ad[i * k + j]
            }
        }
    }

    /** Row-wise cosine similarity (torch F.cosine_similarity, dim=1, eps=1e-8) → [E, 1]. */
    fun rowCosine(a: Var, b: Var): Var {
        require(a.rows == b.rows && a.cols == b.cols)
        val k = a.cols
        val ad = a.value.data
        val bd = b.value.data
        val na = DoubleArray(a.rows)
        val nb = DoubleArray(a.rows)
        val y = Mat(a.rows, 1)
        for (i in 0 until a.rows) {
            var aa = 0.0
            var bb = 0.0
            var ab = 0.0
            for (j in 0 until k) {
                aa += ad[i * k + j] * ad[i * k + j]
                bb += bd[i * k + j] * bd[i * k + j]
                ab += ad[i * k + j] * bd[i * k + j]
            }
            na[i] = max(sqrt(aa), COS_EPS)
            nb[i] = max(sqrt(bb), COS_EPS)
            y.data[i] = ab / (na[i] * nb[i])
        }
        return record(y, arrayOf(a, b)) { out ->
            val g = out.grad!!.data
            // d cos / d a = b/(|a||b|) − cos · a/|a|²   (and symmetrically for b)
            if (a.requiresGrad) a.gradBuf().data.let { d ->
                for (i in 0 until a.rows) {
                    val cos = y.data[i]
                    for (j in 0 until k) {
                        d[i * k + j] += g[i] * (bd[i * k + j] / (na[i] * nb[i]) - cos * ad[i * k + j] / (na[i] * na[i]))
                    }
                }
            }
            if (b.requiresGrad) b.gradBuf().data.let { d ->
                for (i in 0 until a.rows) {
                    val cos = y.data[i]
                    for (j in 0 until k) {
                        d[i * k + j] += g[i] * (ad[i * k + j] / (na[i] * nb[i]) - cos * bd[i * k + j] / (nb[i] * nb[i]))
                    }
                }
            }
        }
    }

    /** Elementwise binary cross-entropy with logits against a constant target (reduction none). */
    fun bceWithLogits(s: Var, target: Double): Var {
        val sd = s.value.data
        val y = Mat(s.rows, s.cols, DoubleArray(sd.size) {
            val v = sd[it]
            max(v, 0.0) - v * target + ln(1.0 + exp(-abs(v)))
        })
        return record(y, arrayOf(s)) { out ->
            val g = out.grad!!.data
            val ds = s.gradBuf().data
            for (j in g.indices) ds[j] += g[j] * (sigmoid(sd[j]) - target)
        }
    }

    fun mean(x: Var): Var {
        val n = x.value.data.size
        val y = Mat(1, 1, doubleArrayOf(x.value.data.sum() / n))
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data[0] / n
            val dx = x.gradBuf().data
            for (j in dx.indices) dx[j] += g
        }
    }

    /** mean((x − target)²) against a constant target (torch F.mse_loss, reduction mean). */
    fun mseConst(x: Var, target: DoubleArray): Var {
        require(target.size == x.value.data.size)
        val xd = x.value.data
        val n = xd.size
        var s = 0.0
        for (j in 0 until n) s += (xd[j] - target[j]) * (xd[j] - target[j])
        val y = Mat(1, 1, doubleArrayOf(s / n))
        return record(y, arrayOf(x)) { out ->
            val g = out.grad!!.data[0]
            val dx = x.gradBuf().data
            for (j in 0 until n) dx[j] += g * 2.0 * (xd[j] - target[j]) / n
        }
    }

    companion object {
        const val COS_EPS = 1e-8

        fun sigmoid(v: Double): Double = if (v >= 0) 1.0 / (1.0 + exp(-v)) else exp(v).let { it / (1.0 + it) }

        /** Rows below this run serially — thread fan-out costs more than it saves. */
        private const val PARALLEL_MIN_ROWS = 64

        private inline fun parallelRows(n: Int, crossinline body: (Int) -> Unit) {
            if (n < PARALLEL_MIN_ROWS) {
                for (i in 0 until n) body(i)
            } else {
                IntStream.range(0, n).parallel().forEach { body(it) }
            }
        }
    }
}

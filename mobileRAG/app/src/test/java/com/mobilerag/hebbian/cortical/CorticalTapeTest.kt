package com.mobilerag.hebbian.cortical

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.max

/**
 * Finite-difference gradient checks, one op at a time. Each case builds a scalar loss
 * L = Σ r ⊙ op(inputs) with a fixed random r, compares tape gradients to central
 * differences. Independent of PyTorch — catches backward bugs op by op.
 */
class CorticalTapeTest {

    private val rng = Random(1)

    private fun randMat(rows: Int, cols: Int, scale: Double = 1.0) =
        Mat(rows, cols, DoubleArray(rows * cols) { (rng.nextDouble() * 2 - 1) * scale })

    /** Scalar projection so every output element influences the loss differently. */
    private fun project(tape: Tape, y: Var, r: DoubleArray): Var = tape.mean(tape.mulConst(y, r))

    private fun check(name: String, inputs: List<Mat>, op: (Tape, List<Var>) -> Var) {
        val probe = Tape().let { t -> op(t, inputs.map { t.leaf(it) }) }
        val r = DoubleArray(probe.value.data.size) { rng.nextDouble() * 2 - 1 }

        fun lossOf(): Double = Tape().let { t -> project(t, op(t, inputs.map { t.leaf(it) }), r).value.data[0] }

        val tape = Tape()
        val vars = inputs.map { tape.leaf(it, requiresGrad = true) }
        tape.backward(project(tape, op(tape, vars), r))

        val h = 1e-6
        for ((k, m) in inputs.withIndex()) {
            val grad = vars[k].grad?.data ?: DoubleArray(m.data.size)
            for (i in m.data.indices) {
                val orig = m.data[i]
                m.data[i] = orig + h
                val up = lossOf()
                m.data[i] = orig - h
                val down = lossOf()
                m.data[i] = orig
                val fd = (up - down) / (2 * h)
                val tol = 1e-6 * max(1.0, abs(fd))
                assertEquals("$name: input $k [$i]", fd, grad[i], tol)
            }
        }
    }

    @Test fun linear() = check("linear", listOf(randMat(5, 4), randMat(3, 4))) { t, v -> t.linear(v[0], v[1]) }

    @Test fun linearParallelPath() =
        check("linear-parallel", listOf(randMat(70, 3), randMat(2, 3))) { t, v -> t.linear(v[0], v[1]) }

    @Test fun headDot() = check("headDot", listOf(randMat(4, 6), randMat(1, 6))) { t, v -> t.headDot(v[0], v[1], 3) }

    @Test fun gatherRows() = check("gather", listOf(randMat(4, 3))) { t, v -> t.gatherRows(v[0], intArrayOf(2, 0, 2, 3, 1)) }

    @Test fun scatterAddRows() =
        check("scatter", listOf(randMat(5, 3))) { t, v -> t.scatterAddRows(v[0], intArrayOf(1, 0, 1, 3, 1), 4) }

    @Test fun segmentSoftmax() =
        check("segSoftmax", listOf(randMat(6, 2, 3.0))) { t, v -> t.segmentSoftmax(v[0], intArrayOf(0, 1, 0, 2, 1, 0), 3) }

    @Test fun headScaleRows() =
        check("headScale", listOf(randMat(4, 6), randMat(4, 2))) { t, v -> t.headScaleRows(v[0], v[1], 2) }

    @Test fun headMean() = check("headMean", listOf(randMat(3, 8))) { t, v -> t.headMean(v[0], 4) }

    @Test fun addBias() = check("addBias", listOf(randMat(4, 3), randMat(1, 3))) { t, v -> t.addBias(v[0], v[1]) }

    @Test fun leakyRelu() = check("leaky", listOf(randMat(4, 3))) { t, v -> t.leakyRelu(v[0], 0.2) }

    @Test fun elu() = check("elu", listOf(randMat(4, 3, 2.0))) { t, v -> t.elu(v[0]) }

    @Test fun rowDot() = check("rowDot", listOf(randMat(4, 3), randMat(4, 3))) { t, v -> t.rowDot(v[0], v[1]) }

    @Test fun rowCosine() = check("rowCos", listOf(randMat(4, 3), randMat(4, 3))) { t, v -> t.rowCosine(v[0], v[1]) }

    @Test fun bceWithLogits() {
        check("bce1", listOf(randMat(5, 1, 4.0))) { t, v -> t.bceWithLogits(v[0], 1.0) }
        check("bce0", listOf(randMat(5, 1, 4.0))) { t, v -> t.bceWithLogits(v[0], 0.0) }
    }

    @Test fun mseConst() {
        val target = DoubleArray(6) { rng.nextDouble() }
        check("mse", listOf(randMat(2, 3))) { t, v -> t.mseConst(v[0], target) }
    }

    // ── Fused ops: finite differences + equality with the composed ops they replace ──

    private val eSrc = intArrayOf(0, 1, 2, 3, 1, 0, 1, 2, 3)
    private val eDst = intArrayOf(1, 2, 3, 0, 0, 0, 1, 2, 3) // includes self loops, like GAT

    @Test fun attentionAggregateConcat() =
        check("attnAgg", listOf(randMat(4, 6), randMat(9, 2), randMat(1, 6))) { t, v ->
            t.attentionAggregate(v[0], v[1], eSrc, eDst, 4, heads = 2, mean = false, bias = v[2])
        }

    @Test fun attentionAggregateMean() =
        check("attnAggMean", listOf(randMat(4, 6), randMat(9, 3), randMat(1, 2))) { t, v ->
            t.attentionAggregate(v[0], v[1], eSrc, eDst, 4, heads = 3, mean = true, bias = v[2])
        }

    @Test fun eluDropoutNoDropoutIsElu() =
        check("eluDrop", listOf(randMat(4, 3, 2.0))) { t, v -> t.eluDropout(v[0], 0.5, Random(9), training = false) }

    @Test fun eluDropoutTraining() =
        // Same seed on every evaluation → same mask, so finite differences are well defined.
        check("eluDropTrain", listOf(randMat(4, 3, 2.0))) { t, v -> t.eluDropout(v[0], 0.4, Random(9), training = true) }

    @Test fun pairDot() = check("pairDot", listOf(randMat(4, 3))) { t, v -> t.pairDot(v[0], eSrc, eDst) }

    @Test fun pairCosine() = check("pairCos", listOf(randMat(4, 3))) { t, v -> t.pairCosine(v[0], eSrc, eDst) }

    /** Fused result and gradients must equal the composed form they replace. */
    private fun sameAsComposed(name: String, inputs: List<Mat>, fused: (Tape, List<Var>) -> Var, composed: (Tape, List<Var>) -> Var) {
        fun run(op: (Tape, List<Var>) -> Var): Pair<DoubleArray, List<DoubleArray>> {
            val t = Tape()
            val vars = inputs.map { t.leaf(it.copy(), requiresGrad = true) }
            val y = op(t, vars)
            val r = DoubleArray(y.value.data.size) { (it % 7) * 0.3 - 1.0 }
            t.backward(t.mean(t.mulConst(y, r)))
            return y.value.data.copyOf() to vars.map { it.grad!!.data.copyOf() }
        }
        val (yf, gf) = run(fused)
        val (yc, gc) = run(composed)
        for (i in yf.indices) assertEquals("$name value[$i]", yc[i], yf[i], 1e-12)
        for (k in gf.indices) for (i in gf[k].indices) assertEquals("$name grad $k[$i]", gc[k][i], gf[k][i], 1e-12)
    }

    @Test fun attentionAggregateEqualsComposed() {
        val h = randMat(4, 6)
        val alpha = randMat(9, 2)
        val bias = randMat(1, 6)
        sameAsComposed("concat", listOf(h, alpha, bias),
            { t, v -> t.attentionAggregate(v[0], v[1], eSrc, eDst, 4, 2, mean = false, bias = v[2]) },
            { t, v -> t.addBias(t.scatterAddRows(t.headScaleRows(t.gatherRows(v[0], eSrc), v[1], 2), eDst, 4), v[2]) })
        val biasC = randMat(1, 3)
        sameAsComposed("mean", listOf(h, alpha, biasC),
            { t, v -> t.attentionAggregate(v[0], v[1], eSrc, eDst, 4, 2, mean = true, bias = v[2]) },
            { t, v -> t.addBias(t.headMean(t.scatterAddRows(t.headScaleRows(t.gatherRows(v[0], eSrc), v[1], 2), eDst, 4), 2), v[2]) })
    }

    @Test fun pairOpsEqualComposed() {
        val z = randMat(4, 5)
        sameAsComposed("pairDot", listOf(z),
            { t, v -> t.pairDot(v[0], eSrc, eDst) },
            { t, v -> t.rowDot(t.gatherRows(v[0], eSrc), t.gatherRows(v[0], eDst)) })
        sameAsComposed("pairCos", listOf(z),
            { t, v -> t.pairCosine(v[0], eSrc, eDst) },
            { t, v -> t.rowCosine(t.gatherRows(v[0], eSrc), t.gatherRows(v[0], eDst)) })
    }

    @Test fun composedReuse() =
        // A value used twice must accumulate both gradient paths.
        check("reuse", listOf(randMat(3, 3))) { t, v -> t.add(t.elu(v[0]), t.scale(v[0], 0.5)) }
}

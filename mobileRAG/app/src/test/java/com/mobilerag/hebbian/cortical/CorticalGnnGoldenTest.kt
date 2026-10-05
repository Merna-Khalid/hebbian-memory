package com.mobilerag.hebbian.cortical

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Random
import kotlin.math.abs
import kotlin.math.max

/**
 * Parity with the real PyTorch/PyG CorticalGNN: docs/cortical_golden.py runs
 * core/cortical_gnn.py's production consolidate() and records inputs, initial weights,
 * the negatives it drew, step-1 forward/gradients, per-epoch losses and final state.
 * PyTorch computes in float32 and the port in Double, hence tolerances rather than
 * equality: tight for one step, looser after 30 compounding Adam steps.
 */
class CorticalGnnGoldenTest {

    private val golden: JSONObject by lazy {
        val candidates = listOf("../docs/cortical-golden-values.json", "docs/cortical-golden-values.json")
        val file = candidates.map(::File).firstOrNull { it.exists() }
            ?: error("cortical-golden-values.json not found (run docs/cortical_golden.py)")
        JSONObject(file.readText())
    }

    private fun cases(): List<JSONObject> = golden.getJSONArray("cases").let { a -> List(a.length()) { a.getJSONObject(it) } }

    private fun doubles(a: JSONArray) = DoubleArray(a.length()) { a.getDouble(it) }
    private fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
    private fun matrix(a: JSONArray): Mat {
        val rows = List(a.length()) { doubles(a.getJSONArray(it)) }
        return Mat(rows.size, rows[0].size, rows.flatMap { it.asIterable() }.toDoubleArray())
    }

    private fun build(case: JSONObject): CorticalGnn {
        val c = case.getJSONObject("config")
        val config = CorticalConfig(
            inDim = c.getInt("in_dim"), hidden = c.getInt("hidden"), heads = c.getInt("heads"),
            outDim = c.getInt("out_dim"), dropout = c.getDouble("dropout"),
            lambdaCausal = c.getDouble("lambda_causal"), lambdaConsist = c.getDouble("lambda_consist"),
            epochs = c.getInt("epochs"), lr = c.getDouble("lr"),
        )
        val params = CorticalParams(config)
        val init = case.getJSONObject("init_params")
        for ((name, mat) in params.named) {
            val values = doubles(init.getJSONObject(name).getJSONArray("values"))
            assertEquals("$name size", mat.data.size, values.size)
            values.copyInto(mat.data)
        }
        return CorticalGnn(params)
    }

    private fun assertClose(what: String, expected: DoubleArray, actual: DoubleArray, absTol: Double, relTol: Double) {
        assertEquals("$what length", expected.size, actual.size)
        for (i in expected.indices) {
            val tol = max(absTol, relTol * abs(expected[i]))
            assertTrue("$what[$i]: expected ${expected[i]}, got ${actual[i]} (tol $tol)", abs(expected[i] - actual[i]) <= tol)
        }
    }

    private class Inputs(val x: Mat, val src: IntArray, val dst: IntArray, val w: DoubleArray, val zPrev: Mat?, val negatives: List<Pair<IntArray, IntArray>>)

    private fun inputs(case: JSONObject): Inputs {
        val negs = case.getJSONArray("negatives")
        return Inputs(
            x = matrix(case.getJSONArray("x")),
            src = ints(case.getJSONArray("edge_src")),
            dst = ints(case.getJSONArray("edge_dst")),
            w = doubles(case.getJSONArray("edge_weight")),
            zPrev = if (case.isNull("z_prev")) null else matrix(case.getJSONArray("z_prev")),
            negatives = List(negs.length()) { e ->
                val pair = negs.getJSONArray(e)
                ints(pair.getJSONArray(0)) to ints(pair.getJSONArray(1))
            },
        )
    }

    @Test
    fun forwardAndStepOneGradientsMatchPyTorch() {
        for (case in cases()) {
            val name = case.getString("name")
            val gnn = build(case)
            val inp = inputs(case)
            val (neg0Src, neg0Dst) = inp.negatives[0]
            val (z0, parts, grads) = gnn.lossAndGradients(inp.x, inp.src, inp.dst, inp.w, inp.zPrev, neg0Src, neg0Dst)

            assertClose("$name z0", matrix(case.getJSONArray("z0")).data, z0.data, 1e-6, 1e-5)
            val loss0 = case.getJSONObject("loss0")
            assertEquals("$name loss_total", loss0.getDouble("loss_total"), parts.total, 1e-5)
            assertEquals("$name loss_recon", loss0.getDouble("loss_recon"), parts.recon, 1e-5)
            assertEquals("$name loss_causal", loss0.getDouble("loss_causal"), parts.causal, 1e-5)
            assertEquals("$name loss_consist", loss0.getDouble("loss_consist"), parts.consist, 1e-5)

            val expectedGrads = case.getJSONObject("grads0")
            for (pname in gnn.params.named.keys) {
                assertClose("$name grad $pname", doubles(expectedGrads.getJSONArray(pname)), grads.getValue(pname).data, 1e-6, 1e-4)
            }
        }
    }

    @Test
    fun thirtyEpochConsolidationMatchesPyTorch() {
        for (case in cases()) {
            val name = case.getString("name")
            val gnn = build(case)
            val inp = inputs(case)
            val replay = NegativeSampler { epoch, _, _, _, _ -> inp.negatives[epoch] }
            val (zFinal, history) = gnn.consolidate(inp.x, inp.src, inp.dst, inp.w, inp.zPrev, replay, Random(0))

            val losses = case.getJSONArray("losses")
            assertEquals("$name epochs", losses.length(), history.size)
            for (e in history.indices) {
                val l = losses.getJSONObject(e)
                assertEquals("$name epoch $e total", l.getDouble("loss_total"), history[e].total, 1e-4)
                assertEquals("$name epoch $e recon", l.getDouble("loss_recon"), history[e].recon, 1e-4)
                assertEquals("$name epoch $e causal", l.getDouble("loss_causal"), history[e].causal, 1e-4)
                assertEquals("$name epoch $e consist", l.getDouble("loss_consist"), history[e].consist, 1e-4)
            }
            assertClose("$name z_final", matrix(case.getJSONArray("z_final")).data, zFinal.data, 1e-4, 1e-3)
            val finalParams = case.getJSONObject("final_params")
            for ((pname, mat) in gnn.params.named) {
                assertClose("$name final $pname", doubles(finalParams.getJSONObject(pname).getJSONArray("values")), mat.data, 1e-4, 1e-3)
            }
        }
    }

    @Test
    fun weightsRoundTripThroughSerialization() {
        val gnn = build(cases()[1])
        val bytes = java.io.ByteArrayOutputStream().also { gnn.params.writeTo(java.io.DataOutputStream(it)) }.toByteArray()
        val back = CorticalParams.readFrom(java.io.DataInputStream(bytes.inputStream()), gnn.params.config)!!
        for ((pname, mat) in gnn.params.named) assertClose("roundtrip $pname", mat.data, back.named.getValue(pname).data, 0.0, 0.0)
        val other = gnn.params.config.copy(inDim = gnn.params.config.inDim + 1)
        assertEquals(null, CorticalParams.readFrom(java.io.DataInputStream(bytes.inputStream()), other))
    }
}

package com.mobilerag.spikes

import android.content.Context
import android.os.Debug
import com.mobilerag.hebbian.cortical.CorticalConfig
import com.mobilerag.hebbian.cortical.CorticalGnn
import com.mobilerag.hebbian.cortical.CorticalParams
import com.mobilerag.hebbian.cortical.Mat
import com.mobilerag.hebbian.cortical.RejectionNegativeSampler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Random

/**
 * Cost of one cortical consolidation at production size (768-d, 4 heads × 256 hidden,
 * ~1.6M params) on synthetic graphs of 300 and 1000 nodes (10 edges/node, capped 5000).
 * Runs 2 epochs each and extrapolates to a 30-epoch run. Correctness is covered on the dev
 * machine by CorticalGnnGoldenTest (PyTorch golden values); this measures the device.
 * Touches no store.
 */
class CorticalConsolidationSpike : Spike {
    override val id = "cortical_consolidation"
    override val title = "Cortical consolidation cost"
    override val description = "Production-size cortical GAT on synthetic 300/1000-node graphs: ms per epoch, extrapolated seconds per 30-epoch run, Java heap."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult = withContext(Dispatchers.Default) {
        try {
            val metrics = linkedMapOf<String, String>()
            val rt = Runtime.getRuntime()
            log("max heap: ${rt.maxMemory() / (1 shl 20)} MB")
            for (n in listOf(300, 1000)) {
                val rng = Random(7)
                val edges = minOf(n * 10, 5000)
                val cfg = CorticalConfig.production(768)
                val gnn = CorticalGnn(CorticalParams.glorot(cfg, rng))
                val x = Mat(n, 768, DoubleArray(n * 768) { rng.nextGaussian() / 28.0 })
                val src = IntArray(edges) { rng.nextInt(n) }
                val dst = IntArray(edges) { (src[it] + 1 + rng.nextInt(n - 1)) % n }
                val w = DoubleArray(edges) { 0.5 + 4.5 * rng.nextDouble() }
                System.gc()
                val before = rt.totalMemory() - rt.freeMemory()
                var peak = before
                val t0 = System.nanoTime()
                gnn.consolidate(x, src, dst, w, null, RejectionNegativeSampler(rng), rng, epochs = 2) { e, parts ->
                    peak = maxOf(peak, rt.totalMemory() - rt.freeMemory())
                    log("N=$n epoch $e loss=%.4f".format(parts.total))
                }
                val msPerEpoch = (System.nanoTime() - t0) / 1e6 / 2
                val line = "%.0f ms/epoch → ≈ %.0f s per run, heap Δ≈%d MB"
                    .format(msPerEpoch, msPerEpoch * 30 / 1000, (peak - before) / (1 shl 20))
                log("N=$n E=$edges: $line")
                metrics["N=$n"] = line
            }
            metrics["native heap"] = "${Debug.getNativeHeapAllocatedSize() / (1 shl 20)} MB"
            SpikeResult(passed = true, summary = "Cortical consolidation measured at production size", metrics = metrics)
        } catch (t: Throwable) {
            SpikeResult(
                passed = false,
                summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                error = t.stackTraceToString().take(2000),
            )
        }
    }
}

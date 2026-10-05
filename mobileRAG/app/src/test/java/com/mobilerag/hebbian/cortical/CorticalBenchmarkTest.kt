package com.mobilerag.hebbian.cortical

import org.junit.Assume.assumeTrue
import org.junit.Test
import java.util.Random

/**
 * Opt-in cost check at production size (CorticalConfig.production(768)):
 *   CORTICAL_BENCH=1 ./gradlew :app:testDebugUnitTest --tests '*CorticalBenchmarkTest*' -i
 * Prints ms per epoch and heap used by one epoch. A dev-machine number — the on-device
 * cortical_consolidation spike is the one that counts.
 */
class CorticalBenchmarkTest {

    @Test
    fun productionSizeEpochCost() {
        assumeTrue(System.getenv("CORTICAL_BENCH") != null)
        val rng = Random(3)
        val n = System.getenv("CORTICAL_BENCH_NODES")?.toInt() ?: 1000
        val edges = System.getenv("CORTICAL_BENCH_EDGES")?.toInt() ?: 5000
        val cfg = CorticalConfig.production(768)
        val gnn = CorticalGnn(CorticalParams.glorot(cfg, rng))
        val x = Mat(n, 768, DoubleArray(n * 768) { rng.nextGaussian() / 28.0 })
        val src = IntArray(edges) { rng.nextInt(n) }
        val dst = IntArray(edges) { (src[it] + 1 + rng.nextInt(n - 1)) % n }
        val w = DoubleArray(edges) { 0.5 + 4.5 * rng.nextDouble() }

        val rt = Runtime.getRuntime()
        System.gc()
        val before = rt.totalMemory() - rt.freeMemory()
        var peak = before
        val start = System.nanoTime()
        gnn.consolidate(x, src, dst, w, null, RejectionNegativeSampler(rng), rng, epochs = 3) { _, _ ->
            peak = maxOf(peak, rt.totalMemory() - rt.freeMemory())
        }
        val msPerEpoch = (System.nanoTime() - start) / 1e6 / 3
        println("cortical bench: N=$n E=$edges params=${gnn.params.count} " +
            "→ %.0f ms/epoch (≈ %.1f s per 30-epoch run), heap Δ≈%d MB"
                .format(msPerEpoch, msPerEpoch * 30 / 1000, (peak - before) / (1 shl 20)))
    }
}

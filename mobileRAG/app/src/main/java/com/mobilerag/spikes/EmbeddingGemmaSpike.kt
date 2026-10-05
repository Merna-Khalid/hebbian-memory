package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.embeddings.EmbeddingGemmaEngine

/** Phase 1 spike: EmbeddingGemma (ONNX) embedding quality + latency on-device. */
class EmbeddingGemmaSpike : Spike {
    override val id = "embeddinggemma"
    override val title = "EmbeddingGemma embeddings"
    override val description = "EmbeddingGemma-300M ONNX (q4f16) via onnxruntime; retrieval-style cosine sanity check."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val instr = Instrumentation(context, log)

        return try {
            instr.mark("model load")
            val engine = EmbeddingGemmaEngine.create(context)
            val loadMs = instr.elapsed("model load")
            log("Engine: ${engine.id}, dims=${engine.dimensions}")

            val query = "when is the tax filing deadline"
            val docs = listOf(
                "The deadline to file federal income tax returns is April 15. Extensions move it to October 15." to "tax notes",
                "Priya approved the revised headcount plan after the Q3 budget meeting." to "meeting notes",
                "Sourdough starter should be fed twice daily with equal parts flour and water." to "bread recipe",
            )

            instr.mark("embeddings x${docs.size + 1}")
            val qEmb = engine.embedQuery(query)
            val dEmbs = docs.map { (text, title) -> engine.embedDocument(text, title) }
            val embMs = instr.elapsed("embeddings x${docs.size + 1}")

            val sims = dEmbs.map { EmbeddingGemmaEngine.cosine(qEmb, it) }
            docs.forEachIndexed { i, (_, title) -> log("sim(query, $title) = %.4f".format(sims[i])) }

            engine.close()

            val bestMatch = sims.indices.maxBy { sims[it] }
            val passed = bestMatch == 0
            SpikeResult(
                passed = passed,
                summary = if (passed) "EmbeddingGemma ranks the tax doc first for a tax query" else
                    "wrong top match: ${docs[bestMatch].second}",
                metrics = mapOf(
                    "model load" to "$loadMs ms",
                    "avg embed" to "${embMs / (docs.size + 1)} ms",
                    "sims" to sims.joinToString(",") { "%.3f".format(it) },
                ),
            )
        } catch (t: Throwable) {
            SpikeResult(
                passed = false,
                summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                error = t.stackTraceToString().take(2000),
            )
        }
    }
}

package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.embeddings.MiniLmEmbeddingEngine

/** Spike A: ONNX Runtime embedding inference on-device. */
class OnnxEmbeddingSpike : Spike {
    override val id = "onnx_embedding"
    override val title = "ONNX Runtime embeddings"
    override val description = "all-MiniLM-L6-v2 (int8 arm64) via onnxruntime-android; sanity-check cosine ordering."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val instr = Instrumentation(context, log)

        instr.mark("model load")
        val engine = MiniLmEmbeddingEngine(context)
        val loadMs = instr.elapsed("model load")

        val sentences = listOf(
            "the cat sat on the mat",
            "a dog is a friendly animal",
            "how do I file my tax return",
            "kittens are young cats",
        )

        instr.mark("embedding x${sentences.size}")
        val embs = sentences.map { engine.embed(it) }
        val embMs = instr.elapsed("embedding x${sentences.size}")

        val simCatKitten = MiniLmEmbeddingEngine.cosine(embs[0], embs[3])
        val simCatTax = MiniLmEmbeddingEngine.cosine(embs[0], embs[2])
        val simCatDog = MiniLmEmbeddingEngine.cosine(embs[0], embs[1])
        log("sim(cat, kittens) = %.4f".format(simCatKitten))
        log("sim(cat, dog)     = %.4f".format(simCatDog))
        log("sim(cat, taxes)   = %.4f".format(simCatTax))

        engine.close()

        val orderingSane = simCatKitten > simCatTax && simCatDog > simCatTax
        val fastEnough = embMs / sentences.size < 100
        val passed = orderingSane && fastEnough
        return SpikeResult(
            passed = passed,
            summary = if (passed) "Embeddings on-device, sane similarity ordering" else
                "orderingSane=$orderingSane fastEnough=$fastEnough",
            metrics = mapOf(
                "model load" to "$loadMs ms",
                "avg inference" to "${embMs / sentences.size} ms",
                "dims" to "${engine.dimensions}",
            ),
        )
    }
}

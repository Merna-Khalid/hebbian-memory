package com.mobilerag.spikes

import android.content.Context
import android.util.Log
import com.mobilerag.extract.Gliner2Extractor

/**
 * Phase 1 spike: GLiNER2 (gliner2-multi-v1, ONNX int8) named-entity extraction on-device.
 * Entity lists are logged to logcat under tag "GlinerSpike" for adb validation.
 */
class GlinerSpike : Spike {
    override val id = "gliner2_ner"
    override val title = "GLiNER2 NER"
    override val description = "GLiNER2 multi-v1 (int8 ONNX) span extraction; schema-prompted entity decoding sanity check."

    private data class Case(val sentence: String, val expectedAny: List<String>)

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val instr = Instrumentation(context, log)

        return try {
            instr.mark("model load")
            val extractor = Gliner2Extractor.create(context)
            val loadMs = instr.elapsed("model load")

            val cases = listOf(
                Case(
                    "The RedMagic 10 Pro runs LadybugDB and EmbeddingGemma on Android 15.",
                    listOf("redmagic", "ladybugdb", "embeddinggemma", "android"),
                ),
                Case(
                    "Marie Curie discovered radium in Paris while working at the Sorbonne.",
                    listOf("marie curie", "paris", "sorbonne", "radium"),
                ),
                Case(
                    "Apple announced the iPhone 17 in Cupertino on September 9, 2026.",
                    listOf("apple", "iphone", "cupertino", "september"),
                ),
            )

            val extractMs = mutableListOf<Long>()
            var allEnoughEntities = true
            var allExpectedHit = true
            val summaryLines = mutableListOf<String>()

            for (case in cases) {
                instr.mark("extract")
                val entities = extractor.extractEntities(case.sentence)
                extractMs += instr.elapsed("extract")

                Log.i(TAG, "sentence: ${case.sentence}")
                entities.forEach { e ->
                    Log.i(TAG, "  ${e.label}: \"${e.text}\" score=%.3f [${e.start},${e.end})".format(e.score))
                }
                val rendered = entities.joinToString("; ") { "${it.text} (${it.label}, %.2f)".format(it.score) }
                log("${case.sentence.take(40)}… -> ${entities.size} entities: $rendered")
                summaryLines += "\"${case.sentence}\" -> ${if (entities.isEmpty()) "(none)" else rendered}"

                if (entities.size < 2) allEnoughEntities = false
                val haystack = entities.joinToString(" ") { it.text }.lowercase()
                if (case.expectedAny.none { it in haystack }) allExpectedHit = false
            }

            extractor.close()

            val passed = allEnoughEntities && allExpectedHit
            SpikeResult(
                passed = passed,
                summary = (if (passed) "PASS: " else "FAIL: ") + summaryLines.joinToString("\n"),
                metrics = mapOf(
                    "model load" to "$loadMs ms",
                    "avg extract" to "${extractMs.average().toLong()} ms",
                    "pss" to "${QueryTimer.pssKb()} KB",
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

    companion object {
        private const val TAG = "GlinerSpike"
    }
}

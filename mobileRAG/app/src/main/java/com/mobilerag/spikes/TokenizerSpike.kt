package com.mobilerag.spikes

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import android.content.Context
import java.io.File

/**
 * Phase 1 de-risk spike: HuggingFace fast tokenizer (tokenizer.json, SentencePiece-backed)
 * running on-device via DJL tokenizers + tokenizer-native.
 *
 * Expects tokenizer.json pushed to the app's internal files dir:
 *   adb push tokenizer.json /data/local/tmp/ &&
 *   adb shell run-as com.mobilerag sh -c 'mkdir -p files/models && cp /data/local/tmp/tokenizer.json files/models/'
 */
class TokenizerSpike : Spike {
    override val id = "hf_tokenizer"
    override val title = "HF tokenizer (DJL)"
    override val description = "EmbeddingGemma tokenizer.json via DJL HuggingFaceTokenizer; encode/decode round-trip."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val instr = Instrumentation(context, log)

        fun findJson(root: File?): File? = root?.listFiles()
            ?.flatMap { f -> if (f.isDirectory) f.listFiles()?.toList() ?: emptyList() else listOf(f) }
            ?.firstOrNull { it.name == "tokenizer.json" && it.canRead() }
        val tokFile = findJson(context.filesDir) ?: findJson(context.getExternalFilesDir(null))
        if (tokFile == null) {
            log("No tokenizer.json under ${context.filesDir.absolutePath} or external files dir")
            log("Run: adb push tokenizer.json /data/local/tmp/ && adb shell run-as com.mobilerag sh -c 'mkdir -p files/models && cp /data/local/tmp/tokenizer.json files/models/'")
            return SpikeResult(false, "No tokenizer.json on device — see log for adb push command")
        }
        log("Tokenizer: ${tokFile.absolutePath} (${tokFile.length() / (1024 * 1024)} MB)")

        return try {
            instr.mark("tokenizer load")
            val tokenizer = HuggingFaceTokenizer.newInstance(tokFile.toPath())
            val loadMs = instr.elapsed("tokenizer load")

            val samples = listOf(
                "the cat sat on the mat",
                "task: search result | query: how do I file my tax return",
                "title: meeting notes | text: Discussed the Q3 budget with Priya; she approved the revised headcount plan.",
            )
            instr.mark("encode x${samples.size}")
            val encodings = samples.map { tokenizer.encode(it) }
            val encMs = instr.elapsed("encode x${samples.size}")

            var roundTripOk = true
            encodings.forEachIndexed { i, enc ->
                val decoded = tokenizer.decode(enc.ids)
                log("sample[$i]: ${enc.ids.size} tokens; decode round-trip ${decoded.length} chars")
                if (decoded.isBlank()) roundTripOk = false
            }
            // EmbeddingGemma chunks target 300–500 tokens; a ~50-word sentence should be far below that
            val countsSane = encodings.all { it.ids.size in 2..512 }
            tokenizer.close()

            val passed = roundTripOk && countsSane
            SpikeResult(
                passed = passed,
                summary = if (passed) "tokenizer.json loads and encodes on-device" else
                    "roundTripOk=$roundTripOk countsSane=$countsSane",
                metrics = mapOf(
                    "tokenizer load" to "$loadMs ms",
                    "avg encode" to "${encMs / samples.size} ms",
                    "sample tokens" to encodings.joinToString(",") { it.ids.size.toString() },
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

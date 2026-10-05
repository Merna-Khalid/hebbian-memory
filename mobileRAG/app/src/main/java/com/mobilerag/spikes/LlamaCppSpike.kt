package com.mobilerag.spikes

import android.content.Context
import android.os.SystemClock
import com.arm.aichat.AiChat
import com.mobilerag.generation.LlmResidency
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

/**
 * Spike C: llama.cpp on-device generation.
 *
 * Expects a GGUF model pushed to the app's external files dir, e.g.:
 *   adb push Qwen3-0.6B-Q8_0.gguf /sdcard/Android/data/com.mobilerag/files/models/
 * Recommended first model: Qwen3-0.6B Q8_0 (~700 MB), then Qwen3-1.7B Q6_K (~1.4 GB).
 */
class LlamaCppSpike : Spike {
    override val id = "llamacpp_gen"
    override val title = "llama.cpp generation"
    override val description = "GGUF via llama.cpp JNI; measures load, TTFT, decode tok/s."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val instr = Instrumentation(context, log)

        fun findGguf(root: File?): File? = root?.listFiles()
            ?.flatMap { f -> if (f.isDirectory) f.listFiles()?.toList() ?: emptyList() else listOf(f) }
            ?.firstOrNull { it.extension == "gguf" && it.canRead() }
        // SELinux blocks run-as/shell-created files on external storage; internal files dir is reliable
        val model = findGguf(context.filesDir) ?: findGguf(context.getExternalFilesDir(null))
        if (model == null) {
            log("No .gguf model under ${context.filesDir.absolutePath} or external files dir")
            log("Run: adb push <model>.gguf /data/local/tmp/ && adb shell run-as com.mobilerag cp /data/local/tmp/<model>.gguf files/models/")
            return SpikeResult(false, "No GGUF model on device — see log for adb push command")
        }
        log("Model: ${model.name} (${model.length() / (1024 * 1024)} MB)")

        val engine = AiChat.getInferenceEngine(context)
        // Hold the shared engine and start cold: the spike measures an mmap load, and must not
        // load over — or leave stale residency behind for — a model a tab is using.
        return LlmResidency.mutex.withLock {
            LlmResidency.unloadLocked(context)
            try {
                instr.mark("model load (mmap)")
                engine.loadModel(model.absolutePath)
                val loadMs = instr.elapsed("model load (mmap)")

                val prompt = "Question: What are the three main benefits of on-device retrieval-augmented generation? Answer briefly."
                engine.setSystemPrompt("You are a concise assistant. Answer in under 100 words.")

                val start = SystemClock.elapsedRealtime()
                var firstTokenMs = -1L
                var tokens = 0
                val answer = StringBuilder()
                withTimeoutOrNull(120_000) {
                    engine.sendUserPrompt(prompt, 256).collect { token ->
                        if (firstTokenMs < 0) firstTokenMs = SystemClock.elapsedRealtime() - start
                        tokens++
                        answer.append(token)
                    }
                }
                val totalMs = SystemClock.elapsedRealtime() - start
                val decodeMs = totalMs - firstTokenMs.coerceAtLeast(0)
                val tokPerSec = if (decodeMs > 0 && tokens > 1) (tokens - 1) * 1000.0 / decodeMs else 0.0
                log("Answer preview: ${answer.toString().take(200)}")

                engine.cleanUp()

                val passed = tokens > 10 && firstTokenMs in 1..60_000
                SpikeResult(
                    passed = passed,
                    summary = if (passed) "Streaming generation works on-device" else
                        "tokens=$tokens ttft=${firstTokenMs}ms",
                    metrics = mapOf(
                        "model load" to "$loadMs ms",
                        "TTFT" to "$firstTokenMs ms",
                        "decode" to "%.1f tok/s".format(tokPerSec),
                        "tokens" to "$tokens",
                    ),
                )
            } catch (t: Throwable) {
                runCatching { engine.cleanUp() }
                SpikeResult(
                    passed = false,
                    summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                    error = t.stackTraceToString().take(2000),
                )
            }
        }
    }
}

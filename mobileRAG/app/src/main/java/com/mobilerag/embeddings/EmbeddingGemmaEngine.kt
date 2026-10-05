package com.mobilerag.embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.mobilerag.core.EmbeddingEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * EmbeddingGemma-300M (ONNX, quantized) via ONNX Runtime.
 *
 * Uses the task prompts from the model card, which are required for retrieval quality:
 *   query:    "task: search result | query: {query}"
 *   document: "title: {title|none} | text: {text}"
 *
 * Expects on-device files (pushed via the run-as path, too large for the APK):
 *   files/models/embeddinggemma/tokenizer.json
 *   files/models/embeddinggemma/model_q4f16.onnx (+ model_q4f16.onnx_data)
 */
class EmbeddingGemmaEngine private constructor(
    private val tokenizer: GemmaTokenizer,
    private val session: OrtSession,
    modelFileName: String,
) : EmbeddingEngine, AutoCloseable {

    override val id = "embeddinggemma-onnx-$modelFileName"
    override val dimensions = 768

    private val env = OrtEnvironment.getEnvironment()

    override suspend fun embed(text: String): FloatArray = embedDocument(text)

    suspend fun embedQuery(query: String): FloatArray = embedRaw("$QUERY_PREFIX$query")

    suspend fun embedDocument(text: String, title: String? = null): FloatArray =
        embedRaw("title: ${title ?: "none"} | text: $text")

    fun countTokens(text: String): Int = tokenizer.countTokens(text)

    private suspend fun embedRaw(text: String): FloatArray = withContext(Dispatchers.Default) {
        val enc = tokenizer.encode(text)
        val seqLen = enc.inputIds.size
        val shape = longArrayOf(1, seqLen.toLong())

        val tensors = mutableListOf<OnnxTensor>()
        try {
            val inputs = mutableMapOf<String, OnnxTensor>()
            for (name in session.inputNames) {
                val tensor = when (name) {
                    "input_ids" -> OnnxTensor.createTensor(env, LongBuffer.wrap(enc.inputIds), shape)
                    "attention_mask" -> OnnxTensor.createTensor(env, LongBuffer.wrap(enc.attentionMask), shape)
                    "token_type_ids" -> OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seqLen)), shape)
                    "position_ids" -> OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seqLen) { it.toLong() }), shape)
                    else -> continue
                }
                tensors += tensor
                inputs[name] = tensor
            }
            session.run(inputs).use { result ->
                // Prefer a pre-pooled sentence embedding if the export has one; else mean-pool token states
                val outputNames = session.outputNames
                if ("sentence_embedding" in outputNames) {
                    val idx = outputNames.indexOf("sentence_embedding")
                    @Suppress("UNCHECKED_CAST")
                    normalize((result[idx].value as Array<FloatArray>)[0])
                } else {
                    @Suppress("UNCHECKED_CAST")
                    val tokenEmb = (result[0].value as Array<Array<FloatArray>>)[0]
                    meanPoolAndNormalize(tokenEmb, enc.attentionMask)
                }
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }

    private fun meanPoolAndNormalize(tokenEmb: Array<FloatArray>, mask: LongArray): FloatArray {
        val dims = tokenEmb[0].size
        val out = FloatArray(dims)
        var count = 0
        for (i in tokenEmb.indices) {
            if (i < mask.size && mask[i] == 1L) {
                val row = tokenEmb[i]
                for (d in 0 until dims) out[d] += row[d]
                count++
            }
        }
        if (count == 0) return out
        for (d in 0 until dims) out[d] /= count
        return normalize(out)
    }

    override fun close() {
        session.close()
        tokenizer.close()
    }

    companion object {
        const val QUERY_PREFIX = "task: search result | query: "
        const val DEFAULT_DIR = "models/embeddinggemma"
        private val MODEL_PREFERENCE = listOf("model_q4f16.onnx", "model_quantized.onnx")

        fun create(context: Context, dir: File? = null): EmbeddingGemmaEngine {
            val modelDir = dir ?: File(context.filesDir, DEFAULT_DIR)
            val tokFile = File(modelDir, "tokenizer.json")
            require(tokFile.canRead()) {
                "Missing ${tokFile.absolutePath} — push via: adb push <f> /data/local/tmp/ && " +
                    "adb shell run-as com.mobilerag sh -c 'mkdir -p files/$DEFAULT_DIR && cp /data/local/tmp/<f> files/$DEFAULT_DIR/'"
            }
            val model = MODEL_PREFERENCE.map { File(modelDir, it) }.firstOrNull { it.canRead() }
                ?: modelDir.listFiles()?.firstOrNull { it.extension == "onnx" && it.canRead() }
                ?: throw IllegalStateException("No .onnx model in ${modelDir.absolutePath} — same push path as tokenizer.json")
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
            // path-based createSession so the external .onnx_data resolves relative to the model dir
            val session = OrtEnvironment.getEnvironment().createSession(model.absolutePath, opts)
            return EmbeddingGemmaEngine(GemmaTokenizer(tokFile.toPath()), session, model.nameWithoutExtension)
        }

        private fun normalize(v: FloatArray): FloatArray {
            var norm = 0f
            for (x in v) norm += x * x
            norm = sqrt(norm)
            if (norm > 0f) for (i in v.indices) v[i] /= norm
            return v
        }

        fun cosine(a: FloatArray, b: FloatArray): Float {
            var dot = 0f
            for (i in a.indices) dot += a[i] * b[i]
            return dot // vectors are pre-normalized
        }
    }
}

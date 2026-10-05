package com.mobilerag.embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.mobilerag.core.EmbeddingEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import kotlin.math.sqrt

/** all-MiniLM-L6-v2 (int8, arm64) via ONNX Runtime. Mean-pooled, L2-normalized sentence embeddings. */
class MiniLmEmbeddingEngine(context: Context) : EmbeddingEngine {

    override val id = "minilm-onnx-int8"
    override val dimensions = 384

    private val tokenizer = WordPieceTokenizer(context)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open("model_qint8_arm64.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        }
        session = env.createSession(modelBytes, opts)
    }

    override suspend fun embed(text: String): FloatArray = withContext(Dispatchers.Default) {
        val enc = tokenizer.encode(text)
        val batch = longArrayOf(1, enc.inputIds.size.toLong())

        val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(enc.inputIds), batch)
        val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(enc.attentionMask), batch)
        val typesTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(enc.tokenTypeIds), batch)

        idsTensor.use { ids ->
            maskTensor.use { mask ->
                typesTensor.use { types ->
                    val inputs = mapOf(
                        "input_ids" to ids,
                        "attention_mask" to mask,
                        "token_type_ids" to types,
                    )
                    session.run(inputs).use { result ->
                        // [1, seq, 384] token embeddings -> mean pool over non-pad tokens
                        @Suppress("UNCHECKED_CAST")
                        val tokenEmb = (result[0].value as Array<Array<FloatArray>>)[0]
                        meanPoolAndNormalize(tokenEmb, enc.attentionMask)
                    }
                }
            }
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
        var norm = 0f
        for (d in 0 until dims) {
            out[d] /= count
            norm += out[d] * out[d]
        }
        norm = sqrt(norm)
        if (norm > 0f) for (d in 0 until dims) out[d] /= norm
        return out
    }

    fun close() = session.close()

    companion object {
        fun cosine(a: FloatArray, b: FloatArray): Float {
            var dot = 0f
            for (i in a.indices) dot += a[i] * b[i]
            return dot // vectors are pre-normalized
        }
    }
}

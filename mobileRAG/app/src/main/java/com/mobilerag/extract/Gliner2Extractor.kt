package com.mobilerag.extract

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.LongBuffer
import kotlin.math.exp

/**
 * GLiNER2 (fastino/gliner2-multi-v1, ONNX int8) named-entity extractor.
 *
 * Decoding ported from the Ruby gem `elcuervo/gliner`, which runs this exact ONNX export.
 * Schema layout in the combined (pretokenized-style) token stream:
 *   ( [P] entities ( [E] <label1> [E] <label2> ... ) ) [SEP_TEXT] <text words...>
 * Each element is encoded individually with addSpecialTokens=false (the tokenizer.json has a
 * TemplateProcessing post-processor that would otherwise inject [CLS]/[SEP] per element).
 *
 * Expects on-device files (pushed via the run-as path, too large for the APK):
 *   files/models/gliner2/tokenizer.json
 *   files/models/gliner2/model_int8.onnx
 */
class Gliner2Extractor private constructor(
    private val tokenizer: HuggingFaceTokenizer,
    private val session: OrtSession,
) : AutoCloseable {

    data class ExtractedEntity(
        val text: String,
        val label: String,
        val score: Float,
        val start: Int,
        val end: Int,
    )

    private val env = OrtEnvironment.getEnvironment()

    private val outputName: String = session.outputNames.let { names ->
        when {
            "span_logits" in names -> "span_logits"
            "logits" in names -> "logits"
            else -> throw IllegalStateException("GLiNER2 model has neither span_logits nor logits output: $names")
        }
    }

    suspend fun extractEntities(
        text: String,
        labels: List<String> = DEFAULT_LABELS,
        threshold: Float = 0.5f,
    ): List<ExtractedEntity> = withContext(Dispatchers.Default) {
        require(labels.isNotEmpty()) { "labels must not be empty" }

        val normalized = normalizeText(text)
        val words = splitWords(normalized)

        // Combined element list C: schema + [SEP_TEXT] + text words
        val combined = ArrayList<String>(4 + 2 * labels.size + 3 + words.size)
        combined.add("(")
        combined.add("[P]")
        combined.add("entities")
        combined.add("(")
        for (label in labels) {
            combined.add("[E]")
            combined.add(label)
        }
        combined.add(")")
        combined.add(")")
        combined.add("[SEP_TEXT]")
        val textStartIndex = combined.size // == S + 1; index of first text word in C
        combined.addAll(words.map { it.text })

        // Encode each element individually (no special tokens) and concatenate
        val ids = ArrayList<Long>()
        val elementStart = IntArray(combined.size) // token offset where element i starts
        for (i in combined.indices) {
            elementStart[i] = ids.size
            val enc = tokenizer.encode(combined[i], false, false)
            for (id in enc.ids) ids.add(id)
        }

        if (Log.isLoggable(TAG, Log.DEBUG)) {
            val sepTextCount = ids.count { it == SEP_TEXT_ID }
            Log.d(TAG, "encoded ${ids.size} tokens; [SEP_TEXT] (id $SEP_TEXT_ID) occurrences: $sepTextCount")
        }

        // Truncate to max_seq_len; effective text words are those fully inside the window
        val seqLen = minOf(ids.size, MAX_SEQ_LEN)
        val inputIds = LongArray(seqLen) { ids[it] }
        var effectiveTextWords = 0
        for (w in words.indices) {
            if (elementStart[textStartIndex + w] < seqLen) effectiveTextWords++ else break
        }
        if (effectiveTextWords == 0) return@withContext emptyList()

        // First subword token position of each text word, and token position of each [E] marker
        val wordStartTokenPos = IntArray(effectiveTextWords) { elementStart[textStartIndex + it] }
        val labelTokenPos = IntArray(labels.size) { elementStart[4 + 2 * it] }

        // words_mask: 1 at the first token position of each text word
        val wordsMask = LongArray(seqLen)
        for (w in 0 until effectiveTextWords) wordsMask[wordStartTokenPos[w]] = 1L

        val numLabels = labels.size
        val tensors = mutableListOf<OnnxTensor>()
        try {
            val inputs = mutableMapOf<String, OnnxTensor>()
            for (name in session.inputNames) {
                val tensor = when (name) {
                    "input_ids" -> tensor(inputIds, longArrayOf(1, seqLen.toLong()))
                    "attention_mask" -> tensor(LongArray(seqLen) { 1L }, longArrayOf(1, seqLen.toLong()))
                    "token_type_ids" -> tensor(LongArray(seqLen), longArrayOf(1, seqLen.toLong()))
                    "words_mask" -> tensor(wordsMask, longArrayOf(1, seqLen.toLong()))
                    "text_lengths" -> tensor(longArrayOf(effectiveTextWords.toLong()), longArrayOf(1))
                    "task_type" -> tensor(longArrayOf(TASK_TYPE_ENTITIES), longArrayOf(1))
                    "label_positions" -> tensor(
                        LongArray(numLabels) { labelTokenPos[it].toLong() },
                        longArrayOf(1, numLabels.toLong()),
                    )
                    "label_mask" -> tensor(LongArray(numLabels) { 1L }, longArrayOf(1, numLabels.toLong()))
                    else -> continue
                }
                tensors += tensor
                inputs[name] = tensor
            }

            session.run(inputs).use { result ->
                val outTensor = result[0] as OnnxTensor
                val shape = outTensor.info.shape // [1, L, maxWidth, Lnum]
                val maxWidth = shape[2].toInt().coerceAtMost(MAX_WIDTH)
                val lastDim = shape[3].toInt()
                // span_logits exports key the last dim by label token position; this export
                // (output_format="logits") keys it by label index directly
                val byLabelIndex = outputName == "logits" && lastDim == numLabels
                val buf = outTensor.floatBuffer

                // Collect all spans above threshold, per label
                val spansByLabel = Array(numLabels) { mutableListOf<Span>() }
                for (w in 0 until effectiveTextWords) {
                    val pos = wordStartTokenPos[w]
                    val posBase = pos * shape[2].toInt() * lastDim
                    val maxD = minOf(maxWidth, effectiveTextWords - w)
                    for (d in 0 until maxD) {
                        val rowBase = posBase + d * lastDim
                        for (i in 0 until numLabels) {
                            val idx = rowBase + if (byLabelIndex) i else labelTokenPos[i].coerceAtMost(lastDim - 1)
                            val score = sigmoid(buf.get(idx))
                            if (score >= threshold) spansByLabel[i].add(Span(w, w + d, score))
                        }
                    }
                }

                // Per label: keep highest-scoring non-overlapping spans
                val entities = mutableListOf<ExtractedEntity>()
                for (i in 0 until numLabels) {
                    val kept = mutableListOf<Span>()
                    for (span in spansByLabel[i].sortedByDescending { it.score }) {
                        if (kept.any { it.overlaps(span) }) continue
                        kept += span
                        val startChar = words[span.startWord].start
                        val endChar = words[span.endWord].end
                        val spanText = normalized.substring(startChar, endChar).trim()
                        if (spanText.isEmpty()) continue
                        entities += ExtractedEntity(spanText, labels[i], span.score, startChar, endChar)
                    }
                }
                entities.sortBy { it.start }
                entities
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }

    override fun close() {
        session.close()
        tokenizer.close()
    }

    private fun tensor(data: LongArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)

    private data class Word(val text: String, val start: Int, val end: Int)

    private data class Span(val startWord: Int, val endWord: Int, val score: Float) {
        fun overlaps(other: Span): Boolean = startWord <= other.endWord && other.startWord <= endWord
    }

    companion object {
        private const val TAG = "Gliner2Extractor"
        const val DEFAULT_DIR = "models/gliner2"
        const val MAX_SEQ_LEN = 512
        const val MAX_WIDTH = 8
        const val TASK_TYPE_ENTITIES = 0L
        const val SEP_TEXT_ID = 250103L

        val DEFAULT_LABELS = listOf(
            "person", "organization", "location", "product",
            "technology", "event", "date", "topic",
        )

        private val WORD_REGEX = Regex(
            "(?:https?://[^\\s]+|www\\.[^\\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}|@[a-z0-9_]+|\\w+(?:[-_]\\w+)*|\\S",
            RegexOption.IGNORE_CASE,
        )

        fun create(context: Context, dir: File? = null): Gliner2Extractor {
            val modelDir = dir ?: File(context.filesDir, DEFAULT_DIR)
            val tokFile = File(modelDir, "tokenizer.json")
            require(tokFile.canRead()) {
                "Missing ${tokFile.absolutePath} — push via: adb push <f> /data/local/tmp/ && " +
                    "adb shell run-as com.mobilerag sh -c 'mkdir -p files/$DEFAULT_DIR && cp /data/local/tmp/<f> files/$DEFAULT_DIR/'"
            }
            val modelFile = File(modelDir, "model_int8.onnx")
            require(modelFile.canRead()) { "Missing ${modelFile.absolutePath} — same push path as tokenizer.json" }
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
            val session = OrtEnvironment.getEnvironment().createSession(modelFile.absolutePath, opts)
            return Gliner2Extractor(HuggingFaceTokenizer.newInstance(tokFile.toPath()), session)
        }

        internal fun normalizeText(text: String): String {
            if (text.isBlank()) return "."
            return if (text.endsWith('.') || text.endsWith('!') || text.endsWith('?')) text else "$text."
        }

        private fun splitWords(text: String): List<Word> =
            WORD_REGEX.findAll(text).map { m ->
                Word(m.value.lowercase(), m.range.first, m.range.last + 1)
            }.toList()

        private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()
    }
}

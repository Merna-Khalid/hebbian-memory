package com.mobilerag.setup

import android.content.Context
import android.os.StatFs
import java.io.File

/**
 * The models the app needs, where each comes from, and where it lives on the phone. First
 * run downloads whatever is missing (setup/ModelDownloadWorker.kt); after that everything
 * runs offline. Only files the code actually loads are listed. Sizes are exact and double as
 * the integrity check (matching the files verified on the dev phone, 2026-09-26).
 */
object ModelCatalog {

    data class Item(
        val url: String,
        /** Relative to filesDir. */
        val path: String,
        val bytes: Long,
        /** Which mind this belongs to, for the setup screen. */
        val group: Group,
    )

    enum class Group(val title: String, val role: String) {
        Language("Qwen3-1.7B", "language · Takemura and Khepri speak and reason"),
        Meaning("EmbeddingGemma-300M", "meaning · memories and notes are compared by sense"),
        Entities("GLiNER2", "entities · people, places and dates found in your notes"),
    }

    private const val HF = "https://huggingface.co"
    private const val GEMMA = "$HF/onnx-community/embeddinggemma-300m-ONNX/resolve/main"
    private const val GLINER = "$HF/cuerbot/gliner2-multi-v1/resolve/main"

    val items = listOf(
        Item("$HF/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf", "models/Qwen3-1.7B-Q8_0.gguf", 1_834_426_016, Group.Language),
        Item("$GEMMA/onnx/model_q4f16.onnx", "models/embeddinggemma/model_q4f16.onnx", 705_221, Group.Meaning),
        Item("$GEMMA/onnx/model_q4f16.onnx_data", "models/embeddinggemma/model_q4f16.onnx_data", 175_410_176, Group.Meaning),
        Item("$GEMMA/tokenizer.json", "models/embeddinggemma/tokenizer.json", 20_323_312, Group.Meaning),
        Item("$GLINER/model_int8.onnx", "models/gliner2/model_int8.onnx", 376_147_771, Group.Entities),
        Item("$GLINER/tokenizer.json", "models/gliner2/tokenizer.json", 16_337_353, Group.Entities),
        // (The repo's config.json / tokenizer_config.json aren't read by Gliner2Extractor.)
    )

    val totalBytes: Long = items.sumOf { it.bytes }

    fun file(context: Context, item: Item) = File(context.filesDir, item.path)

    fun isPresent(context: Context, item: Item): Boolean {
        // Any installed GGUF satisfies the language model (a phone set up by hand may carry
        // a different size; RagPipeline.pickModel chooses among them).
        if (item.group == Group.Language) {
            val models = File(context.filesDir, "models").listFiles() ?: return false
            if (models.any { it.extension == "gguf" && it.length() > 100_000_000 }) return true
        }
        val f = file(context, item)
        return f.isFile && f.length() == item.bytes
    }

    fun missing(context: Context): List<Item> = items.filterNot { isPresent(context, it) }

    fun ready(context: Context): Boolean = missing(context).isEmpty()

    /** Bytes still to fetch, net of any partial (.part) files already on disk. */
    fun remainingBytes(context: Context): Long = missing(context).sumOf { item ->
        item.bytes - (File(context.filesDir, item.path + ".part").takeIf { it.isFile }?.length() ?: 0L)
    }

    fun freeBytes(context: Context): Long = StatFs(context.filesDir.absolutePath).availableBytes
}

package com.mobilerag.embeddings

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Locale

/**
 * Minimal WordPiece tokenizer compatible with BERT-family vocab files (vocab.txt).
 * Sufficient for all-MiniLM-L6-v2; EmbeddingGemma (SentencePiece) will need a different tokenizer later.
 */
class WordPieceTokenizer(context: Context, vocabAsset: String = "vocab.txt") {

    private val vocab: Map<String, Int> = context.assets.open(vocabAsset).use { stream ->
        BufferedReader(InputStreamReader(stream)).lineSequence()
            .mapIndexed { index, token -> token to index }
            .toMap()
    }

    private val clsId = vocab.getValue("[CLS]")
    private val sepId = vocab.getValue("[SEP]")
    private val unkId = vocab.getValue("[UNK]")
    private val padId = vocab.getValue("[PAD]")

    data class Encoding(val inputIds: LongArray, val attentionMask: LongArray, val tokenTypeIds: LongArray)

    fun encode(text: String, maxLength: Int = 256): Encoding {
        val tokens = mutableListOf(clsId)
        for (word in basicTokenize(text)) {
            if (tokens.size >= maxLength - 1) break
            tokens.addAll(wordPieceSplit(word))
        }
        tokens.add(sepId)

        val ids = LongArray(maxLength) { padId.toLong() }
        val mask = LongArray(maxLength) { 0L }
        tokens.forEachIndexed { i, id ->
            ids[i] = id.toLong()
            mask[i] = 1L
        }
        return Encoding(ids, mask, LongArray(maxLength))
    }

    private fun basicTokenize(text: String): List<String> {
        val out = mutableListOf<String>()
        val sb = StringBuilder()
        for (c in text.lowercase(Locale.ROOT)) {
            when {
                c.isWhitespace() -> flush(sb, out)
                isChineseChar(c) || isPunctuation(c) -> {
                    flush(sb, out)
                    out.add(c.toString())
                }
                else -> sb.append(c)
            }
        }
        flush(sb, out)
        return out
    }

    private fun wordPieceSplit(word: String): List<Int> {
        if (word.length > 100) return listOf(unkId)
        val ids = mutableListOf<Int>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var found = -1
            while (start < end) {
                val piece = (if (start > 0) "##" else "") + word.substring(start, end)
                val id = vocab[piece]
                if (id != null) {
                    found = id
                    break
                }
                end--
            }
            if (found < 0) return listOf(unkId)
            ids.add(found)
            start = end
        }
        return ids
    }

    private fun flush(sb: StringBuilder, out: MutableList<String>) {
        if (sb.isNotEmpty()) {
            out.add(sb.toString())
            sb.setLength(0)
        }
    }

    private fun isChineseChar(c: Char): Boolean {
        val cp = c.code
        return (cp in 0x4E00..0x9FFF) || (cp in 0x3400..0x4DBF) || (cp in 0x20000..0x2A6DF) ||
            (cp in 0x2A700..0x2B73F) || (cp in 0x2B740..0x2B81F) || (cp in 0x2B820..0x2CEAF) ||
            (cp in 0xF900..0xFAFF) || (cp in 0x2F800..0x2FA1F)
    }

    private fun isPunctuation(c: Char): Boolean {
        val cp = c.code
        return (cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)
    }
}

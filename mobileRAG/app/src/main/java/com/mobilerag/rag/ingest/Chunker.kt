package com.mobilerag.rag.ingest

import com.mobilerag.embeddings.GemmaTokenizer

/**
 * Sentence-aware chunker: splits on `.!?\n` boundaries and markdown headings, targets
 * ≤ [maxTokens] per chunk (normal prose lands in the 300–500 range) with a trailing-sentence
 * overlap of ~[overlapTokens], and hard-splits any single sentence over
 * [GemmaTokenizer.MAX_TOKENS] at word boundaries.
 */
class Chunker(
    private val countTokens: (String) -> Int,
    private val maxTokens: Int = 500,
    private val overlapTokens: Int = 50,
) {
    data class Chunk(val text: String, val tokenCount: Int)

    private val headerRegex = Regex("^#{1,6}\\s")
    private val sentenceRegex = Regex("(?<=[.!?\\n])\\s+")
    private val whitespaceRegex = Regex("\\s+")

    fun chunk(document: String): List<Chunk> {
        val chunks = mutableListOf<Chunk>()
        val current = ArrayDeque<Pair<String, Int>>() // sentence text → token count
        var currentTokens = 0
        var overlapSeeded = 0 // tokens at the head of current that duplicate the previous chunk's tail

        fun emitCurrent() {
            if (current.isEmpty()) return
            if (currentTokens <= overlapSeeded && chunks.isNotEmpty()) return // pure overlap duplicate
            val text = current.joinToString(" ") { it.first }
            if (text.isNotBlank()) chunks += Chunk(text, currentTokens)
        }

        fun flushWithOverlap() {
            emitCurrent()
            val keep = ArrayDeque<Pair<String, Int>>()
            var kept = 0
            for (part in current.toList().asReversed()) {
                if (keep.isNotEmpty() && kept + part.second > overlapTokens) break
                keep.addFirst(part)
                kept += part.second
                if (kept >= overlapTokens) break
            }
            current.clear()
            current.addAll(keep)
            currentTokens = kept
            overlapSeeded = kept
        }

        for (segment in splitSegments(document)) {
            val segmentTokens = countTokens(segment.text)
            if (segmentTokens > GemmaTokenizer.MAX_TOKENS) {
                flushWithOverlap()
                current.clear()
                currentTokens = 0
                overlapSeeded = 0
                for (piece in hardSplit(segment.text)) {
                    chunks += Chunk(piece, countTokens(piece))
                }
                continue
            }
            if (segment.isHeader) {
                // headings open a fresh section: pending overlap is dropped
                emitCurrent()
                current.clear()
                currentTokens = 0
                overlapSeeded = 0
                current.addLast(segment.text to segmentTokens)
                currentTokens = segmentTokens
                continue
            }
            if (currentTokens > overlapSeeded && currentTokens + segmentTokens > maxTokens) {
                flushWithOverlap()
            }
            current.addLast(segment.text to segmentTokens)
            currentTokens += segmentTokens
        }
        emitCurrent()
        return chunks
    }

    private data class Segment(val text: String, val isHeader: Boolean)

    private fun splitSegments(document: String): List<Segment> = buildList {
        for (line in document.lines()) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue
            if (headerRegex.containsMatchIn(trimmed)) {
                add(Segment(trimmed, isHeader = true))
            } else {
                for (sentence in trimmed.split(sentenceRegex)) {
                    val s = sentence.trim()
                    if (s.isNotEmpty()) add(Segment(s, isHeader = false))
                }
            }
        }
    }

    /** Word-count sum approximates joined token count; tokenizer truncates at MAX_TOKENS regardless. */
    private fun hardSplit(sentence: String): List<String> {
        val pieces = mutableListOf<String>()
        var sb = StringBuilder()
        var tokens = 0
        for (word in sentence.split(whitespaceRegex)) {
            val wordTokens = countTokens(word)
            if (tokens + wordTokens > GemmaTokenizer.MAX_TOKENS && sb.isNotEmpty()) {
                pieces += sb.toString()
                sb = StringBuilder()
                tokens = 0
            }
            if (sb.isNotEmpty()) sb.append(' ')
            sb.append(word)
            tokens += wordTokens
        }
        if (sb.isNotEmpty()) pieces += sb.toString()
        return pieces
    }
}

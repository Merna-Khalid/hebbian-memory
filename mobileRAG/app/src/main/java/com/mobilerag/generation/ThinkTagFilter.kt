package com.mobilerag.generation

/** Drops a leading `<think>…</think>` block (Qwen3 reasoning) from the token stream.
 *  Only the prefix is inspected; once the stream is known not to start with `<think>`,
 *  everything passes through untouched. Shared by RagPipeline and the Hebbian tutor engine. */
internal class ThinkTagFilter {
    private val open = "<think>"
    private val close = "</think>"
    private var inThink = false
    private var decided = false
    private val buf = StringBuilder()

    fun feed(token: String): String {
        buf.append(token)
        return drain(final = false)
    }

    fun flush(): String = drain(final = true)

    private fun drain(final: Boolean): String {
        val out = StringBuilder()
        while (buf.isNotEmpty()) {
            if (!decided) {
                if (!final && buf.length < open.length && open.startsWith(buf)) break // wait for more
                if (buf.startsWith(open)) {
                    buf.delete(0, open.length)
                    inThink = true
                    decided = true
                    continue
                }
                decided = true // doesn't start with <think> — normal stream
            }
            if (inThink) {
                val closeIdx = buf.indexOf(close)
                if (closeIdx >= 0) {
                    buf.delete(0, closeIdx + close.length)
                    // trim leading whitespace/newlines after the think block
                    while (buf.isNotEmpty() && buf[0].isWhitespace()) buf.deleteCharAt(0)
                    inThink = false
                    continue
                }
                // hold back a possible partial "</think>" at the tail
                val keep = if (final) 0 else partialTagTail(buf, close)
                buf.delete(0, buf.length - keep)
                break
            }
            out.append(buf)
            buf.clear()
        }
        if (final && buf.isNotEmpty() && decided && !inThink) {
            out.append(buf)
            buf.clear()
        }
        return out.toString()
    }

    /** Length of the longest suffix of [buf] that is a prefix of [tag]. */
    private fun partialTagTail(buf: StringBuilder, tag: String): Int {
        val max = minOf(buf.length, tag.length - 1)
        for (len in max downTo 1) {
            if (buf.substring(buf.length - len) == tag.substring(0, len)) return len
        }
        return 0
    }
}

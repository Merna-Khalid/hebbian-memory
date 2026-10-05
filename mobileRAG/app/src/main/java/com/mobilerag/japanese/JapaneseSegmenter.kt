package com.mobilerag.japanese

/**
 * Splits Japanese text into tappable units without a morphological analyzer.
 *
 * Japanese has no spaces, so a tap-to-translate overlay needs *some* notion of a word. A real
 * tokenizer (MeCab/Sudachi + a dictionary) is tens of megabytes; this instead exploits the fact
 * that Japanese orthography already marks most word boundaries by script change:
 *
 *  - a kanji run starts a content word;
 *  - the hiragana that follows it is usually okurigana (inflection) and belongs to that word;
 *  - katakana runs are loanwords and stand alone;
 *  - single-kana grammatical particles (は, を, に …) are their own thing and must NOT be glued
 *    onto the noun before them, or 本を読む becomes "本を" + "読む".
 *
 * So: run-length split by script, peel leading particles off every hiragana run, then merge each
 * hiragana remainder back onto the kanji run it trails.
 *
 * This is a heuristic and it mis-segments compounds (長い間 → 長い + 間). It is accurate enough
 * that a tapped token translates sensibly, which is all the overlay needs.
 */
object JapaneseSegmenter {

    /** One unit of segmented text. Only [clickable] tokens get a translation affordance. */
    data class Token(
        val text: String,
        /** Offset of [text] in the original string — used to map taps back to layout positions. */
        val start: Int,
        val clickable: Boolean,
    ) {
        val end: Int get() = start + text.length
    }

    private enum class Script { KANJI, HIRAGANA, KATAKANA, LATIN, DIGIT, OTHER }

    /**
     * Single-kana particles and sentence-final markers. Peeling these keeps content words intact.
     * ("の" and "と" are also parts of many compounds; splitting them costs little and prevents
     * the far worse failure of swallowing the next word.)
     */
    private val PARTICLES = setOf(
        'は', 'が', 'を', 'に', 'へ', 'と', 'で', 'も', 'の', 'や', 'か', 'ね', 'よ', 'ば', 'さ', 'な',
    )

    /** Kana that can never begin a word — small kana and the long-vowel mark. */
    private val NON_INITIAL = setOf('ん', 'ー', 'ゃ', 'ゅ', 'ょ', 'っ', 'ャ', 'ュ', 'ョ', 'ッ')

    private fun scriptOf(c: Char): Script = when {
        // CJK Unified Ideographs + Extension A, plus the iteration mark 々 and 〆.
        c in '一'..'鿿' || c in '㐀'..'䶿' || c == '々' || c == '〆' -> Script.KANJI
        c in '぀'..'ゟ' -> Script.HIRAGANA
        // Katakana block (includes ー U+30FC) plus halfwidth katakana.
        c in '゠'..'ヿ' || c in 'ｦ'..'ﾝ' -> Script.KATAKANA
        c.isLetter() -> Script.LATIN
        c.isDigit() -> Script.DIGIT
        else -> Script.OTHER
    }

    /** True when [text] contains anything worth offering a translation for. */
    fun hasJapanese(text: String): Boolean = text.any {
        val s = scriptOf(it)
        s == Script.KANJI || s == Script.HIRAGANA || s == Script.KATAKANA
    }

    fun segment(text: String): List<Token> {
        if (text.isEmpty()) return emptyList()

        // 1. Run-length split by script.
        data class Run(val text: String, val start: Int, val script: Script)
        val runs = mutableListOf<Run>()
        var runStart = 0
        var i = 1
        while (i <= text.length) {
            val boundary = i == text.length || scriptOf(text[i]) != scriptOf(text[runStart])
            if (boundary) {
                runs += Run(text.substring(runStart, i), runStart, scriptOf(text[runStart]))
                runStart = i
            }
            i++
        }

        // 2. Split hiragana runs into [particle…, remainder], keeping everything else whole.
        data class Piece(val text: String, val start: Int, val script: Script, val isParticle: Boolean)
        val pieces = mutableListOf<Piece>()
        for ((runIndex, run) in runs.withIndex()) {
            if (run.script != Script.HIRAGANA) {
                pieces += Piece(run.text, run.start, run.script, isParticle = false)
                continue
            }
            // Particles only ever *follow* a content word, so peeling is gated on the previous run
            // being kanji or katakana. Without that gate a word that merely starts with a
            // particle kana gets destroyed: ねこ would lose its ね and leave an unclickable こ.
            val followsContentWord = runIndex > 0 &&
                (runs[runIndex - 1].script == Script.KANJI || runs[runIndex - 1].script == Script.KATAKANA)
            var offset = 0
            // A particle immediately followed by kana that cannot start a word (っ, ん, small kana)
            // is really part of a longer word, so stop there.
            while (followsContentWord &&
                offset < run.text.length &&
                run.text[offset] in PARTICLES &&
                (offset + 1 >= run.text.length || run.text[offset + 1] !in NON_INITIAL)
            ) {
                pieces += Piece(run.text[offset].toString(), run.start + offset, Script.HIRAGANA, isParticle = true)
                offset++
            }
            if (offset < run.text.length) {
                pieces += Piece(run.text.substring(offset), run.start + offset, Script.HIRAGANA, isParticle = false)
            }
        }

        // 3. Merge each non-particle hiragana remainder onto the kanji run it directly follows.
        val tokens = mutableListOf<Token>()
        var k = 0
        while (k < pieces.size) {
            val p = pieces[k]
            val next = pieces.getOrNull(k + 1)
            val mergesOkurigana = p.script == Script.KANJI &&
                next != null &&
                next.script == Script.HIRAGANA &&
                !next.isParticle &&
                next.start == p.start + p.text.length
            if (mergesOkurigana) {
                tokens += Token(p.text + next!!.text, p.start, clickable = true)
                k += 2
            } else {
                val clickable = when (p.script) {
                    Script.KANJI, Script.KATAKANA -> true
                    // Standalone kana words are worth a tap; bare particles are not.
                    Script.HIRAGANA -> !p.isParticle && p.text.length >= 2
                    else -> false
                }
                tokens += Token(p.text, p.start, clickable)
                k++
            }
        }
        return tokens
    }
}

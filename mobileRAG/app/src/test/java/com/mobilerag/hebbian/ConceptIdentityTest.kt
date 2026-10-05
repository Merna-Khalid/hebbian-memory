package com.mobilerag.hebbian

import com.mobilerag.hebbian.store.ConceptNode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test

/** Mirrored by "Hebbian Memory/tools/check_identity.py" — keep the label cases in sync. */
class ConceptIdentityTest {

    private fun node(id: String, label: String, type: String = "vocabulary", activations: Int = 1, created: Long = 0) =
        ConceptNode(
            nodeId = id, label = label, textRaw = label, textSummary = label, conceptType = type,
            sourceType = "user", sourceUri = null, createdAt = created, updatedAt = created,
            activationCount = activations, embedding = floatArrayOf(1f, 0f),
        )

    @Test
    fun labelKeyNormalization() {
        val cases = listOf(
            "  食べる " to "食べる",
            "「食べる」" to "食べる",
            "Taberu" to "taberu",
            "ＡＢＣ　ｄｅｆ" to "abc def",          // NFKC folds full-width letters and the ideographic space
            "te  form\tverbs" to "te form verbs",
            "て-form." to "て-form",
            "(passive voice)" to "passive voice",
            "C#" to "c#",                           // meaningful symbols survive
            "C++" to "c++",
            "〜ている" to "〜ている",
            "!!!" to "",
        )
        for ((input, expected) in cases) assertEquals("labelKey($input)", expected, ConceptIdentity.labelKey(input))
    }

    @Test
    fun sameLabelSameTypeIsReused_mostActivatedWins() {
        val a = node("a", "食べる", activations = 2)
        val b = node("b", " 食べる ", activations = 5)
        val hit = ConceptIdentity.resolve("食べる", "vocabulary", true, listOf(a, b), emptyList())
        assertSame(b, hit)
    }

    @Test
    fun activationTieGoesToEarliest() {
        val older = node("old", "食べる", activations = 3, created = 100)
        val newer = node("new", "食べる", activations = 3, created = 200)
        assertSame(older, ConceptIdentity.resolve("食べる", "vocabulary", true, listOf(newer, older), emptyList()))
    }

    @Test
    fun sameLabelDifferentTypeIsNotReused() {
        val grammar = node("g", "て-form", type = "grammar")
        assertNull(ConceptIdentity.resolve("て-form", "example", true, listOf(grammar), emptyList()))
    }

    @Test
    fun embeddingMatchNeedsSameTypeAndHighScore() {
        val n = node("n", "to eat")
        assertSame(n, ConceptIdentity.resolve("食べる", "vocabulary", true, emptyList(), listOf(n to 0.98)))
        assertNull(ConceptIdentity.resolve("食べる", "vocabulary", true, emptyList(), listOf(n to 0.97)))
        assertNull(ConceptIdentity.resolve("食べる", "grammar", true, emptyList(), listOf(n to 0.99)))
    }

    @Test
    fun embeddingMatchSkipsWrongTypeToFindTheRightOne() {
        val g = node("g", "x", type = "grammar")
        val v = node("v", "y", type = "vocabulary")
        assertSame(v, ConceptIdentity.resolve("z", "vocabulary", true, emptyList(), listOf(g to 0.99, v to 0.98)))
    }

    @Test
    fun unvalidatedAndEventsAreNeverDeduplicated() {
        val n = node("n", "食べる")
        assertNull(ConceptIdentity.resolve("食べる", "vocabulary", false, listOf(n), listOf(n to 1.0)))
        val e = node("e", "User: hi", type = "event")
        assertNull(ConceptIdentity.resolve("User: hi", "event", true, listOf(e), listOf(e to 1.0)))
    }

    @Test
    fun punctuationOnlyLabelFallsThroughToEmbedding() {
        val n = node("n", "!!!")
        assertNull(ConceptIdentity.resolve("!!!", "vocabulary", true, listOf(n), emptyList()))
    }
}

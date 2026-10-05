package com.mobilerag.hebbian

import com.mobilerag.hebbian.store.ConceptMerge
import com.mobilerag.hebbian.store.ConceptNode
import com.mobilerag.hebbian.store.HebbianEdge
import com.mobilerag.hebbian.store.InMemoryHebbianStore
import com.mobilerag.hebbian.store.SessionStatRow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import kotlin.math.abs

/**
 * The real stub TutorEngine (IngestPipeline + ConceptExtractor + ConceptIdentity) over an
 * in-memory store — the JVM twin of the on-device hebbian_ingest spike — plus a merge run.
 */
class IdentityEndToEndTest {

    private val turns = listOf(
        "Today I learned 食べる (taberu), to eat. ご飯を食べます。",
        "What is the te-form of 食べる? I want to say \"please eat\".",
        "Now 飲む (nomu), to drink — 水を飲みます。",
        "How do I conjugate 飲む in the past tense?",
        "行く (iku) means to go, right? 学校に行きます。",
        "Why is the te-form of 行く irregular — 行って?",
        "Today I reviewed 食べる again — 食べました。",
        "Is 食べて the te-form I asked about?",
        "Let me try: 毎日ご飯を食べます。",
    )

    @Test
    fun reMentionReusesTheConceptAndTheGraphStillLearns() = runBlocking {
        val store = InMemoryHebbianStore()
        val engine = buildStubEngine(store, Files.createTempDirectory("stub-engine").toFile())
        val sid = engine.startSession("test")
        var done = 0
        for (t in turns) engine.chat(t, sid).collect { if (it is TutorEvent.Done) done++ }
        assertEquals(turns.size, done)

        val labels = store.concepts.values.map { it.label }
        val todays = store.concepts.values.filter { ConceptIdentity.labelKey(it.label) == "today" }
        assertEquals("one 'Today' node among $labels", 1, todays.size)
        assertTrue("re-mention re-activated it", todays[0].activationCount >= 2)
        assertTrue("no duplicate groups left", ConceptMerge.plan(store.concepts.values.toList()).groups.isEmpty())

        val initial = listOf(0.6, 1.0, 1.2, 1.5)
        assertTrue("some hippocampal edge moved off its initial weight",
            store.edges.values.any { e -> initial.all { abs(e.hebbWeight - it) > 1e-6 } })
    }

    @Test
    fun mergeFoldsOldDuplicatesTogether() = runBlocking {
        val store = InMemoryHebbianStore()
        fun put(id: String, label: String, act: Int) {
            store.concepts[id] = ConceptNode(id, label, label, label, "vocabulary", "user", null, 0, 0, act, floatArrayOf(1f, 0f))
        }
        put("a", "食べる", 1)
        put("b", "「食べる」", 3)
        put("c", "飲む", 1)
        fun edge(s: String, d: String, w: Double, co: Int = 1) {
            store.edges[Triple(s, d, "hippocampal")] = HebbianEdge(s, d, w, 0.0, 0.0, co, 0, "hippocampal")
        }
        edge("a", "c", 2.0, co = 2)
        edge("b", "c", 1.0, co = 3)
        edge("c", "a", 1.5)
        edge("a", "b", 1.0)
        store.sessionStats["s" to "a"] = SessionStatRow(null, "s", "a", 1.0, 0.5, 0.5, 5, 0.0)
        store.sessionStats["s" to "b"] = SessionStatRow(null, "s", "b", 1.0, 0.5, 0.5, 2, 0.0)

        val plan = ConceptMerge.plan(store.listConcepts(100))
        assertEquals(mapOf("a" to "b"), plan.remap)
        store.applyMerge(plan)

        assertEquals(setOf("b", "c"), store.concepts.keys)
        assertEquals(4, store.concepts.getValue("b").activationCount)
        val bc = store.edges.getValue(Triple("b", "c", "hippocampal"))
        assertEquals(2.0, bc.hebbWeight, 0.0)
        assertEquals(5, bc.coActivationCount)
        assertEquals(1.5, store.edges.getValue(Triple("c", "b", "hippocampal")).hebbWeight, 0.0)
        assertFalse("self loop dropped", store.edges.containsKey(Triple("b", "b", "hippocampal")))
        assertEquals(listOf(5), store.sessionStats.values.map { it.activationCount })
    }
}

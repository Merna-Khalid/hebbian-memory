package com.mobilerag.hebbian.store

import com.ladybugdb.Connection
import com.ladybugdb.Database
import com.mobilerag.hebbian.buildStubEngine
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

/**
 * LadybugDB 0.20.3 can't replay a WAL record holding a value ≥ 4096 bytes, and Android
 * kills processes without closing them (see LadybugHebbianStore's class doc). The real
 * store runs on the dev machine through the lbug jar's desktop native; "kill" = copying
 * the db + WAL while the store is still open, then opening the copy.
 */
class LadybugCrashTest {

    @Before
    fun nativeAvailable() {
        val ok = runCatching {
            val dir = Files.createTempDirectory("lbug-probe").toFile()
            Database(File(dir, "p.lbug").absolutePath).close()
            dir.deleteRecursively()
        }.isSuccess
        assumeTrue("no LadybugDB native for this dev machine", ok)
    }

    /** Copies [dbPath] and its siblings as a kill would leave them; opens the copy. */
    private fun openKilledCopy(dbPath: String): Pair<LadybugHebbianStore, File> {
        val dir = Files.createTempDirectory("lbug-killed").toFile()
        val src = File(dbPath)
        src.parentFile.listFiles { f -> f.name.startsWith(src.name) }!!.forEach { it.copyTo(File(dir, it.name)) }
        val copy = LadybugHebbianStore()
        runBlocking { copy.open(File(dir, src.name).absolutePath) }
        return copy to dir
    }

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
    fun killAfterAnyTurnKeepsEveryConcept(): Unit = runBlocking {
        val dir = Files.createTempDirectory("lbug-crash").toFile()
        val path = File(dir, "hebbian.lbug").absolutePath
        val store = LadybugHebbianStore()
        store.open(path)
        try {
            val engine = buildStubEngine(store, File(dir, "state"))
            val sid = engine.startSession("test")
            for ((i, t) in turns.withIndex()) {
                engine.chat(t, sid).collect { }
                val live = store.listConcepts(1000).map { it.nodeId }.toSet()
                val (copy, copyDir) = openKilledCopy(path)
                try {
                    assertEquals("turn ${i + 1}: no WAL recovery needed", null, copy.walRecovery)
                    assertEquals("turn ${i + 1}: every concept survives the kill",
                        live, copy.listConcepts(1000).map { it.nodeId }.toSet())
                } finally {
                    copy.close()
                    copyDir.deleteRecursively()
                }
            }
            assertTrue("the run stored concepts", store.listConcepts(1000).isNotEmpty())
        } finally {
            store.close()
            dir.deleteRecursively()
        }
    }

    @Test
    fun unreplayableWalIsSetAsideAndTheLastCheckpointOpens(): Unit = runBlocking {
        val dir = Files.createTempDirectory("lbug-recover").toFile()
        val path = File(dir, "hebbian.lbug").absolutePath
        val store = LadybugHebbianStore()
        store.open(path)
        store.upsertConcept(ConceptNode("kept", "kept", "kept", "kept", "fact", "user", null, 0, 0, 0, FloatArray(768) { 0.01f }))
        store.close()

        // A write that skips the store's checkpoint: a 9 KB string left in the WAL.
        val db = Database(path)
        val conn = Connection(db)
        val big = "0.012345678,".repeat(768)
        val r = conn.query(
            "CREATE (:Concept {node_id: 'lost', label: 'lost', text_raw: '', text_summary: '', concept_type: 'fact', " +
                "source_type: 'user', source_uri: '', created_at: 0, updated_at: 0, activation_count: 0, " +
                "embedding: '$big', mean_m_t: 1.0, mean_dominance: 0.5, mean_r_t: 0.5, recent_events: '[]'})",
        )
        assertTrue(r.errorMessage ?: "", r.isSuccess)
        r.close()
        val (copy, copyDir) = openKilledCopy(path)
        try {
            assertNotNull("the WAL was set aside", copy.walRecovery)
            assertEquals(listOf("kept"), copy.listConcepts(100).map { it.nodeId })
            assertTrue("the unreplayable WAL is kept for inspection",
                copyDir.listFiles()!!.any { it.name.startsWith("hebbian.lbug.wal.corrupt-") })
        } finally {
            copy.close()
            copyDir.deleteRecursively()
            conn.close()
            db.close()
            dir.deleteRecursively()
        }
    }
}

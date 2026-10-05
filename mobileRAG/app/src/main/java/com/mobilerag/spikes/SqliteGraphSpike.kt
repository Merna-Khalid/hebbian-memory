package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.graph.SqliteGraphStore
import java.io.File

/**
 * Spike B-fallback: SQLite graph with recursive-CTE traversal and FTS5.
 * Always runs, so Plan B is proven regardless of the LadybugDB outcome.
 *
 * Uses a scratch db under files/spikes/ — NOT the memory-space graph.db — because the
 * spike wipes its database for a deterministic run.
 */
class SqliteGraphSpike : Spike {
    override val id = "sqlite_graph"
    override val title = "SQLite graph (fallback)"
    override val description = "Entities/edges in SQLite, 2-hop traversal via recursive CTE, FTS5 keyword search."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val instr = Instrumentation(context, log)
        val dbFile = File(context.filesDir, "spikes/sqlite_graph.db")
        dbFile.delete()
        val store = SqliteGraphStore(context, dbFile.absolutePath)

        instr.mark("insert 50 nodes / 60 edges")
        val alice = store.addEntity("Alice", "Person")
        val bob = store.addEntity("Bob", "Person")
        val carol = store.addEntity("Carol", "Person")
        val acme = store.addEntity("Acme Corp", "Organization")
        val report = store.addEntity("Q3 Report", "Document")
        val others = (1..45).map { store.addEntity("Entity$it", "Misc") }
        store.addEdge(alice, bob, "KNOWS")
        store.addEdge(bob, carol, "KNOWS")
        store.addEdge(alice, acme, "WORKS_AT")
        store.addEdge(carol, acme, "WORKS_AT")
        store.addEdge(acme, report, "PUBLISHED")
        others.windowed(2, 1).forEach { (a, b) -> store.addEdge(a, b, "LINKED") }
        val insertMs = instr.elapsed("insert 50 nodes / 60 edges")

        instr.mark("2-hop traversal from Alice")
        val reachable = store.traverse(alice, maxDepth = 2)
        val travMs = instr.elapsed("2-hop traversal from Alice")
        log("Reachable in ≤2 hops from Alice: ${reachable.map { it.name }}")
        val traversalOk = reachable.map { it.name }
            .containsAll(listOf("Bob", "Carol", "Acme Corp", "Q3 Report")) // Q3 Report is exactly 2 hops via Acme

        val ftsHits = run {
            val db = android.database.sqlite.SQLiteDatabase.openDatabase(dbFile.absolutePath, null, 0)
            db.rawQuery("SELECT rowid FROM entity_fts WHERE entity_fts MATCH ?", arrayOf("Acme*")).use { c ->
                buildList { while (c.moveToNext()) add(c.getLong(0)) }.also { db.close() }
            }
        }
        log("FTS5 'Acme*' hits: $ftsHits")
        val ftsOk = acme in ftsHits

        store.close()

        val passed = traversalOk && ftsOk
        return SpikeResult(
            passed = passed,
            summary = if (passed) "Recursive CTE traversal + FTS5 both work" else
                "traversalOk=$traversalOk ftsOk=$ftsOk",
            metrics = mapOf(
                "insert time" to "$insertMs ms",
                "2-hop traversal" to "$travMs ms",
                "entities" to "50",
            ),
        )
    }
}

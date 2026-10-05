package com.mobilerag.spikes

import android.content.Context
import java.io.File

/**
 * Spike B: LadybugDB Cypher round-trip on-device.
 *
 * Hazard found and fixed (2026-09-12): the published com.ladybugdb:lbug jar contains native
 * libs only for linux/osx/windows (glibc-linked) — no android_arm64 build exists despite the
 * docs claiming Android support. We cross-compiled the core (v0.20.3) + JNI binding with
 * NDK 27.2 (two atomic_ref sites patched to __atomic_exchange_n; bundled into this app as
 * jniLibs/arm64-v8a/liblbug.so + resources/liblbug_java_native.so_android_arm64).
 */
class LadybugGraphSpike : Spike {
    override val id = "ladybug_graph"
    override val title = "LadybugDB graph"
    override val description = "Cypher CREATE/MATCH round-trip via the official Java API."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val instr = Instrumentation(context, log)
        return try {
            instr.mark("native lib load + open db")
            val dbPath = File(context.filesDir, "lbug_spike.db").absolutePath
            File(dbPath).deleteRecursively()

            val db = com.ladybugdb.Database(dbPath)
            val conn = com.ladybugdb.Connection(db)
            instr.elapsed("native lib load + open db")

            instr.mark("schema + insert")
            conn.query("CREATE NODE TABLE Person(id INT64 PRIMARY KEY, name STRING)")
            conn.query("CREATE REL TABLE KNOWS(FROM Person TO Person)")
            conn.query("CREATE (:Person {id: 1, name: 'Alice'})")
            conn.query("CREATE (:Person {id: 2, name: 'Bob'})")
            conn.query("CREATE (:Person {id: 3, name: 'Carol'})")
            conn.query("MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:KNOWS]->(b)")
            conn.query("MATCH (a:Person {id: 2}), (b:Person {id: 3}) CREATE (a)-[:KNOWS]->(b)")
            val insertMs = instr.elapsed("schema + insert")

            instr.mark("2-hop MATCH")
            val result = conn.query(
                "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(p:Person) RETURN p.name"
            )
            val names = mutableListOf<String>()
            while (result.hasNext()) {
                val tuple = result.next
                names.add(tuple.getValue(0L).toString())
                tuple.close()
            }
            val travMs = instr.elapsed("2-hop MATCH")
            log("2-hop from Alice: $names")

            conn.close()
            db.close()

            val passed = names.containsAll(listOf("Bob", "Carol"))
            SpikeResult(
                passed = passed,
                summary = if (passed) "Cypher round-trip works on-device" else "Unexpected result: $names",
                metrics = mapOf(
                    "insert" to "$insertMs ms",
                    "2-hop MATCH" to "$travMs ms",
                ),
            )
        } catch (t: Throwable) {
            log("Native load / query failed: ${t.message}")
            SpikeResult(
                passed = false,
                summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                error = t.stackTraceToString().take(2000),
            )
        }
    }
}

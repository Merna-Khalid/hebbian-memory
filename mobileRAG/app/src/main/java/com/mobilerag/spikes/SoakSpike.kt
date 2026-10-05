package com.mobilerag.spikes

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.graph.GraphStoreFactory
import com.mobilerag.profile.SpaceManager
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.RagEvent
import com.mobilerag.rag.RagPipeline
import com.mobilerag.rag.VectorStore
import com.mobilerag.rag.ingest.DocumentImporter
import com.mobilerag.rag.ingest.GraphIndexer
import com.mobilerag.rag.ingest.Indexer
import com.mobilerag.rag.retrieve.RetrievalEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Phase 4 spike: mixed-workload soak. Loops chat-style queries through RagPipeline (alternating
 * vector/hybrid, questions cycled from files/eval/eval-v1.json), unloads + reloads the LLM every
 * [RELOAD_EVERY_QUERIES] queries, and triggers one mid-run re-import of an eval corpus doc
 * (files/eval/corpus) if present. PSS + heap are sampled every [SAMPLE_INTERVAL_MS] and the run
 * is written to files/eval/soak-<timestamp>.json. Per-iteration Throwables are counted, not fatal.
 */
class SoakSpike : Spike {
    override val id = "soak"
    override val title = "Soak test"
    override val description = "Mixed workload for $DURATION_MINUTES min: chat queries (alternating vector/hybrid), " +
        "periodic LLM unload/reload, one mid-run re-import. PSS/heap sampled every 10 s → files/eval/soak-<ts>.json."

    private data class Sample(val tSec: Long, val pssMb: Double, val heapMb: Long)

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        val evalFile = File(context.filesDir, "eval/eval-v1.json")
        if (!evalFile.canRead()) {
            return SpikeResult(
                passed = false,
                summary = "Missing ${evalFile.absolutePath} — push eval-v1.json (see hybrid_eval spike)",
            )
        }
        val questions = parseQuestions(evalFile)
        if (questions.isEmpty()) return SpikeResult(passed = false, summary = "No questions in ${evalFile.name}")
        log("${questions.size} questions, duration $DURATION_MINUTES min")

        val engine = try {
            EmbeddingGemmaEngine.create(context)
        } catch (t: Throwable) {
            return SpikeResult(passed = false, summary = "Engine init failed: ${t.message?.take(200)}", error = t.stackTraceToString().take(2000))
        }
        try {
            val spaceDir = SpaceManager.activeDir(context)
            val db = RagDatabase(context, File(spaceDir, "rag.db").absolutePath)
            val retrieval = try {
                RetrievalEngine(engine, VectorStore(db, engine.dimensions), db, GraphStoreFactory.create(context, spaceDir))
            } catch (t: Throwable) {
                db.close()
                throw t
            }
            val pipeline = RagPipeline(context, engine, retrieval)
            val indexer = Indexer(db, engine)
            val graphIndexer = GraphIndexer(context, db) { GraphStoreFactory.create(context, spaceDir) }
            val savedMode = pipeline.retrievalMode()

            val samples = mutableListOf<Sample>()
            val startMs = SystemClock.elapsedRealtime()
            val startPssKb = QueryTimer.pssKb()
            var lastSampleMs = startMs - SAMPLE_INTERVAL_MS // sample immediately
            var queries = 0
            var errors = 0
            var reloads = 0
            var reimports = 0
            var qIndex = 0
            var reimportDone = false
            val durationMs = DURATION_MINUTES * 60_000L

            try {
                while (SystemClock.elapsedRealtime() - startMs < durationMs) {
                    currentCoroutineContext().ensureActive()
                    val q = questions[qIndex++ % questions.size]
                    pipeline.setRetrievalMode(if (queries % 2 == 0) RagPipeline.MODE_VECTOR else RagPipeline.MODE_HYBRID)
                    try {
                        var failure: String? = null
                        pipeline.ask(q).collect { ev -> if (ev is RagEvent.Failed) failure = ev.error }
                        queries++
                        failure?.let {
                            errors++
                            if (errors <= 5) log("query failed: ${it.take(120)}")
                        }
                    } catch (t: Throwable) {
                        if (t is CancellationException) throw t
                        errors++
                        if (errors <= 5) log("query failed: ${t.javaClass.simpleName}: ${t.message?.take(120)}")
                    }

                    // periodic unload → next query pays the mmap reload
                    if (queries > 0 && queries % RELOAD_EVERY_QUERIES == 0) {
                        pipeline.unloadModel()
                        reloads++
                    }

                    // one mid-run re-import of a bundled eval corpus doc
                    if (!reimportDone && SystemClock.elapsedRealtime() - startMs >= durationMs / 2) {
                        reimportDone = true
                        reimports += reimportCorpusDoc(context, indexer, graphIndexer, log)
                    }

                    val now = SystemClock.elapsedRealtime()
                    if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
                        samples += Sample((now - startMs) / 1000, QueryTimer.pssKb() / 1024.0, QueryTimer.heapMb())
                        lastSampleMs = now
                        if (samples.size % 6 == 1) {
                            val s = samples.last()
                            log("t=${s.tSec}s PSS %.0f MB, heap ${s.heapMb} MB, queries=$queries, errors=$errors".format(s.pssMb))
                        }
                    }
                }
            } finally {
                pipeline.setRetrievalMode(savedMode)
                pipeline.unloadModel()
                db.close()
            }

            val endPssKb = QueryTimer.pssKb()
            val peakPssMb = samples.maxOfOrNull { it.pssMb } ?: (startPssKb / 1024.0)
            val peakHeapMb = samples.maxOfOrNull { it.heapMb } ?: 0L
            val reportFile = writeReport(
                context, samples, queries, errors, reloads, reimports,
                startPssKb / 1024.0, endPssKb / 1024.0, peakPssMb, peakHeapMb,
            )
            Log.i(TAG, "report: ${reportFile.absolutePath}")

            val summary = "%d queries, %d errors, %d reloads · PSS %.0f→%.0f MB (peak %.0f, Δ%+.0f) · heap peak %d MB · %s".format(
                queries, errors, reloads,
                startPssKb / 1024.0, endPssKb / 1024.0, peakPssMb, (endPssKb - startPssKb) / 1024.0,
                peakHeapMb, reportFile.name,
            )
            return SpikeResult(
                passed = queries > 0 && errors == 0,
                summary = summary,
                metrics = mapOf(
                    "queries" to queries.toString(),
                    "errors" to errors.toString(),
                    "llm reloads" to reloads.toString(),
                    "re-imports" to reimports.toString(),
                    "PSS start→end MB" to "%.0f→%.0f".format(startPssKb / 1024.0, endPssKb / 1024.0),
                    "PSS peak MB" to "%.0f".format(peakPssMb),
                    "report" to reportFile.name,
                ),
            )
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            return SpikeResult(
                passed = false,
                summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                error = t.stackTraceToString().take(2000),
            )
        } finally {
            engine.close()
        }
    }

    /** Copies one eval corpus doc into files/corpus and re-indexes it (embed + graph rebuild).
     *  Returns 1 on success, 0 when skipped/failed. */
    private suspend fun reimportCorpusDoc(
        context: Context,
        indexer: Indexer,
        graphIndexer: GraphIndexer,
        log: (String) -> Unit,
    ): Int {
        val file = File(context.filesDir, "eval/corpus").listFiles()
            ?.firstOrNull { it.extension.lowercase() in setOf("md", "txt") && it.canRead() }
        if (file == null) {
            log("no files/eval/corpus docs — skipping mid-run re-import")
            return 0
        }
        return try {
            val corpusDir = File(context.filesDir, "corpus").apply { mkdirs() }
            val out = File(corpusDir, file.name)
            file.copyTo(out, overwrite = true)
            val digest = MessageDigest.getInstance("SHA-256")
            out.inputStream().use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            val sha = digest.digest().joinToString("") { "%02x".format(it) }
            val changed = indexer.index(listOf(DocumentImporter.Imported(out, file.name, sha)))
            if (changed.isNotEmpty()) graphIndexer.buildGraph(changed)
            log("re-imported ${file.name} (${changed.size} reindexed)")
            1
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            Log.w(TAG, "re-import failed", t)
            log("re-import failed: ${t.message?.take(120)}")
            0
        }
    }

    private fun writeReport(
        context: Context,
        samples: List<Sample>,
        queries: Int,
        errors: Int,
        reloads: Int,
        reimports: Int,
        pssStartMb: Double,
        pssEndMb: Double,
        pssPeakMb: Double,
        heapPeakMb: Long,
    ): File {
        val report = JSONObject()
            .put("durationMin", DURATION_MINUTES)
            .put("queries", queries)
            .put("errors", errors)
            .put("llmReloads", reloads)
            .put("reimports", reimports)
            .put("pssStartMb", pssStartMb)
            .put("pssEndMb", pssEndMb)
            .put("pssDeltaMb", pssEndMb - pssStartMb)
            .put("pssPeakMb", pssPeakMb)
            .put("heapPeakMb", heapPeakMb)
            .put("samples", JSONArray().apply {
                samples.forEach { s ->
                    put(JSONObject().put("tSec", s.tSec).put("pssMb", s.pssMb).put("heapMb", s.heapMb))
                }
            })
        val dir = File(context.filesDir, "eval").apply { mkdirs() }
        val ts = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        return File(dir, "soak-$ts.json").also { it.writeText(report.toString(2)) }
    }

    private fun parseQuestions(file: File): List<String> = buildList {
        val json = JSONArray(file.readText())
        for (i in 0 until json.length()) add(json.getJSONObject(i).getString("q"))
    }

    companion object {
        private const val TAG = "SoakSpike"
        const val DURATION_MINUTES = 30 // tweak for a longer/shorter soak
        private const val SAMPLE_INTERVAL_MS = 10_000L
        private const val RELOAD_EVERY_QUERIES = 10
    }
}

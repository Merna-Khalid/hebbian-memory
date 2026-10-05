package com.mobilerag.eval

import android.content.Context
import android.util.Log
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.graph.GraphStoreFactory
import com.mobilerag.profile.SpaceManager
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.VectorStore
import com.mobilerag.rag.retrieve.RetrievalEngine
import com.mobilerag.rag.retrieve.RetrievalResult
import com.mobilerag.rag.retrieve.Rrf
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Setup failures (missing eval file, engine init) — message is safe to show as the spike summary. */
class EvalSetupException(message: String) : Exception(message)

/**
 * Persistent regression eval harness (Phase 4): runs the labelled question set in files/eval/
 * through RetrievalEngine in vector and hybrid modes, computes recall@5 / gate accuracy overall
 * and per category plus mean per-channel ms, persists each run to files/eval/history/, and
 * compares against the newest previous run (PASS / REGRESSION, or BASELINE-SET on first run).
 *
 * History file schema:
 *   {"runAt": iso, "config": {evalFile, questions, minVecScore, rrfK, entities, edges, chunks},
 *    "modes": {mode: {recallAt5, recallHits, answerable, gateAccuracy, gateHits, unanswerable,
 *              perCategory: {cat: {hits, total}}, gatePerCategory: {cat: {hits, total}},
 *              meanChannelMs: {channel: ms}}},
 *    "baseline": file|null, "verdicts": {mode: "PASS"|"REGRESSION"|"BASELINE-SET"}}
 */
class EvalRunner(private val context: Context) {

    private data class Entry(
        val q: String,
        val expectDoc: String?,
        val answerable: Boolean,
        val category: String,
    )

    private data class Outcome(val entry: Entry, val mode: String, val hit: Boolean, val result: RetrievalResult)

    data class CategoryStat(val hits: Int, val total: Int)

    data class ModeStats(
        val recall: Double,
        val recallHits: Int,
        val answerable: Int,
        val gateAccuracy: Double,
        val gateHits: Int,
        val unanswerable: Int,
        val perCategory: Map<String, CategoryStat>,
        val gatePerCategory: Map<String, CategoryStat>,
        val meanChannelMs: Map<String, Double>,
    )

    data class RunResult(
        val report: String,
        val historyFileName: String,
        val baselineFileName: String?,
        val verdicts: Map<String, String>,
        val stats: Map<String, ModeStats>,
    )

    suspend fun run(log: (String) -> Unit = {}): RunResult {
        val evalDir = File(context.filesDir, "eval")
        val evalFile = evalDir.listFiles { f -> f.name.matches(EVAL_FILE_RE) }?.maxByOrNull { it.name }
        if (evalFile == null || !evalFile.canRead()) {
            throw EvalSetupException(
                "Missing eval JSON in ${evalDir.absolutePath} — push via: adb push eval-v1.json /data/local/tmp/ && " +
                    "adb shell run-as com.mobilerag sh -c 'mkdir -p files/eval && cp /data/local/tmp/eval-v1.json files/eval/'"
            )
        }
        val entries = parse(JSONArray(evalFile.readText()))
        log("${entries.size} questions from ${evalFile.name} (${entries.count { it.answerable }} answerable)")

        val engine = try {
            EmbeddingGemmaEngine.create(context)
        } catch (t: Throwable) {
            throw EvalSetupException("Engine init failed: ${t.message?.take(200)}")
        }
        try {
            val spaceDir = SpaceManager.activeDir(context)
            val db = RagDatabase(context, File(spaceDir, "rag.db").absolutePath)
            val store = GraphStoreFactory.create(context, spaceDir)
            val retrieval = try {
                RetrievalEngine(engine, VectorStore(db, engine.dimensions), db, store)
            } catch (t: Throwable) {
                db.close()
                throw t
            }

            val outcomes = mutableListOf<Outcome>()
            for (entry in entries) {
                for (mode in MODES) {
                    val result = retrieval.retrieve(entry.q, hybrid = mode == "hybrid")
                    val hit = if (entry.answerable) {
                        result.hits.take(RECALL_K).any { it.docName == entry.expectDoc }
                    } else {
                        result.gated
                    }
                    outcomes += Outcome(entry, mode, hit, result)
                    Log.i(TAG, "[$mode] ${if (hit) "PASS" else "FAIL"} \"${entry.q}\" " +
                        "gated=${result.gated} top5=${result.hits.take(RECALL_K).map { it.docName }}")
                }
            }
            for (f in outcomes.filter { !it.hit }) {
                Log.w(TAG, "failure [${f.mode}] \"${f.entry.q}\" (${f.entry.category}) expect=${f.entry.expectDoc ?: "gate"} " +
                    "gated=${f.result.gated} got=${f.result.hits.take(RECALL_K).joinToString { "${it.docName}(${it.channel})" }}")
            }

            val config = JSONObject()
                .put("evalFile", evalFile.name)
                .put("questions", entries.size)
                .put("minVecScore", RetrievalEngine.MIN_VEC_SCORE.toDouble())
                .put("rrfK", Rrf.DEFAULT_K)
                .put("entities", store.entityCount())
                .put("edges", store.edgeCount())
                .put("chunks", db.listDocuments().sumOf { it.chunkCount })
            db.close()

            val stats = MODES.associateWith { statsFor(outcomes, it) }

            val historyDir = File(evalDir, "history").apply { mkdirs() }
            val baselineFile = historyDir.listFiles { f -> f.name.endsWith(".json") }?.maxByOrNull { it.name }
            val verdicts: Map<String, String>
            var configNote: String? = null
            if (baselineFile == null) {
                verdicts = MODES.associateWith { VERDICT_BASELINE_SET }
            } else {
                val baselineJson = JSONObject(baselineFile.readText())
                val baselineModes = baselineJson.getJSONObject("modes")
                verdicts = MODES.associateWith { mode ->
                    val b = baselineModes.optJSONObject(mode)
                    if (b == null) {
                        VERDICT_PASS
                    } else {
                        val cur = stats.getValue(mode)
                        val recallDrop = b.getDouble("recallAt$RECALL_K") - cur.recall
                        val gateDrop = b.getDouble("gateAccuracy") - cur.gateAccuracy
                        if (recallDrop > REGRESSION_THRESHOLD || gateDrop > REGRESSION_THRESHOLD) VERDICT_REGRESSION else VERDICT_PASS
                    }
                }
                val changed = configDiff(config, baselineJson.optJSONObject("config"))
                if (changed.isNotEmpty()) configNote = "config changed vs baseline: ${changed.joinToString()}"
            }

            val runFileName = "${RUN_TS_FORMAT.format(Date())}.json"
            val root = JSONObject()
                .put("runAt", ISO_FORMAT.format(Date()))
                .put("config", config)
                .put("modes", JSONObject(stats.mapValues { (mode, s) -> statsJson(s) }.toMap()))
                .put("baseline", baselineFile?.name ?: JSONObject.NULL)
                .put("verdicts", JSONObject(verdicts.toMap()))
            File(historyDir, runFileName).writeText(root.toString(2))
            Log.i(TAG, "eval run saved to history/$runFileName verdicts=$verdicts")

            val failures = outcomes.count { !it.hit }
            val report = buildString {
                append(if (baselineFile != null) "baseline: ${baselineFile.name}\n" else "baseline: none (first run)\n")
                for (mode in MODES) {
                    val s = stats.getValue(mode)
                    append("$mode: recall@$RECALL_K %.2f (%d/%d) %s\n".format(s.recall, s.recallHits, s.answerable, verdicts[mode]))
                    append("  gate %.2f (%d/%d)\n".format(s.gateAccuracy, s.gateHits, s.unanswerable))
                    if (s.perCategory.isNotEmpty()) {
                        append("  " + s.perCategory.entries.joinToString(" · ") { "${it.key} ${it.value.hits}/${it.value.total}" } + "\n")
                    }
                }
                append("mean ms (hybrid): " + stats.getValue("hybrid").meanChannelMs.entries
                    .joinToString(" ") { "${it.key} ${it.value.toInt()}" } + "\n")
                if (failures > 0) append("failures: $failures (see logcat $TAG)\n")
                if (configNote != null) append("note: $configNote\n")
                append("saved: history/$runFileName")
            }.trimEnd()
            Log.i(TAG, "\n$report")

            return RunResult(report, runFileName, baselineFile?.name, verdicts, stats)
        } finally {
            engine.close()
        }
    }

    private fun statsFor(outcomes: List<Outcome>, mode: String): ModeStats {
        val inMode = outcomes.filter { it.mode == mode }
        val answerable = inMode.filter { it.entry.answerable }
        val unanswerable = inMode.filter { !it.entry.answerable }
        val perCategory = answerable.groupBy { it.entry.category }
            .mapValues { (_, v) -> CategoryStat(v.count { it.hit }, v.size) }
        val gatePerCategory = unanswerable.groupBy { it.entry.category }
            .mapValues { (_, v) -> CategoryStat(v.count { it.hit }, v.size) }
        val channels = inMode.flatMap { it.result.channelMs.keys }.distinct()
        val meanChannelMs = channels.associateWith { ch ->
            round4(inMode.map { it.result.channelMs[ch] ?: 0L }.average())
        }
        return ModeStats(
            recall = if (answerable.isEmpty()) 0.0 else answerable.count { it.hit }.toDouble() / answerable.size,
            recallHits = answerable.count { it.hit },
            answerable = answerable.size,
            gateAccuracy = if (unanswerable.isEmpty()) 0.0 else unanswerable.count { it.hit }.toDouble() / unanswerable.size,
            gateHits = unanswerable.count { it.hit },
            unanswerable = unanswerable.size,
            perCategory = perCategory,
            gatePerCategory = gatePerCategory,
            meanChannelMs = meanChannelMs,
        )
    }

    private fun statsJson(s: ModeStats): JSONObject = JSONObject()
        .put("recallAt$RECALL_K", round4(s.recall))
        .put("recallHits", s.recallHits)
        .put("answerable", s.answerable)
        .put("gateAccuracy", round4(s.gateAccuracy))
        .put("gateHits", s.gateHits)
        .put("unanswerable", s.unanswerable)
        .put("perCategory", JSONObject(s.perCategory.mapValues { (_, c) ->
            JSONObject().put("hits", c.hits).put("total", c.total) }.toMap()))
        .put("gatePerCategory", JSONObject(s.gatePerCategory.mapValues { (_, c) ->
            JSONObject().put("hits", c.hits).put("total", c.total) }.toMap()))
        .put("meanChannelMs", JSONObject(s.meanChannelMs.toMap()))

    private fun configDiff(cur: JSONObject, base: JSONObject?): List<String> {
        if (base == null) return emptyList()
        return cur.keys().asSequence().filter { k ->
            !base.has(k) || cur.get(k).toString() != base.get(k).toString()
        }.toList()
    }

    private fun parse(json: JSONArray): List<Entry> = buildList {
        for (i in 0 until json.length()) {
            val o = json.getJSONObject(i)
            val answerable = o.optBoolean("answerable", true)
            add(
                Entry(
                    q = o.getString("q"),
                    expectDoc = if (o.isNull("expectDoc")) null else o.getString("expectDoc"),
                    answerable = answerable,
                    category = o.optString("category").ifEmpty { if (answerable) "vector" else "unanswerable" },
                )
            )
        }
    }

    private fun round4(x: Double) = (x * 10000).toLong() / 10000.0

    companion object {
        private const val TAG = "HybridEval"
        private const val RECALL_K = 5
        private const val REGRESSION_THRESHOLD = 0.05
        private val MODES = listOf("vector", "hybrid")
        private val EVAL_FILE_RE = Regex("eval-v\\d+\\.json")
        const val VERDICT_PASS = "PASS"
        const val VERDICT_REGRESSION = "REGRESSION"
        const val VERDICT_BASELINE_SET = "BASELINE-SET"
        private val RUN_TS_FORMAT = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US)
        private val ISO_FORMAT = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
    }
}

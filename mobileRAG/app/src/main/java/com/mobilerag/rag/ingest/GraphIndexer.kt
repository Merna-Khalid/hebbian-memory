package com.mobilerag.rag.ingest

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.mobilerag.core.GraphStore
import com.mobilerag.extract.EntityResolver
import com.mobilerag.extract.Gliner2Extractor
import com.mobilerag.rag.RagDatabase
import com.mobilerag.spikes.QueryTimer
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * Builds the entity graph over indexed chunks: GLiNER2 extraction per chunk, resolution
 * to canonical entities, entity↔chunk mentions, and a MENTIONED_WITH edge for every
 * unordered pair of distinct entities co-occurring in a chunk. Re-running a document is
 * idempotent (its chunk-derived graph data is removed first). Cancellable between chunks.
 *
 * The GLiNER2 ONNX session (~500 MB) is created lazily and ALWAYS closed when a build
 * finishes, fails, or is cancelled.
 */
class GraphIndexer(
    private val context: Context,
    private val db: RagDatabase,
    private val storeProvider: () -> GraphStore,
) {
    data class Progress(
        val docName: String,
        val docsDone: Int,
        val docsTotal: Int,
        val chunksDone: Int,
        val chunksTotal: Int,
    )

    private val _progress = MutableStateFlow<Progress?>(null)
    val progress = _progress.asStateFlow()

    /** Summary of the last build ("3 docs, 41 chunks, 27 entities, 58 edges, 12.3 s"), or an error string. */
    private val _lastStats = MutableStateFlow<String?>(null)
    val lastStats = _lastStats.asStateFlow()

    private var extractor: Gliner2Extractor? = null

    suspend fun buildGraph(docIds: List<Long>) {
        if (docIds.isEmpty()) return
        val modelFile = File(context.filesDir, "${Gliner2Extractor.DEFAULT_DIR}/model_int8.onnx")
        if (!modelFile.canRead()) {
            _lastStats.value = "Graph skipped — missing ${modelFile.name} " +
                "(push gliner2 model files to files/${Gliner2Extractor.DEFAULT_DIR}/)"
            return
        }
        val store = storeProvider()
        val docs = db.listDocuments().filter { it.id in docIds.toSet() }
        var chunksDone = 0
        var edgesAdded = 0
        val entityIds = LinkedHashSet<Long>()
        val startedMs = SystemClock.elapsedRealtime()
        try {
            extractor = Gliner2Extractor.create(context)
            val resolver = EntityResolver(store)
            docs.forEachIndexed { docIndex, doc ->
                val timer = QueryTimer()
                timer.mark("doc")
                val chunks = db.chunksForDocument(doc.id)
                // Idempotent re-index: drop this document's previous graph data first
                store.removeGraphForChunks(chunks.map { it.id })
                _progress.value = Progress(doc.name, docIndex, docs.size, 0, chunks.size)
                chunks.forEachIndexed { seq, chunk ->
                    currentCoroutineContext().ensureActive()
                    val resolved = resolver.resolve(extractor!!.extractEntities(chunk.text))
                    resolved.forEach { (_, entityId) ->
                        store.addMention(entityId, chunk.id)
                        entityIds += entityId
                    }
                    val pairIds = resolved.map { it.second }.distinct().sorted()
                    for (i in pairIds.indices) {
                        for (j in i + 1 until pairIds.size) {
                            store.addEdge(pairIds[i], pairIds[j], REL_MENTIONED_WITH, sourceChunkId = chunk.id)
                            edgesAdded++
                        }
                    }
                    chunksDone++
                    _progress.value = Progress(doc.name, docIndex, docs.size, seq + 1, chunks.size)
                }
                Log.d(TAG, "${doc.name}: ${chunks.size} chunks graphed in ${timer.since("doc")} ms")
            }
            val secs = (SystemClock.elapsedRealtime() - startedMs) / 1000.0
            _lastStats.value = "%d docs, %d chunks, %d entities, %d edges, %.1f s"
                .format(docs.size, chunksDone, entityIds.size, edgesAdded, secs)
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            Log.w(TAG, "graph indexing failed", t)
            _lastStats.value = "Graph indexing failed: ${t.message?.take(120) ?: t.javaClass.simpleName}"
        } finally {
            try {
                extractor?.close()
            } catch (t: Throwable) {
                Log.w(TAG, "extractor close failed", t)
            }
            extractor = null
            _progress.value = null
        }
    }

    /** Wipes the graph and rebuilds it from every indexed document. */
    suspend fun rebuildAll() {
        storeProvider().clear()
        buildGraph(db.listDocuments().map { it.id })
    }

    companion object {
        private const val TAG = "GraphIndexer"
        const val REL_MENTIONED_WITH = "MENTIONED_WITH"
    }
}

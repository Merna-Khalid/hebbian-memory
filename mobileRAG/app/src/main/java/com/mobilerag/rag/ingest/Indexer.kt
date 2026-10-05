package com.mobilerag.rag.ingest

import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.rag.RagDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

/**
 * Chunks and embeds imported documents into RagDatabase. Documents whose content hash is
 * unchanged are skipped; reindexing embeds only the changed ones. Cancellable between chunks.
 */
class Indexer(
    private val db: RagDatabase,
    private val engine: EmbeddingGemmaEngine,
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

    /** Indexes [imports]; returns the ids of documents that were actually reindexed (Unchanged skipped). */
    suspend fun index(imports: List<DocumentImporter.Imported>): List<Long> {
        val reindexed = mutableListOf<Long>()
        try {
            imports.forEachIndexed { i, imp ->
                when (val upsert = db.upsertDocument(imp.name, imp.sha256)) {
                    is RagDatabase.UpsertResult.Unchanged -> Unit
                    is RagDatabase.UpsertResult.Reindex -> {
                        indexDocument(upsert.docId, imp, i, imports.size)
                        reindexed += upsert.docId
                    }
                }
            }
        } finally {
            _progress.value = null
        }
        return reindexed
    }

    private suspend fun indexDocument(docId: Long, imp: DocumentImporter.Imported, docIndex: Int, docsTotal: Int) {
        val text = withContext(Dispatchers.IO) { imp.file.readText() }
        val chunks = Chunker(engine::countTokens).chunk(text)
        _progress.value = Progress(imp.name, docIndex, docsTotal, 0, chunks.size)
        val stored = mutableListOf<RagDatabase.NewChunk>()
        chunks.forEachIndexed { seq, chunk ->
            currentCoroutineContext().ensureActive()
            val embedding = engine.embedDocument(chunk.text, imp.name)
            stored += RagDatabase.NewChunk(seq, chunk.text, chunk.tokenCount, embedding)
            _progress.value = Progress(imp.name, docIndex, docsTotal, seq + 1, chunks.size)
        }
        db.insertChunks(docId, stored)
    }
}

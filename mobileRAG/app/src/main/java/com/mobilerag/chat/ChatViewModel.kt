package com.mobilerag.chat

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.graph.GraphStoreFactory
import com.mobilerag.profile.SpaceManager
import com.mobilerag.rag.CommunityIndexWorker
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.RagEvent
import com.mobilerag.rag.RagPipeline
import com.mobilerag.rag.VectorStore
import com.mobilerag.rag.ingest.DocumentImporter
import com.mobilerag.rag.ingest.GraphIndexer
import com.mobilerag.rag.ingest.Indexer
import com.mobilerag.rag.retrieve.CommunityRetriever
import com.mobilerag.rag.retrieve.RetrievalEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class Citation(val index: Int, val docName: String, val text: String, val score: Float)

data class ChatMessage(
    val id: Long,
    val isUser: Boolean,
    val text: String,
    val citations: List<Citation> = emptyList(),
    val stats: String? = null,
    val streaming: Boolean = false,
    val isError: Boolean = false,
)

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val models: List<String> = emptyList(),
    val selectedModel: String? = null,
    val documents: List<RagDatabase.Document> = emptyList(),
    val indexing: Indexer.Progress? = null,
    val graphIndexing: GraphIndexer.Progress? = null,
    val graphStats: String? = null,
    val retrievalMode: String = RagPipeline.MODE_HYBRID,
    val generationBackend: String = RagPipeline.BACKEND_AUTO,
    val generating: Boolean = false,
    val modelLoading: Boolean = false,
    val error: String? = null,
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(ChatUiState())
    val ui = _ui.asStateFlow()

    private var engine: EmbeddingGemmaEngine? = null
    private var db: RagDatabase? = null
    private var indexer: Indexer? = null
    private var graphIndexer: GraphIndexer? = null
    private var pipeline: RagPipeline? = null
    private var engineInitError: String? = null
    private var nextMsgId = 0L

    init {
        refreshModels()
        refreshDocuments()
        warmUpModel()
        _ui.update {
            it.copy(
                retrievalMode = prefs().getString(RagPipeline.KEY_RETRIEVAL_MODE, RagPipeline.MODE_HYBRID) ?: RagPipeline.MODE_HYBRID,
                generationBackend = prefs().getString(RagPipeline.KEY_GENERATION_BACKEND, RagPipeline.BACKEND_AUTO) ?: RagPipeline.BACKEND_AUTO,
            )
        }
    }

    /** Eagerly loads the selected GGUF when the chat screen opens (AI Edge Gallery-style), so
     *  the first question doesn't pay the mmap cost. Failures stay silent — ask() retries
     *  lazily and surfaces the error there. */
    private fun warmUpModel() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val pipe = ensurePipeline()
                if (pipe.selectedModel() == null) return@launch
                _ui.update { it.copy(modelLoading = true) }
                pipe.preloadModel()
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
            } finally {
                _ui.update { it.copy(modelLoading = false) }
            }
        }
    }

    fun setRetrievalMode(mode: String) {
        pipeline?.setRetrievalMode(mode)
            ?: prefs().edit().putString(RagPipeline.KEY_RETRIEVAL_MODE, mode).apply()
        _ui.update { it.copy(retrievalMode = mode) }
    }

    fun setGenerationBackend(backend: String) {
        pipeline?.setGenerationBackend(backend)
            ?: prefs().edit().putString(RagPipeline.KEY_GENERATION_BACKEND, backend).apply()
        _ui.update { it.copy(generationBackend = backend) }
    }

    /** Lazily creates engine + DB + indexer + pipeline on first use. Throws (with the adb push
     *  instructions) if files/models/embeddinggemma is missing. */
    private fun ensurePipeline(): RagPipeline {
        pipeline?.let { return it }
        engineInitError?.let { throw IllegalStateException(it) }
        val context = getApplication<Application>()
        return try {
            val eng = EmbeddingGemmaEngine.create(context)
            val spaceDir = SpaceManager.activeDir(context)
            val database = RagDatabase(context, File(spaceDir, "rag.db").absolutePath)
            val idx = Indexer(database, eng)
            val graphIdx = GraphIndexer(context, database) { GraphStoreFactory.create(context, spaceDir) }
            val vectors = VectorStore(database, eng.dimensions)
            val retrieval = RetrievalEngine(eng, vectors, database, GraphStoreFactory.create(context, spaceDir))
            engine = eng
            db = database
            indexer = idx
            graphIndexer = graphIdx
            viewModelScope.launch {
                idx.progress.collect { p -> _ui.update { s -> s.copy(indexing = p) } }
            }
            viewModelScope.launch {
                graphIdx.progress.collect { p -> _ui.update { s -> s.copy(graphIndexing = p) } }
            }
            viewModelScope.launch {
                graphIdx.lastStats.collect { stats -> _ui.update { s -> s.copy(graphStats = stats) } }
            }
            RagPipeline(context, eng, retrieval, CommunityRetriever(database, java.io.File(context.filesDir, "eval/comm-scores.log"))).also { pipeline = it }
        } catch (t: Throwable) {
            engineInitError = t.message ?: t.javaClass.simpleName
            throw t
        }
    }

    fun refreshModels() {
        val models = File(getApplication<Application>().filesDir, "models").listFiles()
            ?.filter { it.extension == "gguf" && it.canRead() }
            ?.map { it.name }?.sorted() ?: emptyList()
        val saved = prefs().getString(RagPipeline.KEY_MODEL, null)
        // Same resolution the pipeline loads with — this used to prefer 1.7B while the
        // pipeline (and tutor) loaded 8B, so the picker showed the wrong resident model.
        _ui.update { it.copy(models = models, selectedModel = RagPipeline.pickModel(models, saved)) }
    }

    fun selectModel(name: String) {
        prefs().edit().putString(RagPipeline.KEY_MODEL, name).apply()
        _ui.update { it.copy(selectedModel = name) }
        warmUpModel() // the new selection only becomes resident on load — warm it now
    }

    fun refreshDocuments() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                ensurePipeline()
                val docs = db!!.listDocuments()
                _ui.update { it.copy(documents = docs, error = null) }
            } catch (t: Throwable) {
                _ui.update { it.copy(error = t.message) }
            }
        }
    }

    fun importDocuments(uris: List<Uri>) {
        if (uris.isEmpty()) return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                ensurePipeline()
                val imported = DocumentImporter(getApplication()).import(uris)
                if (imported.isEmpty()) {
                    _ui.update { it.copy(error = "Only .txt and .md files are supported") }
                    return@launch
                }
                indexer!!.index(imported).let { changedDocIds ->
                    if (changedDocIds.isNotEmpty()) {
                        graphIndexer!!.buildGraph(changedDocIds)
                        // The graph changed — schedule a community-summary rebuild (device idle + charging)
                        CommunityIndexWorker.enqueueNow(getApplication())
                    }
                }
                val docs = db!!.listDocuments()
                _ui.update { it.copy(documents = docs) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(error = "Import failed: ${t.message?.take(150)}") }
            }
        }
    }

    fun deleteDocument(id: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val database = db ?: return@runCatching
                // Capture chunk ids before deleting so the graph provenance can be removed
                val chunkIds = database.chunkIdsForDocument(id)
                database.deleteDocument(id)
                if (chunkIds.isNotEmpty()) {
                    GraphStoreFactory.create(getApplication(), SpaceManager.activeDir(getApplication()))
                        .removeGraphForChunks(chunkIds)
                }
            }
            val docs = db?.listDocuments() ?: return@launch
            _ui.update { it.copy(documents = docs) }
        }
    }

    fun ask(question: String) {
        val q = question.trim()
        if (q.isEmpty() || _ui.value.generating) return
        val userMsg = ChatMessage(nextMsgId++, isUser = true, text = q)
        val assistantId = nextMsgId++
        _ui.update {
            it.copy(
                messages = it.messages + userMsg + ChatMessage(assistantId, isUser = false, text = "", streaming = true),
                generating = true,
                error = null,
            )
        }
        viewModelScope.launch {
            try {
                val pipe = withContext(Dispatchers.IO) { ensurePipeline() }
                pipe.ask(q).collect { ev -> onRagEvent(assistantId, ev) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                onRagEvent(assistantId, RagEvent.Failed(t.message ?: t.javaClass.simpleName))
            } finally {
                _ui.update { s ->
                    s.copy(
                        generating = false,
                        messages = s.messages.map { m -> if (m.id == assistantId) m.copy(streaming = false) else m },
                    )
                }
            }
        }
    }

    private fun onRagEvent(assistantId: Long, ev: RagEvent) {
        _ui.update { s ->
            s.copy(messages = s.messages.map { m ->
                if (m.id != assistantId) m else when (ev) {
                    is RagEvent.Retrieved -> m.copy(
                        citations = ev.chunks.mapIndexed { i, c -> Citation(i + 1, c.docName, c.text, c.score ?: 0f) },
                    )
                    is RagEvent.Token -> m.copy(text = m.text + ev.text)
                    is RagEvent.Done -> m.copy(stats = ev.stats.format())
                    is RagEvent.Failed ->
                        if (m.text.isEmpty()) m.copy(text = ev.error, isError = true)
                        else m.copy(text = m.text + "\n\n" + ev.error, isError = true)
                }
            })
        }
    }

    override fun onCleared() {
        runCatching { db?.close() }
        runCatching { engine?.close() }
    }

    private fun prefs() = getApplication<Application>().getSharedPreferences("rag", Context.MODE_PRIVATE)
}

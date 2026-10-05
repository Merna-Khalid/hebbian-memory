package com.mobilerag.graph.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilerag.core.GraphStore
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.graph.GraphStoreFactory
import com.mobilerag.profile.SpaceManager
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.RagPipeline
import com.mobilerag.rag.VectorStore
import com.mobilerag.rag.ingest.GraphIndexer
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

/** One edge group in the entity detail view: edges to the same other entity + relation. */
data class EdgeGroup(
    val otherId: Long,
    val otherName: String,
    val relation: String,
    val count: Int,
)

data class GraphUiState(
    val storeId: String = "",
    val entityCount: Int = 0,
    val edgeCount: Int = 0,
    val communityCount: Int = 0,
    val communityLastRun: Long = 0,
    val summarizing: Boolean = false,
    val communityError: String? = null,
    val lastStats: String? = null,
    val rebuilding: Boolean = false,
    val rebuildProgress: GraphIndexer.Progress? = null,
    val query: String = "",
    val entities: List<GraphStore.Entity> = emptyList(),
    val selected: GraphStore.Entity? = null,
    val selectedEdges: List<EdgeGroup> = emptyList(),
    val selectedChunks: List<String> = emptyList(),
    val detailLoading: Boolean = false,
    /** The constellation tab's 3D layout; built on first open, invalidated by rebuilds. */
    val layout3d: GraphLayout3D.Result? = null,
    val layout3dLoading: Boolean = false,
)

class GraphViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(GraphUiState())
    val ui = _ui.asStateFlow()

    private val spaceDir: File by lazy { SpaceManager.activeDir(getApplication()) }
    private val store: GraphStore by lazy { GraphStoreFactory.create(getApplication(), spaceDir) }
    private val dbLazy = lazy { RagDatabase(getApplication(), File(spaceDir, "rag.db").absolutePath) }
    private val db: RagDatabase by dbLazy
    private val graphIndexer: GraphIndexer by lazy {
        GraphIndexer(getApplication(), db) { GraphStoreFactory.create(getApplication(), spaceDir) }
    }
    private var engine: EmbeddingGemmaEngine? = null
    private var pipeline: RagPipeline? = null

    init {
        viewModelScope.launch {
            graphIndexer.progress.collect { p -> _ui.update { it.copy(rebuildProgress = p) } }
        }
        viewModelScope.launch {
            graphIndexer.lastStats.collect { s -> _ui.update { it.copy(lastStats = s) } }
        }
        refresh()
    }

    fun refresh() {
        viewModelScope.launch(Dispatchers.IO) {
            _ui.update {
                it.copy(
                    storeId = store.id,
                    entityCount = store.entityCount(),
                    edgeCount = store.edgeCount(),
                    communityCount = db.allCommunities().size,
                    communityLastRun = pipeline?.communityLastRun() ?: prefsLastRun(),
                )
            }
            loadEntities()
        }
    }

    /** Builds the 3D layout (once per graph state) off the main thread. */
    fun ensureLayout3D() {
        val s = _ui.value
        if (s.layout3d != null || s.layout3dLoading) return
        _ui.update { it.copy(layout3dLoading = true) }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val entities = store.allEntities(limit = 2000)
                val edges = store.allEdges()
                val communities = db.allCommunities().map { c -> Triple(c.id, c.summary, c.memberIds) }
                val layout = withContext(Dispatchers.Default) { GraphLayout3D.build(entities, edges, communities) }
                _ui.update { it.copy(layout3d = layout, layout3dLoading = false) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(layout3dLoading = false, communityError = "Constellation failed: ${t.message?.take(120)}") }
            }
        }
    }

    fun onQueryChange(query: String) {
        _ui.update { it.copy(query = query) }
        viewModelScope.launch(Dispatchers.IO) { loadEntities() }
    }

    private suspend fun loadEntities() {
        val q = _ui.value.query.trim()
        val entities = if (q.isEmpty()) store.allEntities() else store.findEntities(q)
        _ui.update { it.copy(entities = entities) }
    }

    /** Toggles the detail view for [entity] (same entity again dismisses). */
    fun select(entity: GraphStore.Entity) {
        if (_ui.value.selected?.id == entity.id) {
            _ui.update { it.copy(selected = null, selectedEdges = emptyList(), selectedChunks = emptyList()) }
            return
        }
        _ui.update { it.copy(selected = entity, selectedEdges = emptyList(), selectedChunks = emptyList(), detailLoading = true) }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val edges = store.edgesFor(entity.id)
                val groups = edges
                    .groupBy { edge -> if (edge.fromId == entity.id) edge.toId else edge.fromId }
                    .map { (otherId, group) ->
                        EdgeGroup(
                            otherId = otherId,
                            otherName = store.getEntity(otherId)?.name ?: "#$otherId",
                            relation = group.first().relation,
                            count = group.size,
                        )
                    }
                    .sortedByDescending { it.count }
                val chunkTexts = db.chunkTextsById(store.chunksForEntity(entity.id))
                val previews = chunkTexts.entries.sortedBy { it.key }.map { (_, text) -> preview(text) }
                _ui.update {
                    if (it.selected?.id == entity.id) {
                        it.copy(selectedEdges = groups, selectedChunks = previews, detailLoading = false)
                    } else it
                }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(detailLoading = false) }
            }
        }
    }

    fun rebuild() {
        if (_ui.value.rebuilding) return
        viewModelScope.launch(Dispatchers.IO) {
            _ui.update { it.copy(rebuilding = true, selected = null, selectedEdges = emptyList(), selectedChunks = emptyList()) }
            try {
                graphIndexer.rebuildAll()
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(lastStats = "Rebuild failed: ${t.message?.take(120)}") }
            } finally {
                _ui.update { it.copy(rebuilding = false, layout3d = null) }
            }
            refresh()
        }
    }

    /** Manual community-summary rebuild (Phase 4): same path as CommunityIndexWorker —
     *  RagPipeline loads the LLM, CommunitySummarizer rebuilds, the LLM is unloaded in a
     *  finally. Progress is indeterminate (CommunitySummarizer exposes no per-community hook). */
    fun rebuildCommunities() {
        if (_ui.value.summarizing) return
        viewModelScope.launch(Dispatchers.IO) {
            _ui.update { it.copy(summarizing = true, communityError = null) }
            try {
                val count = ensurePipeline().rebuildCommunities(store, db)
                _ui.update {
                    it.copy(communityCount = count, communityLastRun = prefsLastRun(), layout3d = null)
                }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(communityError = "Community rebuild failed: ${t.message?.take(120)}") }
            } finally {
                _ui.update { it.copy(summarizing = false) }
            }
        }
    }

    private fun ensurePipeline(): RagPipeline {
        pipeline?.let { return it }
        val context = getApplication<Application>()
        val eng = EmbeddingGemmaEngine.create(context).also { engine = it }
        val retrieval = RetrievalEngine(eng, VectorStore(db, eng.dimensions), db, GraphStoreFactory.create(context, spaceDir))
        return RagPipeline(context, eng, retrieval, CommunityRetriever(db, java.io.File(context.filesDir, "eval/comm-scores.log"))).also { pipeline = it }
    }

    private fun prefsLastRun(): Long =
        getApplication<Application>()
            .getSharedPreferences("rag", Context.MODE_PRIVATE)
            .getLong(RagPipeline.KEY_COMMUNITY_LAST_RUN, 0L)

    private fun preview(text: String, maxChars: Int = 200): String {
        val collapsed = text.replace(Regex("\\s+"), " ").trim()
        return if (collapsed.length <= maxChars) collapsed else collapsed.take(maxChars).trimEnd() + "…"
    }

    override fun onCleared() {
        // The shared graph store (GraphStoreFactory singleton) stays open for the process
        if (dbLazy.isInitialized()) runCatching { db.close() }
        runCatching { engine?.close() }
    }
}

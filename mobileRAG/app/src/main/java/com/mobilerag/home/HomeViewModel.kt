package com.mobilerag.home

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilerag.generation.LlmResidency
import com.mobilerag.graph.GraphStoreFactory
import com.mobilerag.hebbian.ArmeHistoryEntry
import com.mobilerag.hebbian.cortical.CorticalConsolidator
import com.mobilerag.hebbian.ui.ConstellationLayout
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.hebbian.ui.HebbianEngineHolder
import com.mobilerag.profile.SpaceManager
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.RagPipeline
import com.mobilerag.settings.AppSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

/**
 * Dashboard state for the Home screen. Counts are nullable: null renders as "—" until the
 * corresponding section has loaded (or when it failed — each section is guarded so one
 * failure, e.g. the tutor engine not being buildable yet, never blanks the rest).
 */
data class HomeUiState(
    val learnerName: String = "Learner",
    val spaceName: String = "",
    val selectedModel: String? = null,
    val residentModel: String = "none",
    val retrievalMode: String = RagPipeline.MODE_VECTOR,
    val backend: String = RagPipeline.BACKEND_AUTO,
    val preset: String = AppSettings.PRESET_QUALITY,
    val graphStoreId: String? = null,
    val hebbianStoreId: String? = null,
    val documents: Int? = null,
    val chunks: Int? = null,
    val entities: Int? = null,
    val edges: Int? = null,
    val concepts: Int? = null,
    val hebbianEdges: Int? = null,
    val fading: Int? = null,
    val arme: List<ArmeHistoryEntry> = emptyList(),
    val refreshing: Boolean = false,
    /** The strongest part of the graph, laid out (Home's constellation). */
    val constellation: ConstellationLayout.Result? = null,
    /** When the first concept was formed — the entity's birthday. */
    val bornAt: Long? = null,
    /** Concepts with at least one link / met at least twice (the gauges). */
    val linked: Int? = null,
    val recalled: Int? = null,
    /** A cortical consolidation is running right now. */
    val consolidating: Boolean = false,
    val lastConsolidatedAt: Long? = null,
)

class HomeViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(HomeUiState())
    val ui = _ui.asStateFlow()

    init {
        refresh()
    }

    /** Reloads all dashboard data on Dispatchers.IO. Store factories and the tutor engine
     *  open lazily here (never on the main thread); all are process-wide singletons, so a
     *  refresh after the first is cheap. Overlapping refreshes are skipped. */
    fun refresh() {
        if (_ui.value.refreshing) return
        viewModelScope.launch(Dispatchers.IO) {
            _ui.update { it.copy(refreshing = true) }
            try {
                val context = getApplication<Application>()
                val settings = AppSettings(context)
                val space = SpaceManager.activeSpace(context)
                val spaceDir = SpaceManager.activeDir(context)
                _ui.update {
                    it.copy(
                        learnerName = settings.learnerName.trim().ifEmpty { "Learner" },
                        spaceName = space.name,
                        selectedModel = selectedModelName(context),
                        residentModel = LlmResidency.loadedModel?.let { path -> File(path).name } ?: "none",
                        retrievalMode = settings.retrievalMode,
                        backend = settings.generationBackend,
                        preset = settings.performancePreset,
                    )
                }

                guard {
                    val graph = GraphStoreFactory.create(context, spaceDir)
                    val entityCount = graph.entityCount()
                    val edgeCount = graph.edgeCount()
                    _ui.update { it.copy(graphStoreId = graph.id, entities = entityCount, edges = edgeCount) }
                }
                guard {
                    val db = RagDatabase(context, File(spaceDir, "rag.db").absolutePath)
                    try {
                        val docs = db.listDocuments()
                        _ui.update { it.copy(documents = docs.size, chunks = docs.sumOf { d -> d.chunkCount }) }
                    } finally {
                        db.close()
                    }
                }
                guard {
                    val store = HebbianStoreFactory.create(context, spaceDir)
                    val conceptCount = store.listConcepts(limit = 5000).size
                    val hebbianEdgeCount = store.getGlobalGraph(minWeight = 0.0).edges.size
                    _ui.update {
                        it.copy(
                            hebbianStoreId = store.id,
                            concepts = conceptCount,
                            hebbianEdges = hebbianEdgeCount,
                        )
                    }
                }
                guard {
                    // Shared tutor engine (same one the Tutor/Practice/Memory tabs use);
                    // may not be buildable yet (missing embedding model) — show "—" then.
                    val engine = HebbianEngineHolder.tutor(context)
                    val graph = engine.graphData(minWeight = 0.0, limit = 500, includeIsolated = true)
                    val total = graph.nodes.size
                    // "Fading" = recall probability below one half: more likely forgotten than
                    // remembered. (This used to be the size of a 5-row practice query.)
                    val fadingCount = engine.fadingConcepts(limit = maxOf(total, 1)).count { f -> f.recallProb < 0.5 }
                    val linkedIds = HashSet<String>()
                    graph.edges.forEach { e -> linkedIds += e.source; linkedIds += e.target }
                    val layout = ConstellationLayout.build(
                        labels = graph.nodes.associate { n -> n.nodeId to n.label },
                        edges = graph.edges.map { e -> Triple(e.source, e.target, e.hebbWeight) },
                    )
                    val cortex = CorticalConsolidator.forSpace(spaceDir)
                    val arme = engine.armeHistory(limit = 5).asReversed()
                    _ui.update {
                        it.copy(
                            fading = fadingCount,
                            arme = arme,
                            constellation = layout,
                            bornAt = graph.nodes.minOfOrNull { n -> n.createdAt },
                            linked = graph.nodes.count { n -> n.nodeId in linkedIds },
                            recalled = graph.nodes.count { n -> n.activationCount >= 2 },
                            consolidating = cortex.isRunning,
                            lastConsolidatedAt = cortex.lastReport?.at,
                        )
                    }
                }
            } finally {
                _ui.update { it.copy(refreshing = false) }
            }
        }
    }

    /** Mirrors RagPipeline.selectedModel's fallback order (saved → 8B → 1.7B → first). */
    private fun selectedModelName(context: Context): String? {
        val models = File(context.filesDir, "models").listFiles()
            ?.filter { it.extension == "gguf" && it.canRead() }
            ?.sortedBy { it.name } ?: return null
        val saved = context.getSharedPreferences("rag", Context.MODE_PRIVATE)
            .getString(RagPipeline.KEY_MODEL, null)
        return (models.firstOrNull { it.name == saved }
            ?: models.firstOrNull { "8B" in it.name }
            ?: models.firstOrNull { "1.7B" in it.name }
            ?: models.firstOrNull())?.name
    }

    private suspend fun guard(block: suspend () -> Unit) {
        try {
            block()
        } catch (t: CancellationException) {
            throw t
        } catch (_: Throwable) {
            // leave that section at its previous value ("—" until first success)
        }
    }
}

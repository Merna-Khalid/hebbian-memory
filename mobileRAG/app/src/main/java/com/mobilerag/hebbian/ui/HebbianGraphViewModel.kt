package com.mobilerag.hebbian.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilerag.hebbian.GraphEdgeView
import com.mobilerag.hebbian.GraphNodeView
import com.mobilerag.hebbian.TutorEngine
import com.mobilerag.hebbian.store.ConceptNode
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class HebbianGraphUiState(
    val building: Boolean = true,
    val engineReady: Boolean = false,
    val loading: Boolean = false,
    val minWeight: Float = 0f,
    val nodes: List<GraphNodeView> = emptyList(),
    val edgeCount: Int = 0,
    /** Summed weight of each concept's links above the filter (both directions). */
    val strength: Map<String, Double> = emptyMap(),
    /** Distinct neighbours of each concept above the filter. */
    val linkCount: Map<String, Int> = emptyMap(),
    /** node id → label, so link rows can name the other end. */
    val labels: Map<String, String> = emptyMap(),
    val selectedId: String? = null,
    val detailNode: ConceptNode? = null,
    val detailEdges: List<GraphEdgeView> = emptyList(),
    val detailLoading: Boolean = false,
    val error: String? = null,
)

class HebbianGraphViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(HebbianGraphUiState())
    val ui = _ui.asStateFlow()

    private var engine: TutorEngine? = null

    init {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                engine = HebbianEngineHolder.tutor(getApplication())
                _ui.update { it.copy(building = false, engineReady = true) }
                refresh()
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(building = false, error = "Engine build failed: ${t.message?.take(150)}") }
            }
        }
    }

    fun onMinWeightChange(v: Float) {
        _ui.update { it.copy(minWeight = v) }
    }

    /** Re-query with the current slider value (called when the drag ends). */
    fun applyFilter() {
        viewModelScope.launch(Dispatchers.IO) { refresh() }
    }

    fun refresh() {
        viewModelScope.launch(Dispatchers.IO) {
            val eng = engine ?: return@launch
            _ui.update { it.copy(loading = true, error = null) }
            try {
                val data = eng.graphData(
                    minWeight = _ui.value.minWeight.toDouble(),
                    // With a filter on, show only concepts that have a link above it.
                    includeIsolated = _ui.value.minWeight <= 0f,
                )
                val strength = HashMap<String, Double>()
                val neighbours = HashMap<String, MutableSet<String>>()
                for (e in data.edges) {
                    strength[e.source] = (strength[e.source] ?: 0.0) + e.hebbWeight
                    strength[e.target] = (strength[e.target] ?: 0.0) + e.hebbWeight
                    neighbours.getOrPut(e.source) { HashSet() } += e.target
                    neighbours.getOrPut(e.target) { HashSet() } += e.source
                }
                _ui.update {
                    it.copy(
                        nodes = data.nodes.sortedWith(
                            compareByDescending<GraphNodeView> { n -> strength[n.nodeId] ?: 0.0 }
                                .thenByDescending { n -> n.activationCount },
                        ),
                        edgeCount = data.edges.size,
                        strength = strength,
                        linkCount = neighbours.mapValues { (_, v) -> v.size },
                        labels = data.nodes.associate { n -> n.nodeId to n.label },
                        loading = false,
                    )
                }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(loading = false, error = "Graph load failed: ${t.message?.take(150)}") }
            }
        }
    }

    /** Toggles the detail view for [nodeId] (same node again dismisses). */
    fun select(nodeId: String) {
        if (_ui.value.selectedId == nodeId) {
            _ui.update { it.copy(selectedId = null, detailNode = null, detailEdges = emptyList()) }
            return
        }
        _ui.update { it.copy(selectedId = nodeId, detailNode = null, detailEdges = emptyList(), detailLoading = true) }
        viewModelScope.launch(Dispatchers.IO) {
            val eng = engine ?: return@launch
            try {
                val detail = eng.nodeDetail(nodeId)
                // GraphNodeView omits recentEvents/updatedAt/meanRT — read the full node
                // from the engine's store (TutorEngine.store is public) for the detail sheet.
                val full = eng.store.getConcept(nodeId)
                _ui.update {
                    if (it.selectedId == nodeId) {
                        it.copy(
                            detailNode = full,
                            detailEdges = detail?.edges ?: emptyList(),
                            detailLoading = false,
                        )
                    } else it
                }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(detailLoading = false) }
            }
        }
    }
}

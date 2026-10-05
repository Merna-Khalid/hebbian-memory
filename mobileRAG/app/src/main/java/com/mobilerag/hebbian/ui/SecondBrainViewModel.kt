package com.mobilerag.hebbian.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilerag.hebbian.HebbianEngineFactory
import com.mobilerag.hebbian.TutorEngine
import com.mobilerag.hebbian.signals.SimulatedSignalSource
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

data class IngestLogRow(
    val index: Int,
    val scenario: String,
    val utterance: String,
    val quadrant: String,
    val mT: Double,
    val gated: Boolean,
    val attention: Double,
    val hrBpm: Double?,
    val deltaWMean: Double,
    val concepts: List<String>,
)

data class SecondBrainUiState(
    val building: Boolean = true,
    val engineReady: Boolean = false,
    val modelLoading: Boolean = false,
    val modelLoaded: Boolean = false,
    val stepping: Boolean = false,
    val playing: Boolean = false,
    val exhausted: Boolean = false,
    val scenarioLabel: String = "—",
    val hrBpm: Double? = null,
    val attention: Double? = null,
    val rows: List<IngestLogRow> = emptyList(),
    val error: String? = null,
)

class SecondBrainViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(SecondBrainUiState())
    val ui = _ui.asStateFlow()

    // Own engine instance (second_brain domain/preset) — shares the on-disk stores with
    // the tutor engine via the HebbianStoreFactory singleton, as in the Python server.
    private val sim = SimulatedSignalSource(seed = 7)
    private var engine: TutorEngine? = null
    private var sessionId: String? = null
    private var playJob: Job? = null
    private var nextRowIndex = 1

    init {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val eng = HebbianEngineFactory.build(getApplication(), "second_brain", signalSource = sim)
                val sid = eng.startSession("ambient")
                engine = eng
                sessionId = sid
                _ui.update { it.copy(building = false, engineReady = true) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(building = false, error = "Engine build failed: ${t.message?.take(150)}") }
            }
        }
    }

    fun step() {
        if (_ui.value.stepping || _ui.value.playing || _ui.value.exhausted) return
        viewModelScope.launch(Dispatchers.IO) { stepInternal() }
    }

    fun togglePlay() {
        if (_ui.value.playing) {
            playJob?.cancel()
            _ui.update { it.copy(playing = false) }
            return
        }
        if (_ui.value.exhausted || !_ui.value.engineReady) return
        _ui.update { it.copy(playing = true) }
        playJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive && !_ui.value.exhausted) {
                stepInternal()
                delay(PLAY_DELAY_MS)
            }
            _ui.update { it.copy(playing = false) }
        }
    }

    fun reset() {
        playJob?.cancel()
        val eng = engine ?: return
        viewModelScope.launch(Dispatchers.IO) {
            _ui.update { it.copy(playing = false, stepping = true) }
            try {
                sim.reset()
                sessionId?.let { runCatching { eng.endSession(it) } }
                val sid = eng.startSession("reset")
                sessionId = sid
                nextRowIndex = 1
                _ui.update {
                    SecondBrainUiState(
                        building = false,
                        engineReady = true,
                        modelLoaded = it.modelLoaded,
                    )
                }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(stepping = false, error = "Reset failed: ${t.message?.take(150)}") }
            }
        }
    }

    private suspend fun stepInternal() {
        val eng = engine ?: return
        val sid = sessionId ?: return
        _ui.update { it.copy(stepping = true, error = null) }
        try {
            val utt = sim.nextUtterance()
            if (utt == null) {
                _ui.update { it.copy(exhausted = true, playing = false) }
                return
            }
            val physio = sim.currentPhysio()
            _ui.update {
                it.copy(
                    scenarioLabel = sim.currentLabel,
                    hrBpm = physio.hrBpm,
                    attention = physio.attention,
                )
            }
            if (!_ui.value.modelLoaded) {
                // Extraction runs through the resident LLM — load it on first step.
                _ui.update { it.copy(modelLoading = true) }
                HebbianEngineHolder.ensureModel(getApplication(), eng)
                _ui.update { it.copy(modelLoading = false, modelLoaded = true) }
            }
            val result = eng.ingestAmbient(utt, physio, sid)
            val row = IngestLogRow(
                index = nextRowIndex++,
                scenario = sim.currentLabel,
                utterance = result.utterance,
                quadrant = result.arme?.quadrant ?: "—",
                mT = result.arme?.mT ?: 1.0,
                gated = result.arme?.gated ?: false,
                attention = result.attention,
                hrBpm = result.hrBpm,
                deltaWMean = result.arme?.deltaWMean ?: 0.0,
                concepts = result.concepts.map { it.label },
            )
            _ui.update { it.copy(rows = it.rows + row) }
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            _ui.update { it.copy(error = "Ingest failed: ${t.message?.take(150)}", playing = false) }
        } finally {
            _ui.update { it.copy(stepping = false, modelLoading = false) }
        }
    }

    override fun onCleared() {
        playJob?.cancel()
        val eng = engine
        val sid = sessionId
        if (eng != null && sid != null) {
            runCatching { runBlocking { eng.endSession(sid) } }
        }
    }

    companion object {
        private const val PLAY_DELAY_MS = 700L
    }
}

package com.mobilerag.hebbian.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilerag.hebbian.FadingConceptView
import com.mobilerag.hebbian.PracticeGradeResult
import com.mobilerag.hebbian.PracticeOffer
import com.mobilerag.hebbian.TutorEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

data class PracticeUiState(
    val building: Boolean = true,
    val engineReady: Boolean = false,
    val modelLoading: Boolean = false,
    val modelLoaded: Boolean = false,
    val fading: List<FadingConceptView> = emptyList(),
    val offer: PracticeOffer? = null,
    val grade: PracticeGradeResult? = null,
    val busy: Boolean = false,
    val noExercise: Boolean = false,
    val error: String? = null,
)

class PracticeViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(PracticeUiState())
    val ui = _ui.asStateFlow()

    private var engine: TutorEngine? = null
    private var sessionId: String? = null

    init {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val eng = HebbianEngineHolder.tutor(getApplication())
                val sid = eng.startSession("practice")
                engine = eng
                sessionId = sid
                _ui.update { it.copy(building = false, engineReady = true) }
                refreshFading()
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(building = false, error = "Engine build failed: ${t.message?.take(150)}") }
            }
        }
    }

    fun refresh() {
        viewModelScope.launch(Dispatchers.IO) { refreshFading() }
    }

    private suspend fun refreshFading() {
        val eng = engine ?: return
        runCatching { eng.fadingConcepts(limit = 8) }.onSuccess { fading ->
            _ui.update { it.copy(fading = fading) }
        }
    }

    fun nextExercise() {
        val eng = engine ?: return
        val sid = sessionId ?: return
        if (_ui.value.busy) return
        viewModelScope.launch {
            _ui.update { it.copy(busy = true, noExercise = false, offer = null, grade = null, error = null) }
            try {
                if (!_ui.value.modelLoaded) {
                    _ui.update { it.copy(modelLoading = true) }
                    withContext(Dispatchers.IO) { HebbianEngineHolder.ensureModel(getApplication(), eng) }
                    _ui.update { it.copy(modelLoading = false, modelLoaded = true) }
                }
                val offer = withContext(Dispatchers.IO) { eng.nextPractice(sid) }
                _ui.update { it.copy(offer = offer, noExercise = offer == null) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(error = "Exercise failed: ${t.message?.take(150)}") }
            } finally {
                _ui.update { it.copy(busy = false, modelLoading = false) }
            }
            refreshFading()
        }
    }

    fun submit(answer: String) {
        val a = answer.trim()
        val eng = engine ?: return
        val sid = sessionId ?: return
        if (a.isEmpty() || _ui.value.busy || _ui.value.offer == null) return
        viewModelScope.launch {
            _ui.update { it.copy(busy = true, error = null) }
            try {
                val result = withContext(Dispatchers.IO) { eng.submitPracticeAnswer(sid, a) }
                if (result == null) {
                    _ui.update { it.copy(error = "No pending exercise — request a new one.", offer = null) }
                } else {
                    _ui.update { it.copy(grade = result, offer = null) }
                }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(error = "Grading failed: ${t.message?.take(150)}") }
            } finally {
                _ui.update { it.copy(busy = false) }
            }
            refreshFading()
        }
    }

    override fun onCleared() {
        val eng = engine
        val sid = sessionId
        if (eng != null && sid != null) {
            runCatching { runBlocking { eng.endSession(sid) } }
        }
    }
}

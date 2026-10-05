package com.mobilerag.hebbian.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilerag.hebbian.ArmeSummary
import com.mobilerag.hebbian.ConceptSummary
import com.mobilerag.hebbian.TutorEngine
import com.mobilerag.hebbian.TutorEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class TutorMessage(
    val id: Long,
    val isUser: Boolean,
    val text: String,
    val streaming: Boolean = false,
    val isError: Boolean = false,
    val arme: ArmeSummary? = null,
    val concepts: List<ConceptSummary> = emptyList(),
    val lang: String? = null,
)

data class TutorUiState(
    val building: Boolean = true,
    val engineReady: Boolean = false,
    val modelLoading: Boolean = false,
    val modelLoaded: Boolean = false,
    val modelLoadMs: Long = 0,
    val messages: List<TutorMessage> = emptyList(),
    val generating: Boolean = false,
    /** Reply is complete; the turn's extraction/ingest LLM call is still running. */
    val memorizing: Boolean = false,
    val error: String? = null,
)

class TutorViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(TutorUiState())
    val ui = _ui.asStateFlow()

    private var engine: TutorEngine? = null
    private var sessionId: String? = null
    private var nextMsgId = 0L

    init {
        // Engine build opens the Hebbian stores + the EmbeddingGemma ONNX session — off main.
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val eng = HebbianEngineHolder.tutor(getApplication())
                val sid = eng.startSession("manual")
                engine = eng
                sessionId = sid
                _ui.update { it.copy(building = false, engineReady = true) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(building = false, error = "Engine build failed: ${t.message?.take(150)}") }
                return@launch
            }
            // Warm-up (AI Edge Gallery-style): load the GGUF as soon as the chat screen is
            // shown, so the first message doesn't pay the mmap + prefill cost. Failures are
            // left to the lazy path in send() to surface.
            val eng = engine ?: return@launch
            _ui.update { it.copy(modelLoading = true) }
            try {
                val ms = HebbianEngineHolder.ensureModel(getApplication(), eng)
                _ui.update { it.copy(modelLoading = false, modelLoaded = true, modelLoadMs = ms) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                _ui.update { it.copy(modelLoading = false) }
            }
        }
    }

    fun send(text: String) {
        val q = text.trim()
        // memorizing counts as busy: extraction extends the same KV session, so a
        // concurrent chat turn would corrupt the conversation state.
        if (q.isEmpty() || _ui.value.generating || _ui.value.memorizing) return
        val eng = engine ?: return
        val sid = sessionId ?: return
        val userMsg = TutorMessage(nextMsgId++, isUser = true, text = q)
        val assistantId = nextMsgId++
        _ui.update {
            it.copy(
                messages = it.messages + userMsg + TutorMessage(assistantId, isUser = false, text = "", streaming = true),
                generating = true,
                memorizing = false,
                error = null,
            )
        }
        viewModelScope.launch {
            try {
                // Always re-assert: it's a no-op when the tutor persona is resident, a prompt
                // swap when Khepri chat took the engine since the last turn, and a reload only
                // after an unload (memory pressure / Settings). It also picks up a changed
                // learner profile. Skipping it on `modelLoaded` let the tutor answer under the
                // RAG grounding prompt after a tab switch.
                _ui.update { it.copy(modelLoading = true) }
                val ms = withContext(Dispatchers.IO) { HebbianEngineHolder.ensureModel(getApplication(), eng) }
                _ui.update { it.copy(modelLoading = false, modelLoaded = true, modelLoadMs = if (ms > 0) ms else it.modelLoadMs) }
                eng.chat(q, sid).collect { ev -> onTutorEvent(assistantId, ev) }
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                onTutorEvent(assistantId, TutorEvent.Failed(t.message ?: t.javaClass.simpleName))
            } finally {
                _ui.update { s ->
                    s.copy(
                        generating = false,
                        memorizing = false,
                        modelLoading = false,
                        messages = s.messages.map { m -> if (m.id == assistantId) m.copy(streaming = false) else m },
                    )
                }
            }
        }
    }

    private fun onTutorEvent(assistantId: Long, ev: TutorEvent) {
        // ReplyComplete finalizes the bubble and re-enables the composer while the engine
        // is still extracting/ingesting — that phase is surfaced as "memorizing", not typing.
        if (ev is TutorEvent.ReplyComplete) {
            _ui.update { s ->
                s.copy(
                    generating = false,
                    memorizing = ev.memorizing,
                    messages = s.messages.map { m ->
                        if (m.id == assistantId) m.copy(text = ev.response.ifEmpty { m.text }, streaming = false) else m
                    },
                )
            }
            return
        }
        if (ev is TutorEvent.Done || ev is TutorEvent.Failed) {
            _ui.update { it.copy(memorizing = false) }
        }
        _ui.update { s ->
            s.copy(messages = s.messages.map { m ->
                if (m.id != assistantId) m else when (ev) {
                    is TutorEvent.Token -> m.copy(text = m.text + ev.text)
                    is TutorEvent.Done -> m.copy(
                        text = ev.result.response.ifEmpty { m.text },
                        arme = ev.result.arme,
                        concepts = ev.result.concepts,
                        lang = ev.result.lang,
                    )
                    is TutorEvent.Failed ->
                        if (m.text.isEmpty()) m.copy(text = ev.error, isError = true)
                        else m.copy(text = m.text + "\n\n" + ev.error, isError = true)
                    is TutorEvent.Retrieved -> m
                    is TutorEvent.ReplyComplete -> m // handled above (early return)
                }
            })
        }
    }

    override fun onCleared() {
        // The engine itself lives in HebbianEngineHolder (shared with Practice/Memory) —
        // only this screen's session ends here. endSession may flush buffered turns through
        // the LLM, so it goes to the holder scope, never a blocking call on the main thread.
        val eng = engine
        val sid = sessionId
        if (eng != null && sid != null) {
            HebbianEngineHolder.scope.launch { runCatching { eng.endSession(sid) } }
        }
    }
}

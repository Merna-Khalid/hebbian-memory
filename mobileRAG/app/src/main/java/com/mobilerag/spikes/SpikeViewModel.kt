package com.mobilerag.spikes

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class SpikeUiState(
    val spike: Spike,
    val running: Boolean = false,
    val result: SpikeResult? = null,
    val logLines: List<String> = emptyList(),
)

class SpikeViewModel(app: Application) : AndroidViewModel(app) {

    private val _states = MutableStateFlow(SpikeRegistry.all.map { SpikeUiState(it) })
    val states = _states.asStateFlow()

    fun runSpike(spikeId: String) {
        val context = getApplication<Application>()
        viewModelScope.launch(Dispatchers.Default) {
            update(spikeId) { it.copy(running = true, result = null, logLines = emptyList()) }
            val result = try {
                SpikeRegistry.byId(spikeId).run(context) { line ->
                    update(spikeId) { it.copy(logLines = it.logLines + line) }
                }
            } catch (t: Throwable) {
                SpikeResult(
                    passed = false,
                    summary = "Crashed: ${t.javaClass.simpleName}",
                    error = t.stackTraceToString().take(2000),
                )
            }
            update(spikeId) {
                it.copy(
                    running = false,
                    result = result,
                    logLines = it.logLines + listOfNotNull(
                        if (result.passed) "PASS: ${result.summary}" else "FAIL: ${result.summary}",
                        result.error?.let { err -> "ERROR: $err" },
                    ) + result.metrics.map { (k, v) -> "  $k: $v" },
                )
            }
        }
    }

    private fun update(spikeId: String, f: (SpikeUiState) -> SpikeUiState) {
        _states.update { list -> list.map { if (it.spike.id == spikeId) f(it) else it } }
    }
}

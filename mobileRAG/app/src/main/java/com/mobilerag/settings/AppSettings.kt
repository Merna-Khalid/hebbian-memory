package com.mobilerag.settings

import android.content.Context
import android.content.SharedPreferences
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import com.mobilerag.rag.RagPipeline
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

class AppSettings(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val ragPrefs = context.getSharedPreferences(RAG_PREFS, Context.MODE_PRIVATE)

    private val _changes = MutableStateFlow(0L)

    /** Bumped on every write; collect this in Compose to recompose when settings change.
     *  Driven by a SharedPreferences listener so writes from ANY AppSettings instance
     *  (e.g. a settings screen deep in the nav graph) recompose the root theme too. */
    val changes: StateFlow<Long> = _changes

    // Held in a field on purpose: SharedPreferences keeps listeners in a WeakHashMap, so a
    // bare lambda passed to register…() is garbage-collected at the next GC and silently stops
    // firing — why theme changes applied only sometimes.
    private val prefsListener = SharedPreferences.OnSharedPreferenceChangeListener { _, _ ->
        _changes.update { it + 1 }
    }

    init {
        prefs.registerOnSharedPreferenceChangeListener(prefsListener)
    }

    var themeMode: String
        get() = prefs.getString(KEY_THEME_MODE, THEME_SYSTEM) ?: THEME_SYSTEM
        set(value) = put { putString(KEY_THEME_MODE, value) }

    var themeAccent: String
        get() = prefs.getString(KEY_THEME_ACCENT, ACCENT_DYNAMIC) ?: ACCENT_DYNAMIC
        set(value) = put { putString(KEY_THEME_ACCENT, value) }

    var performancePreset: String
        get() = prefs.getString(KEY_PERFORMANCE_PRESET, PRESET_QUALITY) ?: PRESET_QUALITY
        set(value) = put { putString(KEY_PERFORMANCE_PRESET, value) }

    var learnerName: String
        get() = prefs.getString(KEY_LEARNER_NAME, "") ?: ""
        set(value) = put { putString(KEY_LEARNER_NAME, value) }

    var learnerLevel: String
        get() = prefs.getString(KEY_LEARNER_LEVEL, LEVEL_N5) ?: LEVEL_N5
        set(value) = put { putString(KEY_LEARNER_LEVEL, value) }

    var learnerGoals: String
        get() = prefs.getString(KEY_LEARNER_GOALS, "") ?: ""
        set(value) = put { putString(KEY_LEARNER_GOALS, value) }

    var activeSpaceId: String
        get() = prefs.getString(KEY_ACTIVE_SPACE_ID, DEFAULT_SPACE_ID) ?: DEFAULT_SPACE_ID
        set(value) = put { putString(KEY_ACTIVE_SPACE_ID, value) }

    // Individual overrides of what a performance preset would set: each marks the preset "custom".

    var selectedModel: String?
        get() = ragPrefs.getString(RagPipeline.KEY_MODEL, null)
        set(value) {
            ragPrefs.edit().putString(RagPipeline.KEY_MODEL, value).apply()
            performancePreset = PRESET_CUSTOM
        }

    var retrievalMode: String
        get() = ragPrefs.getString(RagPipeline.KEY_RETRIEVAL_MODE, RagPipeline.MODE_VECTOR)
            ?: RagPipeline.MODE_VECTOR
        set(value) {
            ragPrefs.edit().putString(RagPipeline.KEY_RETRIEVAL_MODE, value).apply()
            performancePreset = PRESET_CUSTOM
        }

    var generationBackend: String
        get() = ragPrefs.getString(RagPipeline.KEY_GENERATION_BACKEND, RagPipeline.BACKEND_AUTO)
            ?: RagPipeline.BACKEND_AUTO
        set(value) {
            ragPrefs.edit().putString(RagPipeline.KEY_GENERATION_BACKEND, value).apply()
            performancePreset = PRESET_CUSTOM
        }

    /**
     * Extra system-prompt block describing the learner, or null when everything is default
     * (empty name, N5 level, empty goals) so tutors stay unprompted.
     */
    fun learnerPromptBlock(): String? {
        val name = learnerName.trim()
        val level = learnerLevel
        val goals = learnerGoals.trim()
        if (name.isEmpty() && level == LEVEL_N5 && goals.isEmpty()) return null

        val parts = mutableListOf<String>()
        if (name.isNotEmpty()) parts += name
        if (level != LEVEL_N5) parts += "level $level"
        else if (name.isNotEmpty()) parts += "level $LEVEL_N5"
        return buildString {
            if (parts.isEmpty()) append("Learner.") else append("Learner: ")
                .append(parts.joinToString(", ")).append(".")
            if (goals.isNotEmpty()) append(" Goals: ").append(goals).append(".")
            if (parts.isNotEmpty()) {
                append(" Adjust explanations, vocabulary, and example difficulty to this level.")
            }
        }
    }

    /**
     * Applies a named performance preset to the RAG pipeline's own preferences.
     * The model write is skipped when no installed model name contains the preset's token.
     * "custom" is a no-op (reached only by editing model/mode/backend individually).
     */
    fun applyPerformancePreset(
        preset: String,
        ragPrefs: SharedPreferences,
        installedModels: List<String>,
    ) {
        val (modelToken, mode, backend) = when (preset) {
            PRESET_QUALITY -> Triple("8B", RagPipeline.MODE_HYBRID, RagPipeline.BACKEND_LLAMACPP)
            PRESET_BALANCED -> Triple("1.7B", RagPipeline.MODE_HYBRID, RagPipeline.BACKEND_LLAMACPP)
            PRESET_SPEED -> Triple("1.7B", RagPipeline.MODE_VECTOR, RagPipeline.BACKEND_LLAMACPP)
            else -> return
        }
        ragPrefs.edit()
            .putString(RagPipeline.KEY_RETRIEVAL_MODE, mode)
            .putString(RagPipeline.KEY_GENERATION_BACKEND, backend)
            .apply()
        installedModels.firstOrNull { modelToken in it }?.let { model ->
            ragPrefs.edit().putString(RagPipeline.KEY_MODEL, model).apply()
        }
        performancePreset = preset
    }

    private fun put(edit: SharedPreferences.Editor.() -> Unit) {
        prefs.edit().apply(edit).apply()
        _changes.update { it + 1 }
    }

    companion object {
        private const val PREFS = "app_settings"
        private const val RAG_PREFS = "rag"

        const val KEY_THEME_MODE = "themeMode"
        const val KEY_THEME_ACCENT = "themeAccent"
        const val KEY_PERFORMANCE_PRESET = "performancePreset"
        const val KEY_LEARNER_NAME = "learnerName"
        const val KEY_LEARNER_LEVEL = "learnerLevel"
        const val KEY_LEARNER_GOALS = "learnerGoals"
        const val KEY_ACTIVE_SPACE_ID = "activeSpaceId"

        const val THEME_SYSTEM = "system"
        const val THEME_LIGHT = "light"
        const val THEME_DARK = "dark"

        const val ACCENT_DYNAMIC = "dynamic"
        const val ACCENT_INDIGO = "indigo"
        const val ACCENT_SAKURA = "sakura"
        const val ACCENT_MATCHA = "matcha"

        const val PRESET_QUALITY = "quality"
        const val PRESET_BALANCED = "balanced"
        const val PRESET_SPEED = "speed"
        const val PRESET_CUSTOM = "custom"

        const val LEVEL_N5 = "N5"
        const val LEVEL_N4 = "N4"
        const val LEVEL_N3 = "N3"
        const val LEVEL_N2 = "N2"
        const val LEVEL_N1 = "N1"

        const val DEFAULT_SPACE_ID = "personal"
    }
}

@Composable
fun rememberAppSettings(): AppSettings {
    val context = LocalContext.current
    val settings = remember { AppSettings(context.applicationContext) }
    // Reading .value during composition is what subscribes the caller to settings writes —
    // merely calling collectAsState() without reading it never triggered a recomposition, so
    // the root theme only picked up changes when something else happened to recompose it.
    // (This function returns a value, so it isn't its own restart scope: the read lands in
    // the caller's scope.)
    settings.changes.collectAsState().value
    return settings
}

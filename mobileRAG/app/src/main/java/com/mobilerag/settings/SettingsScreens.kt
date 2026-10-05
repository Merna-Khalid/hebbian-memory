package com.mobilerag.settings

import android.app.Activity
import android.content.Context
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.arm.aichat.AiChat
import com.mobilerag.generation.LlmResidency
import com.mobilerag.graph.GraphStoreFactory
import com.mobilerag.hebbian.DuplicateMerge
import com.mobilerag.hebbian.cortical.CorticalConsolidator
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.hebbian.ui.HebbianEngineHolder
import com.mobilerag.profile.SpaceManager
import com.mobilerag.profile.SpaceManager.Space
import com.mobilerag.rag.RagDatabase
import com.mobilerag.rag.RagPipeline
import com.mobilerag.ui.Routes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import com.mobilerag.japanese.JaTranslator
import com.mobilerag.ui.theme.AllAccents
import com.mobilerag.ui.theme.accentById
import com.mobilerag.ui.theme.AppTheme
import com.mobilerag.ui.theme.GhostButton
import com.mobilerag.ui.theme.GlassCard
import com.mobilerag.ui.theme.GlassPill
import com.mobilerag.ui.theme.GlassSection
import com.mobilerag.ui.theme.GlassTextField
import com.mobilerag.ui.theme.NeonButton
import com.mobilerag.ui.theme.NeonIndeterminate
import com.mobilerag.ui.theme.ScreenHeader
import com.mobilerag.ui.theme.SectionLabel
import com.mobilerag.ui.theme.StatRow
import com.mobilerag.ui.theme.Telemetry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

/** Settings hub — the fifth bottom-nav destination. Rows navigate to subscreens and to the
 *  Diagnostics/Help top-level routes. */
@Composable
fun SettingsScreen(onNavigate: (String) -> Unit, onBack: (() -> Unit)? = null) {
    val rows = listOf(
        Triple("Appearance", "Theme mode and accent color", Routes.SettingsAppearance),
        Triple("Performance", "Presets, model, retrieval mode, backend", Routes.SettingsPerformance),
        Triple("Learner profile", "Name, Japanese level, goals — the tutor adapts", Routes.SettingsLearner),
        Triple("Memory spaces", "Isolated memory profiles with their own data", Routes.SettingsSpaces),
        Triple("Model & memory", "Resident model, installed models", Routes.SettingsModels),
        Triple("Data", "Clear memories, documents, practice state", Routes.SettingsData),
        Triple("Diagnostics", "On-device verification spikes", Routes.Spikes),
        Triple("Help & documentation", "How everything works", Routes.Help),
        Triple("About", "Version, stores, models", Routes.SettingsAbout),
    )
    Column(Modifier.fillMaxSize()) {
        ScreenHeader(title = "Settings")
        LazyColumn(
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(rows.size) { i ->
                val (title, subtitle, route) = rows[i]
                GlassCard(
                    Modifier.fillMaxWidth(),
                    onClick = { onNavigate(route) },
                    elevation = 6.dp,
                    glowAlpha = 0.15f,
                    fill = AppTheme.glass.panelLow,
                ) {
                    Row(
                        Modifier.fillMaxWidth().padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(title, style = MaterialTheme.typography.titleMedium)
                            Text(
                                subtitle,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowForward,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(18.dp),
                        )
                    }
                }
            }
        }
    }
}

/** Renders one settings subscreen for [route] (called from the NavHost). */
@Composable
fun SettingsSubScreen(route: String, navController: NavHostController) {
    val onBack: () -> Unit = { navController.popBackStack() }
    when (route) {
        Routes.SettingsAppearance -> AppearanceScreen(onBack)
        Routes.SettingsPerformance -> PerformanceScreen(onBack)
        Routes.SettingsLearner -> LearnerProfileScreen(onBack)
        Routes.SettingsSpaces -> SpacesScreen(onBack)
        Routes.SettingsModels -> ModelsScreen(onBack)
        Routes.SettingsData -> DataScreen(onBack)
        Routes.SettingsAbout -> AboutScreen(onBack)
        else -> SubScaffold("Settings", onBack) { Text("Unknown settings page") }
    }
}

@Composable
private fun SubScaffold(title: String, onBack: () -> Unit, content: @Composable () -> Unit) {
    Column(Modifier.fillMaxSize().imePadding()) {
        ScreenHeader(title = title, onBack = onBack)
        Column(
            Modifier
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp)
                .padding(bottom = 32.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) { content() }
    }
}

// ── Appearance ───────────────────────────────────────────────────────────────

@Composable
private fun AppearanceScreen(onBack: () -> Unit) {
    val settings = rememberAppSettings()
    SubScaffold("Appearance", onBack) {
        GlassSection(title = "Theme mode") {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                listOf(
                    AppSettings.THEME_SYSTEM to "System",
                    AppSettings.THEME_LIGHT to "Light",
                    AppSettings.THEME_DARK to "Dark",
                ).forEach { (value, label) ->
                    SegmentOption(
                        label = label,
                        selected = settings.themeMode == value,
                        modifier = Modifier.weight(1f),
                        onClick = { settings.themeMode = value },
                    )
                }
            }
        }

        GlassSection(title = "Accent") {
            // Each accent previews as its two roles: the marker on the dark sea (left) and the
            // selection slab in the white room (right).
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp), modifier = Modifier.fillMaxWidth()) {
                AllAccents.forEach { accent ->
                    AccentSwatch(
                        label = accent.label,
                        brush = Brush.horizontalGradient(
                            0f to accent.mark, 0.5f to accent.mark, 0.5f to accent.slab, 1f to accent.slab,
                        ),
                        selected = accentById(settings.themeAccent).id == accent.id,
                        onClick = { settings.themeAccent = accent.id },
                    )
                }
            }
            Text(
                "Used sparingly — markers and ticks in dark mode, the selection slab in light mode.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/** One option in a glass segmented control. */
@Composable
private fun SegmentOption(
    label: String,
    selected: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val glass = AppTheme.glass
    val shape = MaterialTheme.shapes.small
    Box(
        modifier
            .clip(shape)
            .then(
                if (selected) Modifier.background(glass.accentWash(0.30f))
                else Modifier.background(glass.panelLow),
            )
            .border(1.dp, if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.55f) else glass.rimDim, shape)
            .clickable(onClick = onClick)
            .padding(vertical = 12.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelLarge,
            color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** An accent choice, previewed as a gradient orb with a ring when active. */
@Composable
private fun AccentSwatch(
    label: String,
    brush: Brush,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp),
        modifier = Modifier.clickable(onClick = onClick),
    ) {
        Box(
            Modifier
                .size(46.dp)
                .clip(CircleShape)
                .then(
                    if (selected) Modifier.border(2.dp, MaterialTheme.colorScheme.onSurface, CircleShape)
                    else Modifier,
                )
                .padding(if (selected) 4.dp else 0.dp)
                .clip(CircleShape)
                .background(brush),
        )
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = if (selected) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

// ── Performance ──────────────────────────────────────────────────────────────

@Composable
private fun PerformanceScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val settings = rememberAppSettings()
    val models = remember { installedModels(context) }
    SubScaffold("Performance", onBack) {
        Text("Preset", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        val presets = listOf(
            Triple(AppSettings.PRESET_QUALITY, "Quality", "8B model · hybrid retrieval"),
            Triple(AppSettings.PRESET_BALANCED, "Balanced", "1.7B model · hybrid retrieval"),
            Triple(AppSettings.PRESET_SPEED, "Speed", "1.7B model · vector retrieval"),
        )
        presets.forEach { (value, label, subtitle) ->
            val selected = settings.performancePreset == value
            GlassCard(
                Modifier.fillMaxWidth(),
                onClick = {
                    settings.applyPerformancePreset(
                        value,
                        context.getSharedPreferences("rag", Context.MODE_PRIVATE),
                        models.map { it.name },
                    )
                },
                elevation = if (selected) 16.dp else 6.dp,
                glowAlpha = if (selected) 0.5f else 0.12f,
                fill = if (selected) null else AppTheme.glass.panelLow,
            ) {
                Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            label,
                            style = MaterialTheme.typography.titleMedium,
                            color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
                        )
                        Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    if (selected) Icon(Icons.Default.Check, contentDescription = "Active", tint = MaterialTheme.colorScheme.primary)
                }
            }
        }
        if (settings.performancePreset == AppSettings.PRESET_CUSTOM) {
            GlassPill("custom · overridden below", active = true)
        }

        GlassSection(title = "Overrides") {
        SettingDropdown(
            label = "Model",
            value = settings.selectedModel ?: "(default)",
            options = models.map { it.name },
            onSelect = { settings.selectedModel = it },
        )
        SettingDropdown(
            label = "Retrieval mode",
            value = settings.retrievalMode,
            options = listOf(RagPipeline.MODE_VECTOR, RagPipeline.MODE_HYBRID, RagPipeline.MODE_GLOBAL),
            onSelect = { settings.retrievalMode = it },
        )
        SettingDropdown(
            label = "Generation backend",
            value = settings.generationBackend,
            options = listOf(RagPipeline.BACKEND_AUTO, RagPipeline.BACKEND_LLAMACPP, RagPipeline.BACKEND_MLKIT),
            onSelect = { settings.generationBackend = it },
        )
        Text(
            "Overrides write the same preferences the Chat tab uses and mark the preset as custom.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        }
    }
}

@Composable
private fun SettingDropdown(label: String, value: String, options: List<String>, onSelect: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Row(
        Modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.small)
            .clickable { open = true }
            .padding(vertical = 10.dp, horizontal = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            label,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Box {
            GlassPill(value, active = true)
            DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                options.forEach { opt ->
                    DropdownMenuItem(text = { Text(opt) }, onClick = { onSelect(opt); open = false })
                }
            }
        }
    }
}

// ── Learner profile ──────────────────────────────────────────────────────────

@Composable
private fun LearnerProfileScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val settings = rememberAppSettings()
    var name by remember { mutableStateOf(settings.learnerName) }
    var level by remember { mutableStateOf(settings.learnerLevel) }
    var goals by remember { mutableStateOf(settings.learnerGoals) }
    var saved by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val translateState by JaTranslator.modelState.collectAsState()

    SubScaffold("Learner profile", onBack) {
        Text(
            "The tutor adapts its explanations, vocabulary, and example difficulty to this profile.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        GlassSection(title = "You") {
            GlassTextField(
                value = name,
                onValueChange = { name = it; saved = false },
                placeholder = "Your name",
                label = "Name",
            )
            Spacer(Modifier.height(4.dp))
            SectionLabel("Japanese level")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                listOf(
                    AppSettings.LEVEL_N5, AppSettings.LEVEL_N4, AppSettings.LEVEL_N3,
                    AppSettings.LEVEL_N2, AppSettings.LEVEL_N1,
                ).forEach { l ->
                    SegmentOption(
                        label = l,
                        selected = level == l,
                        modifier = Modifier.weight(1f),
                        onClick = { level = l; saved = false },
                    )
                }
            }
            Spacer(Modifier.height(4.dp))
            GlassTextField(
                value = goals,
                onValueChange = { goals = it; saved = false },
                placeholder = "e.g. pass JLPT N4, travel conversations",
                label = "Goals",
                singleLine = false,
                minLines = 2,
            )
            Spacer(Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                NeonButton(
                    text = "Save",
                    onClick = {
                        settings.learnerName = name
                        settings.learnerLevel = level
                        settings.learnerGoals = goals
                        saved = true
                        // No unload needed: the learner block is part of the persona prompt's
                        // residency key, so the next tutor turn swaps the new prompt in over the
                        // resident weights.
                    },
                )
                if (saved) {
                    Text(
                        "Saved — applies from the next turn.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }

        // Surfaced here so tap-to-translate can be primed before the learner is mid-conversation
        // and hits a download prompt inside the overlay.
        GlassSection(
            title = "Tap-to-translate",
            trailing = {
                GlassPill(
                    text = when (translateState) {
                        is JaTranslator.ModelState.Ready -> "ready"
                        is JaTranslator.ModelState.Downloading -> "downloading"
                        is JaTranslator.ModelState.Failed -> "failed"
                        else -> "not installed"
                    },
                    active = translateState is JaTranslator.ModelState.Ready,
                )
            },
        ) {
            Text(
                "Tap any Japanese word in the tutor to see what it means. The JA→EN pack downloads once (~30 MB), then works offline.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (translateState is JaTranslator.ModelState.Downloading) {
                NeonIndeterminate()
            } else {
                GhostButton(
                    text = if (translateState is JaTranslator.ModelState.Ready) "Re-check pack" else "Download pack",
                    onClick = { scope.launch { JaTranslator.ensureModel(context) } },
                )
            }
            (translateState as? JaTranslator.ModelState.Failed)?.let {
                Text(
                    it.message,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

// ── Memory spaces ────────────────────────────────────────────────────────────

@Composable
private fun SpacesScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var spaces by remember { mutableStateOf(SpaceManager.listSpaces(context)) }
    var active by remember { mutableStateOf(SpaceManager.activeSpace(context).id) }
    var dialog by remember { mutableStateOf<SpaceDialog?>(null) }

    SubScaffold("Memory spaces", onBack) {
        Text(
            "Each space has its own documents, entity graph, Hebbian memory, and practice state. Models and appearance are shared.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(12.dp))
        spaces.forEach { space ->
            val isActive = space.id == active
            GlassCard(
                Modifier.fillMaxWidth(),
                elevation = if (isActive) 16.dp else 6.dp,
                glowAlpha = if (isActive) 0.5f else 0.12f,
                fill = if (isActive) null else AppTheme.glass.panelLow,
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            space.name,
                            style = MaterialTheme.typography.titleMedium,
                            color = if (isActive) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.weight(1f),
                        )
                        if (isActive) GlassPill("active", active = true)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (!isActive) {
                            GhostButton("Switch", { dialog = SpaceDialog.Switch(space) })
                        }
                        GhostButton("Rename", { dialog = SpaceDialog.Rename(space) })
                        if (spaces.size > 1) {
                            GhostButton("Delete", { dialog = SpaceDialog.Delete(space) })
                        }
                    }
                }
            }
        }
        NeonButton("Add space", { dialog = SpaceDialog.Add })
    }

    when (val d = dialog) {
        is SpaceDialog.Add -> NameDialog(
            title = "New space",
            initial = "",
            confirmLabel = "Create",
            onDismiss = { dialog = null },
            onConfirm = { name ->
                SpaceManager.addSpace(context, name)
                spaces = SpaceManager.listSpaces(context)
                dialog = null
            },
        )
        is SpaceDialog.Rename -> NameDialog(
            title = "Rename ${d.space.name}",
            initial = d.space.name,
            confirmLabel = "Rename",
            onDismiss = { dialog = null },
            onConfirm = { name ->
                SpaceManager.renameSpace(context, d.space.id, name)
                spaces = SpaceManager.listSpaces(context)
                dialog = null
            },
        )
        is SpaceDialog.Delete -> ConfirmDialog(
            title = "Delete ${d.space.name}?",
            body = "All documents, graphs, Hebbian memory, and practice state in this space are permanently deleted.",
            confirmLabel = "Delete",
            onDismiss = { dialog = null },
            onConfirm = {
                scope.launch {
                    val wasActive = d.space.id == active
                    // deleteSpace moves the active marker off a deleted active space first
                    SpaceManager.deleteSpace(context, d.space.id)
                    if (wasActive) {
                        // full reload against the new active space (recreates the activity)
                        switchSpace(context, SpaceManager.activeSpace(context).id)
                    } else {
                        spaces = SpaceManager.listSpaces(context)
                        dialog = null
                    }
                }
            },
        )
        is SpaceDialog.Switch -> ConfirmDialog(
            title = "Switch to ${d.space.name}?",
            body = "The app unloads the model and reloads all data from this space.",
            confirmLabel = "Switch",
            onDismiss = { dialog = null },
            onConfirm = {
                scope.launch { switchSpace(context, d.space.id) }
            },
        )
        null -> {}
    }
}

private sealed interface SpaceDialog {
    data object Add : SpaceDialog
    data class Rename(val space: Space) : SpaceDialog
    data class Delete(val space: Space) : SpaceDialog
    data class Switch(val space: Space) : SpaceDialog
}

/** Full space switch: drop engines/stores/resident model, set active, recreate the activity
 *  so every singleton rebuilds against the new space (documented choice — see plan). */
private suspend fun switchSpace(context: Context, spaceId: String) {
    withContext(Dispatchers.IO) {
        runCatching { HebbianEngineHolder.reset() }
        unloadResidentModel(context)
        runCatching { GraphStoreFactory.closeCurrent() }
        runCatching { HebbianStoreFactory.closeCurrent() }
        SpaceManager.setActive(context, spaceId)
    }
    (context as? Activity)?.recreate()
}

// ── Model & memory ───────────────────────────────────────────────────────────

@Composable
private fun ModelsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val settings = rememberAppSettings()
    var resident by remember { mutableStateOf(LlmResidency.loadedModel) }
    val models = remember { installedModels(context) }
    SubScaffold("Model & memory", onBack) {
        GlassSection(title = "Resident model") {
            Text(
                resident?.substringAfterLast('/') ?: "none loaded",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
            GhostButton(
                text = "Unload model",
                onClick = {
                    unloadResidentModel(context)
                    resident = LlmResidency.loadedModel // still set if a generation was running
                },
                enabled = resident != null,
            )
        }

        GlassSection(title = "Installed models · shared across spaces") {
            models.forEach { f ->
                StatRow(f.name, formatBytes(f.length()), mono = true)
            }
            if (models.isEmpty()) {
                Text(
                    "No .gguf models installed — push one via adb (see Help → Models).",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            if (settings.selectedModel == null && models.isNotEmpty()) {
                Text(
                    "No explicit selection — the app prefers 8B, then 1.7B, then the first model.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

/** Unloads whatever the shared llama.cpp engine currently holds. */
// see SettingsCommon.kt: unloadResidentModel(context)

// ── Data ─────────────────────────────────────────────────────────────────────

@Composable
private fun DataScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val space = remember { SpaceManager.activeSpace(context) }
    var action by remember { mutableStateOf<DataAction?>(null) }
    var mergePreview by remember { mutableStateOf<DuplicateMerge.Preview?>(null) }
    var busy by remember { mutableStateOf(false) }
    var done by remember { mutableStateOf<String?>(null) }

    SubScaffold("Data", onBack) {
        Text(
            "These actions apply to the active space: ${space.name}. They cannot be undone.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error,
        )
        DestructiveRow("Clear Hebbian memory", "Concepts, edges, ARM-E stats, sessions") { action = DataAction.ClearHebbian }
        DestructiveRow("Clear documents & entity graph", "Imported notes, chunks, entities, communities") { action = DataAction.ClearDocuments }
        DestructiveRow("Reset practice state", "Learned half-lives and practice history") { action = DataAction.ResetPractice }
        DestructiveRow(
            "Merge duplicate concepts",
            "Concepts with the same label and type become one. A backup is saved first.",
        ) {
            busy = true
            done = null
            scope.launch {
                runCatching { withContext(Dispatchers.IO) { DuplicateMerge.preview(context) } }
                    .onSuccess { p -> if (p.duplicates == 0) done = "No duplicate concepts found." else mergePreview = p }
                    .onFailure { done = "Failed: ${it.message?.take(120)}" }
                busy = false
            }
        }
        if (busy) NeonIndeterminate()
        done?.let {
            Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
        }
    }

    mergePreview?.let { p ->
        ConfirmDialog(
            title = "Merge duplicate concepts",
            body = "${p.duplicates} duplicate concepts will be merged into ${p.groups} in space " +
                "\"${space.name}\": their edges, activation counts and history combine. A backup " +
                "of the Hebbian store is saved to this space's backups folder first. Continue?",
            confirmLabel = "Merge",
            onDismiss = { mergePreview = null },
            onConfirm = {
                mergePreview = null
                busy = true
                scope.launch {
                    done = runCatching { withContext(Dispatchers.IO) { DuplicateMerge.run(context) } }
                        .fold(onSuccess = { it }, onFailure = { "Failed: ${it.message?.take(120)}" })
                    busy = false
                }
            },
        )
    }

    action?.let { a ->
        ConfirmDialog(
            title = a.title,
            body = "This permanently deletes the data in space \"${space.name}\". Continue?",
            confirmLabel = "Delete",
            onDismiss = { action = null },
            onConfirm = {
                action = null
                busy = true
                done = null
                scope.launch {
                    done = runCatching { withContext(Dispatchers.IO) { a.run(context) } }
                        .fold(onSuccess = { it }, onFailure = { "Failed: ${it.message?.take(120)}" })
                    busy = false
                }
            },
        )
    }
}

@Composable
private fun DestructiveRow(label: String, subtitle: String, onClick: () -> Unit) {
    val shape = MaterialTheme.shapes.medium
    Column(
        Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(MaterialTheme.colorScheme.error.copy(alpha = 0.08f))
            .border(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.30f), shape)
            .clickable(onClick = onClick)
            .padding(16.dp),
    ) {
        Text(label, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.error)
        Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

private sealed class DataAction(val title: String, val run: suspend (Context) -> String) {
    data object ClearHebbian : DataAction("Clear Hebbian memory", { context ->
        val spaceDir = SpaceManager.activeDir(context)
        val store = HebbianStoreFactory.create(context, spaceDir)
        store.clear()
        // Consolidated weights and z_prev describe nodes that no longer exist.
        CorticalConsolidator.forSpace(spaceDir).clear()
        "Hebbian memory cleared."
    })

    data object ClearDocuments : DataAction("Clear documents & entity graph", { context ->
        val spaceDir = SpaceManager.activeDir(context)
        val db = RagDatabase(context, File(spaceDir, "rag.db").absolutePath)
        try {
            val graph = GraphStoreFactory.create(context, spaceDir)
            val docs = db.listDocuments()
            for (doc in docs) {
                val chunkIds = db.chunkIdsForDocument(doc.id)
                db.deleteDocument(doc.id)
                if (chunkIds.isNotEmpty()) graph.removeGraphForChunks(chunkIds)
            }
            "Deleted ${docs.size} documents and their graph data."
        } finally {
            db.close()
        }
    })

    data object ResetPractice : DataAction("Reset practice state", { context ->
        val dir = File(SpaceManager.activeDir(context), "practice_state")
        val n = dir.listFiles()?.count { it.delete() } ?: 0
        "Practice state reset ($n files)."
    })
}

// ── About ────────────────────────────────────────────────────────────────────

@Composable
private fun AboutScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    var stores by remember { mutableStateOf<Pair<String, String>?>(null) }
    androidx.compose.runtime.LaunchedEffect(Unit) {
        stores = withContext(Dispatchers.IO) {
            runCatching {
                val dir = SpaceManager.activeDir(context)
                GraphStoreFactory.create(context, dir).id to HebbianStoreFactory.create(context, dir).id
            }.getOrNull()
        }
    }
    val models = remember { installedModels(context) }
    SubScaffold("About", onBack) {
        GlassSection {
            Text("mobileRAG", style = MaterialTheme.typography.headlineSmall)
            Telemetry("v0.1.0 · on-device GraphRAG + Hebbian memory")
        }
        GlassSection(title = "Stack") {
            StatRow("Entity graph store", stores?.first ?: "—", mono = true)
            StatRow("Hebbian store", stores?.second ?: "—", mono = true)
            StatRow("Embeddings", "EmbeddingGemma-300M")
            StatRow("Entity extraction", "GLiNER2 multi (int8)")
            StatRow("Generation", "llama.cpp · ${models.size} GGUF")
            StatRow("Translation", "ML Kit JA→EN")
        }
        Text(
            "Everything runs on-device: documents, memories, embeddings, and models never leave the phone. The models (on first run) and the translation pack are the only things ever downloaded; nothing you write is ever sent anywhere.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

// ── Shared bits ──────────────────────────────────────────────────────────────
// ConfirmDialog / NameDialog / installedModels / formatBytes / unloadResidentModel
// live in SettingsCommon.kt (same package).

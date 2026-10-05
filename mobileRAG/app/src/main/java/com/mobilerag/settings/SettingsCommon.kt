package com.mobilerag.settings

import android.content.Context
import android.util.Log
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import com.mobilerag.generation.LlmResidency
import java.io.File

/** Small top bar with a back arrow shared by all settings subscreens. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SettingsTopBar(title: String, onBack: () -> Unit) {
    TopAppBar(
        title = { Text(title) },
        navigationIcon = {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
            }
        },
    )
}

/** Destructive-action confirm dialog; the confirm button is rendered in the error color. */
@Composable
internal fun ConfirmDialog(
    title: String,
    body: String,
    confirmLabel: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(body) },
        confirmButton = {
            TextButton(onClick = { onDismiss(); onConfirm() }) {
                Text(confirmLabel, color = MaterialTheme.colorScheme.error)
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

/** Single-line name entry dialog (add/rename space). */
@Composable
internal fun NameDialog(
    title: String,
    initial: String,
    confirmLabel: String,
    onConfirm: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var value by rememberSaveable { mutableStateOf(initial) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            OutlinedTextField(value = value, onValueChange = { value = it }, singleLine = true)
        },
        confirmButton = {
            TextButton(
                onClick = { onDismiss(); onConfirm(value.trim()) },
                enabled = value.isNotBlank(),
            ) { Text(confirmLabel) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

/** Installed chat GGUFs under files/models — the same source RagPipeline.availableModels uses. */
internal fun installedModels(context: Context): List<File> =
    File(context.filesDir, "models").listFiles()
        ?.filter { it.extension == "gguf" && it.canRead() }
        ?.sortedBy { it.name } ?: emptyList()

internal fun formatBytes(bytes: Long): String = when {
    bytes >= 1L shl 30 -> "%.2f GB".format(bytes / (1024.0 * 1024.0 * 1024.0))
    bytes >= 1L shl 20 -> "%.1f MB".format(bytes / (1024.0 * 1024.0))
    else -> "%d KB".format(bytes / 1024)
}

/**
 * Drops whatever GGUF is resident in the shared AiChat InferenceEngine singleton (RAG chat or
 * a Hebbian persona) so its pages can be reclaimed; the next consumer reloads it lazily.
 * Skipped while a generation is running.
 */
internal fun unloadResidentModel(context: Context) {
    if (LlmResidency.unloadIfIdle(context)) Log.i("Settings", "resident model unloaded")
}

package com.mobilerag.rag.ingest

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.security.MessageDigest

/** Stream-copies picked documents to filesDir/corpus under a sanitized name, computing SHA-256 on the fly. */
class DocumentImporter(private val context: Context) {

    data class Imported(val file: File, val name: String, val sha256: String)

    private val corpusDir = File(context.filesDir, "corpus")

    suspend fun import(uris: List<Uri>): List<Imported> = withContext(Dispatchers.IO) {
        corpusDir.mkdirs()
        uris.mapNotNull { uri ->
            // MIME detection for .md is unreliable, so the picker is launched with */* and we filter by extension
            val name = displayName(uri)?.let(::sanitize) ?: return@mapNotNull null
            if (name.substringAfterLast('.').lowercase() !in SUPPORTED_EXTENSIONS) return@mapNotNull null
            val digest = MessageDigest.getInstance("SHA-256")
            val out = File(corpusDir, name)
            context.contentResolver.openInputStream(uri)?.use { input ->
                out.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        digest.update(buffer, 0, read)
                        output.write(buffer, 0, read)
                    }
                }
            } ?: return@mapNotNull null
            Imported(out, name, digest.digest().joinToString("") { "%02x".format(it) })
        }
    }

    private fun displayName(uri: Uri): String? =
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: uri.lastPathSegment

    private fun sanitize(name: String): String =
        name.replace(Regex("[^A-Za-z0-9._-]"), "_").ifBlank { "document" }

    companion object {
        private val SUPPORTED_EXTENSIONS = setOf("txt", "md", "markdown")
    }
}

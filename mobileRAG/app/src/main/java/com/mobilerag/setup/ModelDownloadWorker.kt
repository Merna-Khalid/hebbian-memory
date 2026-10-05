package com.mobilerag.setup

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.mobilerag.R
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.TimeUnit

/**
 * Downloads the missing models (ModelCatalog) as a foreground job with a progress
 * notification, so it survives the app being backgrounded. Each file streams into
 * `<path>.part` and resumes with an HTTP Range request after a drop or a retry; it is renamed
 * into place only when its size matches the catalogue exactly.
 */
class ModelDownloadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val ctx = applicationContext
        val todo = ModelCatalog.missing(ctx)
        if (todo.isEmpty()) return@withContext Result.success()
        // Progress is measured against the whole catalogue, so it only ever climbs across
        // retries (measuring just the remaining files made 80% drop back to 22%).
        val total = ModelCatalog.totalBytes
        var doneBefore = ModelCatalog.items.filter { it !in todo }.sumOf { it.bytes }
        runCatching { setForeground(foregroundInfo(0, total, todo.first().group.title)) }
            .onFailure { Log.w(TAG, "foreground unavailable; downloading anyway", it) }
        try {
            for (item in todo) {
                downloadOne(item) { itemDone ->
                    val done = doneBefore + itemDone
                    setProgress(workDataOf(KEY_DONE to done, KEY_TOTAL to total, KEY_FILE to item.group.title))
                    runCatching { setForeground(foregroundInfo(done, total, item.group.title)) }
                }
                doneBefore += item.bytes
            }
            Result.success()
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            Log.w(TAG, "download interrupted (${t.message}); will resume", t)
            // Partial files stay; the retry continues from where each one stopped.
            if (runAttemptCount < MAX_ATTEMPTS) Result.retry()
            else Result.failure(workDataOf(KEY_ERROR to (t.message ?: t.javaClass.simpleName)))
        }
    }

    private suspend fun downloadOne(item: ModelCatalog.Item, onProgress: suspend (Long) -> Unit) {
        val target = ModelCatalog.file(applicationContext, item)
        target.parentFile?.mkdirs()
        val part = File(target.path + ".part")
        var have = if (part.isFile) part.length() else 0L
        if (have > item.bytes) { part.delete(); have = 0L }

        if (have < item.bytes) {
            val conn = (URL(item.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                if (have > 0) setRequestProperty("Range", "bytes=$have-")
            }
            try {
                val code = conn.responseCode
                val append = when {
                    code == HttpURLConnection.HTTP_PARTIAL -> true
                    code == HttpURLConnection.HTTP_OK -> { have = 0L; false } // server ignored the range
                    else -> throw IOException("HTTP $code for ${item.path}")
                }
                conn.inputStream.use { input ->
                    FileOutputStream(part, append).use { out ->
                        val buf = ByteArray(256 * 1024)
                        var lastReport = 0L
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            have += n
                            val now = System.currentTimeMillis()
                            if (now - lastReport > 500) {
                                lastReport = now
                                onProgress(have)
                            }
                        }
                    }
                }
            } finally {
                conn.disconnect()
            }
        }
        if (part.length() != item.bytes) {
            throw IOException("${item.path}: got ${part.length()} of ${item.bytes} bytes")
        }
        if (target.exists()) target.delete()
        if (!part.renameTo(target)) throw IOException("could not move ${part.name} into place")
        onProgress(item.bytes)
    }

    private fun foregroundInfo(done: Long, total: Long, file: String): ForegroundInfo {
        val nm = applicationContext.getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Model download", NotificationManager.IMPORTANCE_LOW))
        }
        val pct = if (total > 0) (done * 100 / total).toInt() else 0
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentTitle("Awakening — downloading models")
            .setContentText("$file · $pct%% of %.1f GB".format(total / 1e9))
            .setProgress(100, pct, total == 0L)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
        return ForegroundInfo(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
    }

    companion object {
        private const val TAG = "ModelDownload"
        const val NAME = "model_download"
        const val KEY_DONE = "done"
        const val KEY_TOTAL = "total"
        const val KEY_FILE = "file"
        const val KEY_ERROR = "error"
        private const val CHANNEL = "model_download"
        private const val NOTIFICATION_ID = 4101
        /** Linear 30 s steps: a long Wi-Fi outage then waits minutes, not the hours exponential
         *  backoff reaches after a few failures — and gives up only after ~an hour of trying. */
        private const val MAX_ATTEMPTS = 20

        /** Starts (or, when the user asks, restarts right now) the download. Partial files are kept. */
        fun start(context: Context) {
            val request = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .setBackoffCriteria(BackoffPolicy.LINEAR, 30, TimeUnit.SECONDS)
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(NAME, ExistingWorkPolicy.REPLACE, request)
        }

        fun pause(context: Context) {
            WorkManager.getInstance(context).cancelUniqueWork(NAME)
        }
    }
}

package com.mobilerag.rag

import android.content.Context
import android.util.Log
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.mobilerag.embeddings.EmbeddingGemmaEngine
import com.mobilerag.graph.GraphStoreFactory
import com.mobilerag.profile.SpaceManager
import com.mobilerag.rag.retrieve.CommunityRetriever
import com.mobilerag.rag.retrieve.RetrievalEngine
import kotlinx.coroutines.CancellationException
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Phase 4 background community indexing: regenerates community summaries daily while the
 * device is idle and charging, and once (replaced) after each document import. Runs in the
 * app process — GraphStoreFactory and the llama.cpp InferenceEngine are process-wide
 * singletons, so the shared [RagPipeline] path (model load → rebuild → unload in a finally)
 * is reused directly.
 */
class CommunityIndexWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val engine = try {
            EmbeddingGemmaEngine.create(applicationContext)
        } catch (t: Throwable) {
            Log.w(TAG, "embedding engine unavailable — not retrying", t)
            return Result.failure()
        }
        val spaceDir = SpaceManager.activeDir(applicationContext)
        val db = RagDatabase(applicationContext, File(spaceDir, "rag.db").absolutePath)
        try {
            val graph = GraphStoreFactory.create(applicationContext, spaceDir)
            val retrieval = RetrievalEngine(engine, VectorStore(db, engine.dimensions), db, graph)
            val pipeline = RagPipeline(applicationContext, engine, retrieval, CommunityRetriever(db))
            val count = pipeline.rebuildCommunities(graph, db)
            Log.i(TAG, "rebuilt $count community summaries")
            return Result.success()
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            Log.w(TAG, "community rebuild failed (attempt ${runAttemptCount + 1})", t)
            return if (runAttemptCount + 1 < MAX_ATTEMPTS) Result.retry() else Result.failure()
        } finally {
            runCatching { db.close() }
            engine.close()
        }
    }

    companion object {
        private const val TAG = "CommunityIndexWorker"
        private const val PERIODIC_NAME = "community_index"
        private const val ONE_SHOT_NAME = "community_index_now"
        private const val MAX_ATTEMPTS = 3

        private fun constraints() = Constraints.Builder()
            .setRequiresCharging(true)
            .setRequiresDeviceIdle(true)
            .build()

        /** Daily rebuild; KEEP so app restarts don't reset the schedule. */
        fun enqueuePeriodic(context: Context) {
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                PERIODIC_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                PeriodicWorkRequestBuilder<CommunityIndexWorker>(1, TimeUnit.DAYS)
                    .setConstraints(constraints())
                    .build(),
            )
        }

        /** One-shot rebuild after a document import changed the graph; REPLACE cancels any
         *  pending post-import run so only the latest corpus state is summarized. */
        fun enqueueNow(context: Context) {
            WorkManager.getInstance(context).enqueueUniqueWork(
                ONE_SHOT_NAME,
                ExistingWorkPolicy.REPLACE,
                OneTimeWorkRequestBuilder<CommunityIndexWorker>()
                    .setConstraints(constraints())
                    .build(),
            )
        }
    }
}

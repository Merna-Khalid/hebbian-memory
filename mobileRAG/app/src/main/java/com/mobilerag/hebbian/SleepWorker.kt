package com.mobilerag.hebbian

import android.content.Context
import android.util.Log
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.mobilerag.hebbian.cortical.CorticalConsolidator
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.profile.SpaceManager
import kotlinx.coroutines.CancellationException
import java.util.concurrent.TimeUnit

/**
 * Nightly "sleep": a cortical consolidation of the active memory space while the device is
 * idle and charging. Groundwork for phase 7 (docs/phase7-inner-state-proposal.md), which
 * grows this into the full cycle — replay, consolidation, insight detection, homeostasis,
 * dream. Same WorkManager pattern as rag/CommunityIndexWorker.
 */
class SleepWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = try {
        val spaceDir = SpaceManager.activeDir(applicationContext)
        val store = HebbianStoreFactory.create(applicationContext, spaceDir)
        val report = CorticalConsolidator.forSpace(spaceDir).consolidate(store, "sleep")
        Log.i(TAG, report?.let { "slept: ${it.nodes} nodes, ${it.edges} edges in ${it.elapsedMs} ms" }
            ?: "slept: nothing to consolidate (or a run was already in progress)")
        Result.success()
    } catch (t: Throwable) {
        if (t is CancellationException) throw t
        Log.w(TAG, "sleep consolidation failed", t)
        Result.failure()
    }

    companion object {
        private const val TAG = "SleepWorker"
        private const val PERIODIC_NAME = "hebbian_sleep"

        /** Daily; KEEP so app restarts don't reset the schedule. */
        fun enqueuePeriodic(context: Context) {
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                PERIODIC_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                PeriodicWorkRequestBuilder<SleepWorker>(1, TimeUnit.DAYS)
                    .setConstraints(
                        Constraints.Builder()
                            .setRequiresCharging(true)
                            .setRequiresDeviceIdle(true)
                            .build(),
                    )
                    .build(),
            )
        }
    }
}

package com.mobilerag.hebbian

import android.content.Context
import com.mobilerag.hebbian.cortical.CorticalConsolidator
import com.mobilerag.hebbian.store.ConceptMerge
import com.mobilerag.hebbian.store.HebbianStoreFactory
import com.mobilerag.hebbian.ui.HebbianEngineHolder
import com.mobilerag.profile.SpaceManager
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Settings → Data → "Merge duplicate concepts": the opt-in cleanup for duplicates created
 * before concept identity existed (new ones are prevented at ingest). Always backs up the
 * active space's Hebbian store first; the merge itself follows [ConceptMerge].
 */
object DuplicateMerge {

    data class Preview(val groups: Int, val duplicates: Int)

    private const val ALL = 1_000_000

    /** Dry run: what a merge would do, without changing anything. */
    suspend fun preview(context: Context): Preview {
        val store = HebbianStoreFactory.create(context, SpaceManager.activeDir(context))
        val plan = ConceptMerge.plan(store.listConcepts(ALL))
        return Preview(plan.groups.size, plan.droppedCount)
    }

    /** Backup, then merge. Returns a user-facing summary. */
    suspend fun run(context: Context): String {
        val spaceDir = SpaceManager.activeDir(context)
        if (CorticalConsolidator.forSpace(spaceDir).isRunning) {
            return "Memory consolidation is running — try again in a minute."
        }
        val store = HebbianStoreFactory.create(context, spaceDir)
        val plan = ConceptMerge.plan(store.listConcepts(ALL))
        if (plan.groups.isEmpty()) return "No duplicate concepts found."

        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val backup = File(spaceDir, "backups/hebbian-$stamp")
        store.backupTo(backup)
        store.applyMerge(plan)

        // Practice history: through the live tutor engine if there is one (so its next save
        // doesn't write the old ids back), otherwise straight to the persisted state.
        HebbianEngineHolder.peek()?.remapPracticeNodes(plan.remap)
            ?: PracticeScheduler(File(spaceDir, "practice_state")).remapNodes(plan.remap)

        return "Merged ${plan.droppedCount} duplicates into ${plan.groups.size} concept${if (plan.groups.size == 1) "" else "s"}. " +
            "Backup: backups/hebbian-$stamp"
    }
}

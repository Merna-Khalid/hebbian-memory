package com.mobilerag.spikes

import android.content.Context
import com.mobilerag.eval.EvalRunner
import com.mobilerag.eval.EvalSetupException

/**
 * Phase 4 spike: thin UI over [EvalRunner], the persistent regression eval harness. Runs the
 * labelled question set in files/eval/ in vector and hybrid modes (retrieval only, no LLM),
 * renders recall@5 / gate accuracy per mode and category, the verdict vs the previous run, and
 * the history file this run was saved to. Reports to logcat under "HybridEval".
 */
class HybridRetrievalSpike : Spike {
    override val id = "hybrid_eval"
    override val title = "Hybrid retrieval eval"
    override val description = "Vector vs hybrid recall@5 + gate accuracy over files/eval/eval-vN.json; persists runs and compares vs baseline."

    override suspend fun run(context: Context, log: (String) -> Unit): SpikeResult {
        return try {
            val result = EvalRunner(context).run(log)
            result.report.lineSequence().forEach(log)
            SpikeResult(
                passed = result.verdicts.values.none { it == EvalRunner.VERDICT_REGRESSION },
                summary = result.report,
                metrics = buildMap {
                    for ((mode, s) in result.stats) {
                        put("$mode recall@5", "%.2f (%s)".format(s.recall, result.verdicts[mode]))
                        put("$mode gate acc", "%.2f".format(s.gateAccuracy))
                    }
                    put("history", result.historyFileName)
                },
            )
        } catch (e: EvalSetupException) {
            SpikeResult(passed = false, summary = e.message ?: "Eval setup failed")
        } catch (t: Throwable) {
            SpikeResult(
                passed = false,
                summary = "Failed: ${t.javaClass.simpleName}: ${t.message?.take(200)}",
                error = t.stackTraceToString().take(2000),
            )
        }
    }
}

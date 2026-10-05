package com.mobilerag.spikes

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.SystemClock

/** Timing + memory sampling for spike runs. All heavy weights live in native memory, so we
 *  track both the Java heap and total PSS. */
class Instrumentation(private val context: Context, private val log: (String) -> Unit) {

    private var startMs: Long = 0
    private var startPssKb: Int = 0

    fun mark(label: String) {
        startMs = SystemClock.elapsedRealtime()
        startPssKb = totalPssKb()
        log("[$label] start — PSS ${startPssKb / 1024} MB, heap ${heapMb()} MB")
    }

    /** Returns elapsed ms. */
    fun elapsed(label: String): Long {
        val ms = SystemClock.elapsedRealtime() - startMs
        val pssKb = totalPssKb()
        val deltaKb = pssKb - startPssKb
        log("[$label] ${ms} ms — PSS ${pssKb / 1024} MB (Δ ${deltaKb / 1024} MB), heap ${heapMb()} MB")
        return ms
    }

    fun totalPssKb(): Int {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = am.getProcessMemoryInfo(intArrayOf(android.os.Process.myPid())).firstOrNull()
        return info?.totalPss ?: Debug.getPss().toInt()
    }

    fun heapMb(): Long = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)
}

/** Lightweight per-query timer for the RAG pipeline: named marks + elapsed, with a memory snapshot. */
class QueryTimer {
    private val t0 = SystemClock.elapsedRealtime()
    private val marks = mutableMapOf<String, Long>()

    fun mark(name: String) {
        marks[name] = SystemClock.elapsedRealtime()
    }

    /** ms since [name] was marked (or since construction if the mark is unknown). */
    fun since(name: String): Long = SystemClock.elapsedRealtime() - (marks[name] ?: t0)

    companion object {
        fun pssKb(): Long = Debug.getPss()
        fun heapMb(): Long = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)
    }
}

package com.mobilerag.hebbian

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * ARM-E: Autonomous Reinforcement Modulation Engine — v2
 * (port of the production math in core/arme_v2.py; demos and plots omitted).
 *
 * 1. Quadrant-aware valence x arousal interaction for e_t.
 * 2. Sparse eligibility gating: m_t is global, per-edge traces decide
 *    which edges receive the signal.
 * 3. Anti-Hebbian normalization on non-eligible edges.
 */
data class EmotionalState(
    val valence: Double,    // [-1, +1]  negative <-> positive
    val arousal: Double,    // [0,   1]  calm <-> excited
    val dominance: Double,  // [0,   1]  controlled <-> in-control
    val quadrant: String,   // Q1/Q2/Q3/Q4
    val eT: Double,         // [-1, +1] final signal
)

data class ARMEState(
    val rT: Double,
    val uT: Double,
    val deltaT: Double,
    val emotion: EmotionalState,
    val sT: Double,
    val mTRaw: Double,
    val mT: Double,
    val dominance: Double,
    val gated: Boolean,
    val timestamp: Double,
)

/**
 * Quadrant-aware emotional intensity (arousal threshold 0.5), clipped to [-1, 1].
 * Q1 (v>=0, a>=0.5): amplify. Q2 (v<0, a>=0.5): strong suppression.
 * Q3 (v<0, a<0.5): mild suppression. Q4 (v>=0, a<0.5): mild positive.
 */
fun computeETQuadrant(
    valence: Double,
    arousal: Double,
    alphaE: Double = 0.6,
    betaE: Double = 0.4,
): EmotionalState {
    val vAbs = abs(valence)
    val threshold = 0.5
    val quadrant: String
    var eT: Double
    if (valence >= 0 && arousal >= threshold) {
        quadrant = "Q1"
        eT = tanh(alphaE * valence + betaE * arousal)
    } else if (valence >= 0) {
        quadrant = "Q4"
        eT = tanh(alphaE * valence + betaE * arousal * 0.5)
    } else if (arousal >= threshold) {
        quadrant = "Q2"
        eT = -tanh(alphaE * vAbs * (1.0 + betaE * arousal))
    } else {
        quadrant = "Q3"
        eT = -tanh(alphaE * vAbs * (1.0 - betaE * arousal))
    }
    eT = max(-1.0, min(1.0, eT))
    return EmotionalState(valence = valence, arousal = arousal, dominance = 0.5, quadrant = quadrant, eT = eT)
}

data class HebbianStats(
    val nEligible: Int,
    val nTotal: Int,
    val meanCoact: Double,
    val meanW: Double,
    val eligibleMeanW: Double,
    val ineligibleMeanW: Double,
    val meanAbsDeltaW: Double,
    val maxAbsDeltaW: Double,
)

data class HebbianResult(
    val edgeWeight: DoubleArray,
    val eligibility: DoubleArray,
    val stats: HebbianStats,
)

class Arme(
    val wR: Double = 0.6,
    val wU: Double = -0.2,
    val wD: Double = 0.5,
    val wE: Double = 0.4,
    val wDom: Double = 0.0,     // dominance weight — zero until Phase 4 tuning
    val bias: Double = -0.3,
    val mMin: Double = 0.2,
    val mMax: Double = 3.0,
    val emaLambda: Double = 0.4,
    val tau: Double = 300.0,
    val alphaR: Double = 0.7,
    val gateThreshold: Double = 0.15,   // min signal confidence to activate ARM-E
    val lamAnti: Double = 0.003,
    val tauElig: Double = 5.0,
    val thetaElig: Double = 0.2,
    val kappa: Double = 1.0,
    private val timeNowS: () -> Double = { System.currentTimeMillis() / 1000.0 },
) {
    private var mPrev: Double = 1.0
    private val lastSeen = HashMap<String, Double>()
    private val _history = ArrayList<ARMEState>()

    val history: List<ARMEState> get() = _history

    // ── Signal computations ───────────────────────────────────────────

    fun computeRT(queryEmb: FloatArray, retrievedEmbs: Array<FloatArray>, scores: DoubleArray? = null): Double {
        if (retrievedEmbs.isEmpty()) return 0.5
        val q = normalize(queryEmb)
        val h = Array(retrievedEmbs.size) { normalize(retrievedEmbs[it]) }
        val k = h.size
        val cosQ = DoubleArray(k) { i -> max(0.0, dot(q, h[i])) }
        val r = if (scores != null) {
            require(scores.size == k) { "scores size ${scores.size} != retrieved size $k" }
            val p = softmax(scores)
            var s = 0.0
            for (i in 0 until k) s += p[i] * cosQ[i]
            s
        } else {
            cosQ.average()
        }
        val d = if (k < 2) {
            1.0
        } else {
            var sum = 0.0
            var count = 0
            for (i in 0 until k) {
                for (j in 0 until k) {
                    if (i == j) continue
                    sum += max(0.0, dot(h[i], h[j]))
                    count++
                }
            }
            1.0 - sum / count
        }
        return alphaR * r + (1.0 - alphaR) * d
    }

    fun computeUT(nodeActivations: Int, totalActivations: Int): Double =
        if (totalActivations > 0) min(1.0, nodeActivations.toDouble() / totalActivations) else 0.0

    fun computeDeltaT(nodeId: String): Double {
        val now = timeNowS()
        val last = lastSeen[nodeId]
        val delta = if (last == null) 1.0 else 1.0 - exp(-(now - last) / tau)
        lastSeen[nodeId] = now
        return delta
    }

    // ── Quadrant-aware e_t ────────────────────────────────────────────

    fun computeET(
        text: String,
        valence: Double? = null,
        arousal: Double? = null,
        dominance: Double? = null,
    ): EmotionalState {
        var v = valence
        var a = arousal
        if (v == null || a == null) {
            val stub = stubClassifier(text)
            v = stub.first
            a = stub.second
        }
        val state = computeETQuadrant(v, a)
        return state.copy(dominance = dominance ?: 0.5)
    }

    private fun stubClassifier(text: String): Pair<Double, Double> {
        val t = text.lowercase()
        var valence = 0.0
        var arousal = 0.3
        for (w in arrayOf("excellent", "amazing", "love", "brilliant", "fascinating", "eureka")) {
            if (w in t) { valence += 0.8; arousal = maxOf(arousal, 0.85); break }
        }
        for (w in arrayOf("good", "interesting", "helpful", "nice", "clear", "thanks")) {
            if (w in t) { valence += 0.4; arousal = maxOf(arousal, 0.5); break }
        }
        for (w in arrayOf("terrible", "hate", "useless", "broken", "wrong", "frustrated", "stuck")) {
            if (w in t) { valence -= 0.8; arousal = maxOf(arousal, 0.75); break }
        }
        for (w in arrayOf("unclear", "not sure", "hmm", "weird", "doesn't work", "confused")) {
            if (w in t) { valence -= 0.3; arousal = maxOf(arousal, 0.45); break }
        }
        for (w in arrayOf("why", "how", "what if", "curious", "wonder", "explain", "?")) {
            if (w in t) { valence += 0.2; arousal = maxOf(arousal, 0.7); break }
        }
        return Pair(max(-1.0, min(1.0, valence)), max(0.0, min(1.0, arousal)))
    }

    private fun signalConfidence(eT: Double, rT: Double, attention: Double = 1.0): Double =
        abs(eT) * rT * attention

    // ── m_t: final modulation scalar ──────────────────────────────────

    /**
     * Novelty gate: delta_t only amplifies in Q1/Q4.
     * Signal-confidence gate: |e_t| * r_t * attention < gateThreshold returns
     * m_t = 1.0 WITHOUT updating the EMA.
     */
    fun computeMT(
        rT: Double,
        uT: Double,
        deltaT: Double,
        emotion: EmotionalState,
        attention: Double = 1.0,
    ): Pair<Double, ARMEState> {
        val deltaGated = if (emotion.quadrant == "Q1" || emotion.quadrant == "Q4") deltaT else 0.0

        val sT = wR * rT + wU * uT + wD * deltaGated + wE * emotion.eT + wDom * emotion.dominance + bias
        val mRaw = exp(sT)
        val mEma = (1.0 - emaLambda) * mPrev + emaLambda * mRaw
        val mClamp = max(mMin, min(mMax, mEma))

        val confidence = signalConfidence(emotion.eT, rT, attention)
        val gated = confidence < gateThreshold

        val mT: Double
        if (gated) {
            mT = 1.0    // neutral — don't amplify or suppress on weak signals
        } else {
            mT = mClamp
            mPrev = mT  // only update EMA when the signal is real
        }

        val state = ARMEState(
            rT = rT, uT = uT, deltaT = deltaT,
            emotion = emotion, sT = sT, mTRaw = mRaw,
            mT = mT, dominance = emotion.dominance, gated = gated,
            timestamp = timeNowS(),
        )
        _history.add(state)
        return Pair(mT, state)
    }

    // ── Main entry point ──────────────────────────────────────────────

    fun step(
        queryEmb: FloatArray,
        retrievedEmbs: Array<FloatArray>,
        userText: String,
        nodeId: String,
        nodeActivations: Int = 1,
        totalActivations: Int = 10,
        retrievalScores: DoubleArray? = null,
        valenceOverride: Double? = null,
        arousalOverride: Double? = null,
        dominanceOverride: Double? = null,
        attention: Double = 1.0,
    ): Pair<Double, ARMEState> {
        val rT = computeRT(queryEmb, retrievedEmbs, retrievalScores)
        val uT = computeUT(nodeActivations, totalActivations)
        val deltaT = computeDeltaT(nodeId)
        val emotion = computeET(userText, valenceOverride, arousalOverride, dominanceOverride)
        return computeMT(rT, uT, deltaT, emotion, attention)
    }

    // ── Hebbian update with eligibility + anti-Hebbian ────────────────

    fun modulatedHebbianUpdate(
        edgeSrc: IntArray,
        edgeDst: IntArray,
        edgeWeight: DoubleArray,
        hPost: Array<FloatArray>,
        mT: Double,
        eta: Double = 0.05,
        lamOja: Double = 0.008,
        wFloor: Double = 0.1,
        wCeil: Double = 5.0,
        eligibility: DoubleArray? = null,
    ): HebbianResult = modulatedHebbianUpdate(
        edgeSrc, edgeDst, edgeWeight, hPost, mT, eta,
        DoubleArray(edgeSrc.size) { lamOja }, wFloor, wCeil, eligibility,
    )

    /**
     * Full Hebbian update with all three improvements.
     *
     * [hPost] is the POST-propagation feature matrix: co-activation and the
     * Oja decay term sum(h_src^2) are computed on conv outputs, matching the
     * Python production path. [lamOja] is per-edge (type-conditioned decay,
     * see TypeProfiles.lamFor). [eligibility] carries the persisted per-edge
     * traces; pass null to start from zero traces. Returns the updated
     * weights, the updated traces (persist these), and stats.
     */
    fun modulatedHebbianUpdate(
        edgeSrc: IntArray,
        edgeDst: IntArray,
        edgeWeight: DoubleArray,
        hPost: Array<FloatArray>,
        mT: Double,
        eta: Double = 0.05,
        lamOja: DoubleArray,
        wFloor: Double = 0.1,
        wCeil: Double = 5.0,
        eligibility: DoubleArray? = null,
    ): HebbianResult {
        val e = edgeSrc.size
        require(edgeDst.size == e && edgeWeight.size == e && lamOja.size == e) {
            "edge arrays must have equal length"
        }
        require(eligibility == null || eligibility.size == e) {
            "eligibility size ${eligibility?.size} != edge count $e"
        }

        val coact = DoubleArray(e)
        val srcSq = DoubleArray(e)
        for (i in 0 until e) {
            val hs = hPost[edgeSrc[i]]
            val hd = hPost[edgeDst[i]]
            var c = 0.0
            var sq = 0.0
            for (d in hs.indices) {
                c += hs[d].toDouble() * hd[d]
                sq += hs[d].toDouble() * hs[d]
            }
            coact[i] = c
            srcSq[i] = sq
        }

        // Stateless eligibility traces: decay + accumulate, caller owns the state.
        val eligVals = DoubleArray(e) { i ->
            (eligibility?.get(i) ?: 0.0) * (1.0 - 1.0 / tauElig) + kappa * max(0.0, coact[i])
        }
        val eligible = BooleanArray(e) { eligVals[it] >= thetaElig }

        // Oja's rule on ALL edges; ARM-E modulation only on eligible edges.
        val newW = DoubleArray(e)
        for (i in 0 until e) {
            val hebb = eta * coact[i]
            val decay = lamOja[i] * edgeWeight[i] * srcSq[i]
            val deltaBase = hebb - decay
            val mFactor = if (eligible[i]) mT else 1.0
            var w = edgeWeight[i] + mFactor * deltaBase
            // Anti-Hebbian normalization on non-eligible edges.
            if (!eligible[i]) w = max(wFloor, w - lamAnti * w)
            newW[i] = max(wFloor, min(wCeil, w))
        }

        var nEligible = 0
        var sumCoact = 0.0
        var sumW = 0.0
        var sumEligW = 0.0
        var sumIneligW = 0.0
        var nIneligible = 0
        var sumAbsDw = 0.0
        var maxAbsDw = 0.0
        for (i in 0 until e) {
            sumCoact += coact[i]
            sumW += newW[i]
            if (eligible[i]) { nEligible++; sumEligW += newW[i] } else { nIneligible++; sumIneligW += newW[i] }
            val dw = abs(newW[i] - edgeWeight[i])
            sumAbsDw += dw
            if (dw > maxAbsDw) maxAbsDw = dw
        }
        val stats = HebbianStats(
            nEligible = nEligible,
            nTotal = e,
            meanCoact = if (e > 0) sumCoact / e else 0.0,
            meanW = if (e > 0) sumW / e else 0.0,
            eligibleMeanW = if (nEligible > 0) sumEligW / nEligible else 0.0,
            ineligibleMeanW = if (nIneligible > 0) sumIneligW / nIneligible else 0.0,
            meanAbsDeltaW = if (e > 0) sumAbsDw / e else 0.0,
            maxAbsDeltaW = maxAbsDw,
        )
        return HebbianResult(newW, eligVals, stats)
    }

    fun resetEma() {
        mPrev = 1.0
    }

    companion object {
        private fun normalize(v: FloatArray): DoubleArray {
            var sum = 0.0
            for (x in v) sum += x.toDouble() * x
            val norm = max(sqrt(sum), 1e-12)
            return DoubleArray(v.size) { v[it] / norm }
        }

        private fun dot(a: DoubleArray, b: DoubleArray): Double {
            var s = 0.0
            for (i in a.indices) s += a[i] * b[i]
            return s
        }

        private fun softmax(scores: DoubleArray): DoubleArray {
            val maxScore = scores.max()
            val exps = DoubleArray(scores.size) { exp(scores[it] - maxScore) }
            val sum = exps.sum()
            return DoubleArray(scores.size) { exps[it] / sum }
        }
    }
}

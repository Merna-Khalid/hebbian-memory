package com.mobilerag.spikes

object SpikeRegistry {
    val all: List<Spike> = listOf(
        OnnxEmbeddingSpike(),
        LadybugGraphSpike(),
        SqliteGraphSpike(),
        LlamaCppSpike(),
        MlKitProbeSpike(),
        TokenizerSpike(),
        EmbeddingGemmaSpike(),
        GlinerSpike(),
        HybridRetrievalSpike(),
        SoakSpike(),
        HebbianMathSpike(),
        HebbianIngestSpike(),
        SecondBrainSimSpike(),
        CorticalConsolidationSpike(),
        ConceptMergeSpike(),
        SleepNowSpike(),
    )

    fun byId(id: String): Spike = all.first { it.id == id }
}

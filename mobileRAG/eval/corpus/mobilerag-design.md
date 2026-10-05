# mobileRAG Design Decisions

The graph store is LadybugDB (a Kùzu fork) with SQLite as the schema of record behind a GraphStore interface. Embeddings use EmbeddingGemma-300M via ONNX Runtime. Generation runs through llama.cpp with a Qwen3 GGUF model.

Gemini Nano was unavailable on the RedMagic 10 Pro because AICore is not installed. The GNN phase comes last, after naive RAG, graph memory, and hybrid retrieval.

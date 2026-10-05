"""
ingest_pipeline.py
The full Hebbian RAG ingest pipeline.

Wires together:
    EmotionClassifier (vad-bert) → ARME → HippocampalGNN → GraphStore

One call to IngestPipeline.ingest() takes a text chunk and:
    1. Embeds it with a sentence transformer
    2. Classifies emotion (valence, arousal, dominance) via vad-bert
    3. Upserts the Concept node to Neo4j
    4. Pulls the local subgraph around the new node
    5. Runs the hippocampal GNN with ARM-E modulation
    6. Writes updated Hebbian weights back to Neo4j
    7. Updates ARM-E stats (lifetime running mean + circular buffer)
    8. Updates SessionStats for Phase 4 w_dom regression

Cortical consolidation is NOT called here — it runs on a separate
schedule (hourly / nightly). See cortical_gnn.py::consolidate().
"""

import uuid
import time
import json
import hashlib
import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from graph_store import GraphStore, ConceptNode, HebbianEdge
from hippocampal_gnn_pyg import HippocampalGNN
from arme_v2 import ARME
from emotion_classifier import EmotionClassifier

try:
    from type_profiles import lam_for
    import concept_identity
except ImportError:
    from core.type_profiles import lam_for
    from core import concept_identity

# Bootstrap edge weight for content that failed validation (extraction
# fallback / unverified). Still connected for retrieval, but weakly —
# validation-gated plasticity: unverified associations must earn strength.
UNVALIDATED_BOOTSTRAP_W = 0.6


# ── Ingest result dataclass ───────────────────────────────────────────

@dataclass
class IngestResult:
    node_id      : str
    label        : str
    m_t          : float
    gated        : bool
    quadrant     : str
    valence      : float
    arousal      : float
    dominance    : float
    mean_w_before: float
    mean_w_after : float
    delta_w_mean : float
    n_edges      : int
    subgraph_size: int
    elapsed_ms   : float
    reused       : bool = False   # re-activated an existing concept instead of creating one


# ── Pipeline ──────────────────────────────────────────────────────────

class IngestPipeline:
    """
    Full Hebbian RAG ingest pipeline.

    Usage:
        pipeline = IngestPipeline(
            graph_store   = GraphStore("neo4j://localhost:7687", "neo4j", "pw"),
            embedding_fn  = your_embedding_function,   # text → List[float]
            summary_fn    = your_summary_function,     # text → str (LLM call)
        )
        session_id = pipeline.start_session()

        result = pipeline.ingest(
            text         = "Hebbian learning strengthens synaptic connections...",
            label        = "Hebbian plasticity",
            concept_type = "fact",
            source_type  = "document",
            source_uri   = "https://doi.org/10.1037/h0042519",
            user_text    = "This is fascinating, tell me more!",
            session_id   = session_id,
        )

        pipeline.end_session(session_id)
    """

    def __init__(
        self,
        graph_store   : GraphStore,
        embedding_fn,                    # callable: str → List[float]
        summary_fn    = None,            # callable: str → str, optional
        feature_dim   : int   = 1024,
        eta           : float = 0.05,
        lam           : float = 0.008,
        w_floor       : float = 0.1,
        w_ceil        : float = 5.0,
        subgraph_hops : int   = 2,
        subgraph_max  : int   = 100,
        min_hebb_weight     : float = 0.3,   # floor for pulling subgraph edges
        bootstrap_k         : int   = 5,     # how many neighbors to connect on first ingest
        bootstrap_threshold : float = 0.3,   # min cosine similarity to create bootstrap edge
        device              : str   = "auto",
        emotion_classifier  = None,   # pass pre-built EmotionClassifier to avoid double load
        scheduler           = None,   # ConsolidationScheduler — if None, no auto-consolidation
    ):
        self.store          = graph_store
        self.embedding_fn   = embedding_fn
        self.summary_fn     = summary_fn
        self.subgraph_hops       = subgraph_hops
        self.subgraph_max        = subgraph_max
        self.min_hebb_weight     = min_hebb_weight
        self.bootstrap_k         = bootstrap_k
        self.bootstrap_threshold = bootstrap_threshold

        # Components
        # Accept a pre-built EmotionClassifier (e.g. from model_setup.py)
        # to avoid loading vad-bert twice when model_setup already loaded it.
        self.emotion_clf = emotion_classifier if emotion_classifier is not None \
                           else EmotionClassifier(device=device)
        self.arme        = ARME()
        self.hipp_gnn    = HippocampalGNN(
            feature_dim = feature_dim,
            eta         = eta,
            lam         = lam,
            w_floor     = w_floor,
            w_ceil      = w_ceil,
        )

        self.scheduler = scheduler

        # Session state
        self._session_id          : Optional[str] = None
        self._session_activations : int = 0
        self._node_activation_counts: dict = {}   # node_id → count this session

    # ── Session management ────────────────────────────────────────────

    def start_session(self, trigger: str = "time_window") -> str:
        """Start a new session. Returns session_id."""
        self._session_id           = self.store.start_session(trigger)
        self._session_activations  = 0
        self._node_activation_counts = {}
        self.arme.reset_ema()
        print(f"Session started: {self._session_id} (trigger={trigger})")
        return self._session_id

    def end_session(self, session_id: str):
        """Close the session. Triggers final consolidation if scheduler attached."""
        self.store.end_session(session_id)
        if self.scheduler is not None:
            self.scheduler.on_session_end()
        self._session_id = None
        print(f"Session closed: {session_id}  "
              f"({self._session_activations} activations)")

    # ── Main ingest entry point ───────────────────────────────────────

    def ingest(
        self,
        text         : str,
        label        : str,
        concept_type : str  = "fact",
        source_type  : str  = "document",
        source_uri   : Optional[str] = None,
        user_text    : str  = "",        # user message that triggered this ingest
        session_id   : Optional[str] = None,
        node_id      : Optional[str] = None,   # supply to update existing node
        attention    : Optional[float] = None,  # [0,1] engagement signal, e.g. from a BCI
        validated    : bool  = True,   # False when extraction fell back / unverified
    ) -> IngestResult:
        """
        Ingest one text chunk into the Hebbian RAG memory.

        Steps:
            1. Embed text
            2. Generate summary (if summary_fn provided)
            3. Classify emotion via vad-bert → valence, arousal, dominance
            4. Upsert Concept node to Neo4j
            5. Pull local subgraph around new node
            6. Compute ARM-E modulation m_t
            7. Propagate through hippocampal GNN, then ARM-E-modulated
               Hebbian update (eligibility gating + anti-Hebbian decay)
            8. Write updated Hebbian weights + eligibility to Neo4j
            9. Update ARM-E stats on nodes
            10. Update SessionStats
        """
        t0 = time.time()
        sid = session_id or self._session_id
        attn = attention if attention is not None else 1.0

        # ── 1. Embed ──────────────────────────────────────────────────
        embedding = self.embedding_fn(text)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.tolist()

        # ── 1b. Identity: a concept the graph already holds? ──────────
        # One vector search serves identity and bootstrap. It runs before the
        # node is written, so for a new node it can't contain the node itself
        # (bootstrap below takes the top bootstrap_k others — the same set as
        # before). An explicit node_id skips resolution.
        similar = self.store.vector_search(
            query_embedding = embedding,
            top_k           = self.bootstrap_k + 1,
        )
        existing = None
        if node_id is None:
            existing = concept_identity.resolve(
                label, concept_type, validated,
                label_matches = self.store.find_by_label_key(
                    concept_identity.label_key(label)),
                vector_hits   = similar,
            )
        reused = existing is not None
        nid = existing.node_id if reused else (node_id or str(uuid.uuid4()))

        # ── 3. Classify emotion ───────────────────────────────────────
        vad = self.emotion_clf(user_text) if user_text.strip() else None
        valence   = vad.valence   if vad else 0.0
        arousal   = vad.arousal   if vad else 0.5
        dominance = vad.dominance if vad else 0.5

        # ── 2 + 4. Summary + upsert (new nodes only) ──────────────────
        # A reused node keeps its stored embedding — possibly a consolidated
        # cortical one — and its first description; skipping the summary also
        # saves an LLM call per repeat.
        if not reused:
            if self.summary_fn is not None:
                summary = self.summary_fn(text)
            else:
                # Truncate to first 2 sentences as fallback
                sentences = text.replace("?", ".").replace("!", ".").split(".")
                summary   = ". ".join(s.strip() for s in sentences[:2] if s.strip()) + "."
            concept = ConceptNode(
                node_id          = nid,
                label            = label,
                text_raw         = text,
                text_summary     = summary,
                concept_type     = concept_type,
                source_type      = source_type,
                source_uri       = source_uri,
                created_at       = time.time(),
                updated_at       = time.time(),
                activation_count = 0,
                embedding        = embedding,
            )
            self.store.upsert_concept(concept)
        self.store.increment_activation(nid)

        # Track session activation counts for u_t
        self._session_activations += 1
        self._node_activation_counts[nid] = (
            self._node_activation_counts.get(nid, 0) + 1
        )

        # ── 4b. Bootstrap edges to similar existing nodes ────────────
        # Problem: a brand-new node has no edges, so the GNN never fires.
        # Fix: connect it to similar existing nodes with hebb_weight=1.0
        # (starting weight); the GNN updates them on subsequent ingests.
        # A reused node is bootstrapped only if it has no usable edges.
        bootstrap_w = 1.0 if validated else UNVALIDATED_BOOTSTRAP_W

        def bootstrap():
            others = [(n, sc) for n, sc in similar if n.node_id != nid]
            for neighbor, score in others[:self.bootstrap_k]:
                if score < self.bootstrap_threshold:
                    continue   # too dissimilar
                # Create bidirectional edges — Hebbian learning is symmetric
                # at initialization (will diverge as weights update).
                # Unvalidated content starts weaker and must earn strength.
                for src_id, dst_id in [(nid, neighbor.node_id),
                                        (neighbor.node_id, nid)]:
                    self.store.upsert_edge(HebbianEdge(
                        src_id              = src_id,
                        dst_id              = dst_id,
                        hebb_weight         = bootstrap_w,
                        eligibility         = 0.0,
                        causal_score        = 0.0,
                        co_activation_count = 0,
                        last_updated        = time.time(),
                        layer               = "hippocampal",
                    ))

        # ── 5. Pull local subgraph ────────────────────────────────────
        def local_subgraph():
            return self.store.get_local_subgraph(
                seed_node_ids  = [nid],
                hops           = self.subgraph_hops,
                max_nodes      = self.subgraph_max,
                layer          = "hippocampal",
                min_weight     = self.min_hebb_weight,
            )

        if not reused:
            bootstrap()
        subgraph = local_subgraph()
        if reused and len(subgraph.edges) == 0:
            bootstrap()
            subgraph = local_subgraph()

        # If subgraph is empty or only the seed node, nothing to update
        if len(subgraph.nodes) < 2 or len(subgraph.edges) == 0:
            # Still compute ARM-E for stats, but skip GNN
            m_t, arme_state = self.arme.step(
                query_emb          = torch.tensor(embedding),
                retrieved_embs     = torch.zeros(1, len(embedding)),
                user_text          = user_text,
                node_id            = nid,
                node_activations   = self._node_activation_counts[nid],
                total_activations  = max(self._session_activations, 1),
                valence_override   = valence,
                arousal_override   = arousal,
                dominance_override = dominance,
                attention          = attn,
            )
            self._update_stats(sid, nid, arme_state, delta_w_mean=0.0)
            if self.scheduler is not None:
                self.scheduler.on_ingest()
            elapsed = (time.time() - t0) * 1000
            return IngestResult(
                node_id=nid, label=label, m_t=m_t,
                gated=arme_state.gated, quadrant=arme_state.emotion.quadrant,
                valence=valence, arousal=arousal, dominance=dominance,
                mean_w_before=0.0, mean_w_after=0.0, delta_w_mean=0.0,
                n_edges=0, subgraph_size=1, elapsed_ms=elapsed, reused=reused,
            )

        # ── 6. Build PyG Data from subgraph ───────────────────────────
        # Map global node_ids to local indices for the GNN
        node_id_list  = [n.node_id for n in subgraph.nodes]
        id_to_idx     = {nid_: i for i, nid_ in enumerate(node_id_list)}

        H = torch.tensor(
            [n.embedding for n in subgraph.nodes], dtype=torch.float
        )
        # Normalize embeddings to unit sphere before Hebbian
        H = F.normalize(H, dim=1)

        # Filter to edges whose endpoints are both in the capped node set.
        # Build the list ONCE so edge positions stay aligned everywhere
        # downstream (weights, eligibility traces, write-back).
        valid_edges = [e for e in subgraph.edges
                       if e.src_id in id_to_idx and e.dst_id in id_to_idx]
        src_idx = [id_to_idx[e.src_id] for e in valid_edges]
        dst_idx = [id_to_idx[e.dst_id] for e in valid_edges]
        weights = [e.hebb_weight for e in valid_edges]

        if not src_idx:
            if self.scheduler is not None:
                self.scheduler.on_ingest()
            elapsed = (time.time() - t0) * 1000
            return IngestResult(
                node_id=nid, label=label, m_t=1.0,
                gated=True, quadrant="Q4",
                valence=valence, arousal=arousal, dominance=dominance,
                mean_w_before=0.0, mean_w_after=0.0, delta_w_mean=0.0,
                n_edges=0, subgraph_size=len(subgraph.nodes), elapsed_ms=elapsed, reused=reused,
            )

        edge_index  = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        mean_w_before = edge_weight.mean().item()

        # ── 7. ARM-E modulation ───────────────────────────────────────
        # Retrieved embeddings for r_t computation
        retrieved_embs = H[[id_to_idx[n.node_id]
                             for n in subgraph.nodes
                             if n.node_id != nid][:10]]  # cap at 10

        query_emb = torch.tensor(embedding, dtype=torch.float)

        m_t, arme_state = self.arme.step(
            query_emb          = query_emb,
            retrieved_embs     = retrieved_embs,
            user_text          = user_text,
            node_id            = nid,
            node_activations   = self._node_activation_counts[nid],
            total_activations  = max(self._session_activations, 1),
            valence_override   = valence,
            arousal_override   = arousal,
            dominance_override = dominance,
            attention          = attn,
        )

        # Validation gating (Kairos-style): unverified content never gets
        # an amplifying m_t — it can encode at baseline or below, but a
        # hallucinated/fallback turn cannot be deep-encoded.
        if not validated:
            m_t = min(m_t, 1.0)

        # ── 8. Propagate, then modulated Hebbian update ───────────────
        # Message passing first — co-activation is computed on the
        # POST-propagation features, so the GNN actually shapes the
        # weight update (with raw embeddings the conv was decorative).
        H_prop = self.hipp_gnn.propagate(H, edge_index, edge_weight)

        # Eligibility traces are persisted per-edge in Neo4j. Pass them
        # in so gating uses the same state we write back afterwards —
        # local subgraph indices are meaningless across ingests, so
        # ARM-E's internal EligibilityTracker can't be used here.
        elig_in = torch.tensor([e.eligibility for e in valid_edges],
                               dtype=torch.float)

        # Type-conditioned Oja decay: each edge's λ scales by its source
        # concept's decay profile — vocabulary erodes fast, grammar rules
        # are sticky (see type_profiles.py).
        nid_to_type = {n.node_id: n.concept_type for n in subgraph.nodes}
        lam_edges = torch.tensor([
            lam_for(nid_to_type.get(e.src_id, "fact"), self.hipp_gnn.lam)
            for e in valid_edges
        ], dtype=torch.float)

        new_w_t, gnn_stats = self.arme.modulated_hebbian_update(
            edge_index  = edge_index,
            edge_weight = edge_weight,
            H_pre       = H_prop,
            m_t         = m_t,
            eta         = self.hipp_gnn.eta,
            lam_oja     = lam_edges,
            w_floor     = self.hipp_gnn.w_floor,
            w_ceil      = self.hipp_gnn.w_ceil,
            eligibility = elig_in,
        )
        mean_w_after = gnn_stats["mean_w"]
        delta_w_mean = mean_w_after - mean_w_before

        # ── 9. Write updated weights + eligibility to Neo4j ───────────
        new_weights       = new_w_t.tolist()
        new_eligibilities = gnn_stats["eligibility"].tolist()

        weight_updates = [
            (e.src_id, e.dst_id, new_weights[k], new_eligibilities[k], "hippocampal")
            for k, e in enumerate(valid_edges)
        ]
        if weight_updates:
            self.store.update_hebb_weights(weight_updates)

        # ── 10. Update stats ──────────────────────────────────────────
        self._update_stats(sid, nid, arme_state, delta_w_mean)

        elapsed = (time.time() - t0) * 1000
        # Notify scheduler — triggers consolidation every N ingests
        if self.scheduler is not None:
            self.scheduler.on_ingest()

        return IngestResult(
            node_id       = nid,
            label         = label,
            m_t           = m_t,
            gated         = arme_state.gated,
            quadrant      = arme_state.emotion.quadrant,
            valence       = valence,
            arousal       = arousal,
            dominance     = dominance,
            mean_w_before = mean_w_before,
            mean_w_after  = mean_w_after,
            delta_w_mean  = delta_w_mean,
            n_edges       = len(weight_updates),
            subgraph_size = len(subgraph.nodes),
            elapsed_ms    = elapsed,
            reused        = reused,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _update_stats(self, session_id, node_id, arme_state, delta_w_mean):
        """Write ARM-E stats to node and SessionStats."""
        self.store.update_concept_arme_stats(
            node_id   = node_id,
            m_t       = arme_state.m_t,
            dominance = arme_state.dominance,
            r_t       = arme_state.r_t,
        )
        if session_id:
            self.store.upsert_session_stats(
                session_id = session_id,
                node_id    = node_id,
                m_t        = arme_state.m_t,
                dominance  = arme_state.dominance,
                r_t        = arme_state.r_t,
                delta_w    = delta_w_mean,
            )
            self.store.increment_session_activations(session_id)


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math, random

    print("=== Ingest Pipeline Demo ===\n")

    # ── Stub embedding function ───────────────────────────────────────
    # Replace with your actual sentence transformer in production:
    #
    #   from sentence_transformers import SentenceTransformer
    #   model = SentenceTransformer("all-mpnet-base-v2")
    #   embedding_fn = lambda text: model.encode(text).tolist()
    #
    def stub_embedding_fn(text: str, dim: int = 1024) -> list:
        """Deterministic fake embedding based on text hash."""
        seed = int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:4], "big"
        )
        random.seed(seed)
        v = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x*x for x in v))
        return [x / norm for x in v]

    # ── Stub summary function ─────────────────────────────────────────
    # Replace with an actual LLM call in production, e.g. any
    # OpenAI-compatible chat endpoint (local server, hosted API, ...):
    #
    #   from openai import OpenAI
    #   client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
    #   def summary_fn(text):
    #       resp = client.chat.completions.create(
    #           model="local-model", max_tokens=100,
    #           messages=[{"role":"user",
    #                      "content":f"Summarise in one sentence: {text}"}])
    #       return resp.choices[0].message.content
    #
    def stub_summary_fn(text: str) -> str:
        sentences = text.replace("?",".").replace("!",".").split(".")
        return ". ".join(s.strip() for s in sentences[:2] if s.strip()) + "."

    # ── Connect ───────────────────────────────────────────────────────
    store = GraphStore(
        uri      = "neo4j://localhost:7687",
        user     = "neo4j",
        password = "password",    # ← change to your Neo4j Desktop password
    )

    pipeline = IngestPipeline(
        graph_store  = store,
        embedding_fn = stub_embedding_fn,
        summary_fn   = stub_summary_fn,
        feature_dim  = 1024,
    )

    # ── Session ───────────────────────────────────────────────────────
    session_id = pipeline.start_session(trigger="manual")

    # ── Ingest documents with different emotional contexts ────────────
    documents = [
        {
            "text"        : "Hebbian learning states that neurons that fire together wire together. "
                            "Synaptic connections strengthen when pre- and post-synaptic neurons "
                            "are co-activated repeatedly.",
            "label"       : "Hebbian plasticity",
            "concept_type": "fact",
            "source_type" : "document",
            "source_uri"  : "https://doi.org/10.1037/h0042519",
            "user_text"   : "This is fascinating! I never knew synapses worked this way.",
        },
        {
            "text"        : "Oja's rule extends Hebbian learning with a weight decay term: "
                            "Δw = η(h_i h_j - λ w_ij h_i²). This prevents weights from "
                            "growing without bound.",
            "label"       : "Oja's rule",
            "concept_type": "fact",
            "source_type" : "web",
            "source_uri"  : "https://neurophysics.ucsd.edu/courses/physics_171/Oja_1982.pdf",
            "user_text"   : "Interesting, how does this prevent explosion?",
        },
        {
            "text"        : "Graph Attention Networks use learned attention coefficients "
                            "α_ij to weight neighbor messages. Unlike fixed weights, "
                            "attention is computed from node feature content.",
            "label"       : "Graph Attention Networks",
            "concept_type": "fact",
            "source_type" : "web",
            "source_uri"  : "https://arxiv.org/abs/1710.10903",
            "user_text"   : "I'm confused, how is this different from regular GNNs?",
        },
        {
            "text"        : "The hippocampus encodes episodic memories rapidly through "
                            "pattern separation and completion. It binds contextual "
                            "elements into coherent memory traces.",
            "label"       : "Hippocampal encoding",
            "concept_type": "fact",
            "source_type" : "document",
            "source_uri"  : "https://www.nature.com/articles/nrn2963",
            "user_text"   : "Amazing, this maps perfectly to our GNN architecture!",
        },
        {
            "text"        : "Robotic arm trajectory planning using reinforcement learning "
                            "requires balancing exploration and exploitation. The agent "
                            "must learn to reach target positions without collision.",
            "label"       : "Robot arm trajectory RL",
            "concept_type": "procedure",
            "source_type" : "synthesized",
            "source_uri"  : None,
            "user_text"   : "This is relevant to our robot project.",
        },
    ]

    print(f"\n{'#':<3} {'label':<32} {'Q':<4} {'m_t':<7} {'gated':<7} "
          f"{'Δw':<8} {'edges':<7} {'ms':<7} {'V':>6} {'A':>6} {'D':>6}")
    print("-" * 95)

    results = []
    for i, doc in enumerate(documents):
        result = pipeline.ingest(
            text         = doc["text"],
            label        = doc["label"],
            concept_type = doc["concept_type"],
            source_type  = doc["source_type"],
            source_uri   = doc["source_uri"],
            user_text    = doc["user_text"],
            session_id   = session_id,
        )
        results.append(result)

        gated_str = "YES" if result.gated else "no"
        print(f"{i+1:<3} {result.label:<32} {result.quadrant:<4} "
              f"{result.m_t:<7.3f} {gated_str:<7} "
              f"{result.delta_w_mean:<+8.4f} {result.n_edges:<7} "
              f"{result.elapsed_ms:<7.1f} "
              f"{result.valence:>+6.3f} {result.arousal:>6.3f} {result.dominance:>6.3f}")

    pipeline.end_session(session_id)

    # ── LLM retrieval test ────────────────────────────────────────────
    print("\n=== LLM retrieval after ingest ===")
    print("Query: 'How do neural networks learn associations?'\n")

    query_emb = stub_embedding_fn("How do neural networks learn associations?")
    seed_ids  = [r.node_id for r in results[:2]]

    retrieved = store.retrieve_for_llm(
        query_embedding = query_emb,
        seed_node_ids   = seed_ids,
        top_k           = 3,
    )

    for r in retrieved:
        print(f"[{r['concept_type']}] {r['label']}")
        print(f"  score={r['score']:.3f}  "
              f"cos={r['cos_sim']:.3f}  assoc={r['assoc']:.3f}  causal={r['causal']:.3f}")
        print(f"  summary: {r['text_summary'][:80]}...")
        print(f"  source:  {r['source_type']} — {r['source_uri']}")
        print()

    # ── Context string for LLM ────────────────────────────────────────
    print("=== Context string the LLM would receive ===\n")
    context_parts = []
    for r in retrieved:
        src = f" (source: {r['source_uri']})" if r['source_uri'] else ""
        context_parts.append(
            f"[{r['concept_type'].upper()}] {r['label']}{src}\n{r['text_summary']}"
        )
    context = "\n\n".join(context_parts)
    print(context)

    store.close()
    print("\n\nDone — full ingest pipeline working.")
    print("Next: swap stub_embedding_fn with a real sentence transformer")
    print("      swap stub_summary_fn with a real LLM call")
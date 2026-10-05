"""
GraphStore — the single Neo4j interface for the Hebbian RAG system.

Every component (hippocampal GNN, cortical GNN, ARM-E, LLM retrieval)
talks to this class. No raw Cypher anywhere else.

Schema:
    (:Concept)                       — knowledge nodes
    (:Concept)-[:ASSOCIATED_WITH]->  — Hebbian edges with weights
    (:Session)                       — time-windowed or trigger-based
    (:SessionStats)-[:TRACKED_IN]->  — Option A per-(session, concept) aggregate
    (:Session)-[:AGGREGATES]->       — links session to its stats

Embedding storage: LIST<FLOAT> + vector index (Community Edition compatible).
"""

import json
import uuid
import time
import math
import warnings
from typing import Optional, Any
from dataclasses import dataclass, field

from neo4j import GraphDatabase, Driver

# Min normalized spread (peak = 1) for a node outside the vector pool to become a
# retrieval candidate in retrieve_for_llm.
SPREAD_CANDIDATE_MIN = 0.3


def _neo4j_cosine(a: list[float], b: list[float]) -> Optional[float]:
    """Cosine on the vector index's own scale, (1 + cos) / 2, so spread-only
    candidates are comparable with vector_search scores."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return (1.0 + dot / (na * nb)) / 2.0

try:
    from type_profiles import staleness as _type_staleness
    from concept_identity import label_key as _label_key
except ImportError:          # package-style import
    from core.type_profiles import staleness as _type_staleness
    from core.concept_identity import label_key as _label_key


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class ConceptNode:
    """
    Represents a Concept node as returned from Neo4j.
    All fields map 1:1 to node properties.
    """
    node_id         : str
    label           : str
    text_raw        : str
    text_summary    : str
    concept_type    : str           # "fact"|"entity"|"procedure"|"event"|"question"
    source_type     : str           # "document"|"web"|"sensor"|"user"|"synthesized"
    source_uri      : Optional[str]
    created_at      : float
    updated_at      : float
    activation_count: int
    embedding       : list[float]
    mean_m_t        : float = 1.0
    mean_dominance  : float = 0.5
    mean_r_t        : float = 0.5
    recent_events   : list[dict] = field(default_factory=list)


@dataclass
class HebbianEdge:
    """Represents an ASSOCIATED_WITH relationship."""
    src_id              : str
    dst_id              : str
    hebb_weight         : float
    eligibility         : float
    causal_score        : float
    co_activation_count : int
    last_updated        : float
    layer               : str    # "hippocampal" | "cortical"


@dataclass
class Subgraph:
    """Local neighborhood returned for GNN processing."""
    nodes : list[ConceptNode]
    edges : list[HebbianEdge]


# ── GraphStore ────────────────────────────────────────────────────────

class GraphStore:
    """
    Neo4j interface for the Hebbian RAG system.

    Usage:
        store = GraphStore("bolt://localhost:7687", "neo4j", "your_password")
        store.setup_schema()          # run once — creates indexes + constraints
        store.close()                 # clean up driver
    """

    RECENT_EVENTS_MAX = 20    # circular buffer size
    W_CEIL            = 5.0   # Hebbian weight ceiling (matches w_ceil elsewhere)

    def __init__(
        self,
        uri      : str = "bolt://localhost:7687",
        user     : str = "neo4j",
        password : str = "password",
        database : str = "neo4j",
    ):
        self._driver   = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        print(f"GraphStore connected → {uri} / {database}")

    def close(self):
        self._driver.close()

    def _run(self, query: str, params: dict = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    # ── Schema setup ──────────────────────────────────────────────────

    def setup_schema(self):
        """
        Create indexes and constraints. Run once on first startup.
        Safe to re-run — uses IF NOT EXISTS throughout.

        Vector index uses LIST<FLOAT> (Community Edition compatible).
        Dimension 1024 matches BAAI/bge-m3 (cross-lingual EN+JA).
        Change to match your actual embedding model output size.
        """
        statements = [
            # Uniqueness constraints
            "CREATE CONSTRAINT concept_id IF NOT EXISTS "
            "FOR (c:Concept) REQUIRE c.node_id IS UNIQUE",

            "CREATE CONSTRAINT session_id IF NOT EXISTS "
            "FOR (s:Session) REQUIRE s.session_id IS UNIQUE",

            # Regular indexes for frequent lookups
            "CREATE INDEX concept_type IF NOT EXISTS "
            "FOR (c:Concept) ON (c.concept_type)",

            "CREATE INDEX concept_source IF NOT EXISTS "
            "FOR (c:Concept) ON (c.source_type)",

            "CREATE INDEX concept_updated IF NOT EXISTS "
            "FOR (c:Concept) ON (c.updated_at)",

            "CREATE INDEX session_started IF NOT EXISTS "
            "FOR (s:Session) ON (s.started_at)",

            # Concept identity lookup (concept_identity.label_key)
            "CREATE INDEX concept_label_key IF NOT EXISTS "
            "FOR (c:Concept) ON (c.label_key)",

            # Vector index on embedding — ANN search
            # Community Edition: works on LIST<FLOAT>, no block format
            # similarityFunction: "cosine" matches normalized sentence embeddings
            """CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
               FOR (c:Concept) ON c.embedding
               OPTIONS {
                 indexConfig: {
                   `vector.dimensions`: 1024,
                   `vector.similarity_function`: 'cosine'
                 }
               }""",
        ]

        for stmt in statements:
            try:
                self._run(stmt)
            except Exception as e:
                # Index already exists — fine
                if "already exists" not in str(e).lower():
                    print(f"Schema warning: {e}")

        self._backfill_label_keys()
        print("Schema ready.")

    def _backfill_label_keys(self):
        """One-time: give pre-identity nodes a label_key. Computed in Python —
        Cypher has no NFKC normalization. No-op once every node has one."""
        rows = self._run(
            "MATCH (c:Concept) WHERE c.label_key IS NULL "
            "RETURN c.node_id AS nid, c.label AS label"
        )
        if not rows:
            return
        self._run(
            "UNWIND $rows AS r MATCH (c:Concept {node_id: r.nid}) SET c.label_key = r.key",
            {"rows": [{"nid": r["nid"], "key": _label_key(r["label"] or "")} for r in rows]},
        )
        print(f"Backfilled label_key on {len(rows)} concepts.")

    # ── Concept CRUD ──────────────────────────────────────────────────

    def upsert_concept(self, concept: ConceptNode) -> str:
        """
        Create or update a Concept node.
        Returns node_id.

        MERGE on node_id: safe to call repeatedly — won't duplicate.
        SET updates all mutable fields; created_at is only set on first create.
        """
        query = """
        MERGE (c:Concept {node_id: $node_id})
        ON CREATE SET
            c.label          = $label,
            c.label_key      = $label_key,
            c.text_raw       = $text_raw,
            c.text_summary   = $text_summary,
            c.concept_type   = $concept_type,
            c.source_type    = $source_type,
            c.source_uri     = $source_uri,
            c.created_at     = $now,
            c.updated_at     = $now,
            c.activation_count = 0,
            c.embedding      = $embedding,
            c.mean_m_t       = 1.0,
            c.mean_dominance = 0.5,
            c.mean_r_t       = 0.5,
            c.recent_events  = '[]'
        ON MATCH SET
            c.label          = $label,
            c.label_key      = $label_key,
            c.text_raw       = $text_raw,
            c.text_summary   = $text_summary,
            c.concept_type   = $concept_type,
            c.source_type    = $source_type,
            c.source_uri     = $source_uri,
            c.updated_at     = $now,
            c.embedding      = $embedding
        RETURN c.node_id AS node_id
        """
        params = {
            "node_id"      : concept.node_id,
            "label"        : concept.label,
            "label_key"    : _label_key(concept.label),
            "text_raw"     : concept.text_raw,
            "text_summary" : concept.text_summary,
            "concept_type" : concept.concept_type,
            "source_type"  : concept.source_type,
            "source_uri"   : concept.source_uri,
            "embedding"    : concept.embedding,
            "now"          : time.time(),
        }
        result = self._run(query, params)
        return result[0]["node_id"]

    def get_concept(self, node_id: str) -> Optional[ConceptNode]:
        """Fetch a single concept by ID."""
        query = "MATCH (c:Concept {node_id: $node_id}) RETURN c"
        result = self._run(query, {"node_id": node_id})
        if not result:
            return None
        return self._row_to_concept(result[0]["c"])

    def find_by_label_key(self, key: str) -> list[ConceptNode]:
        """Concepts whose normalized label equals key (any type) — the identity
        lookup ingest runs before creating a node."""
        if not key:
            return []
        rows = self._run(
            "MATCH (c:Concept {label_key: $key}) RETURN c", {"key": key}
        )
        return [self._row_to_concept(r["c"]) for r in rows]

    def update_concept_embeddings(self, embeddings: dict[str, list[float]]):
        """
        Batch-replace concept embeddings after cortical consolidation.
        Deliberately leaves updated_at alone: it means "last seen" to the
        practice queue, and a consolidation run is not an exposure.
        """
        if not embeddings:
            return
        self._run(
            "UNWIND $updates AS u MATCH (c:Concept {node_id: u.node_id}) "
            "SET c.embedding = u.embedding",
            {"updates": [{"node_id": k, "embedding": v} for k, v in embeddings.items()]},
        )

    def increment_activation(self, node_id: str):
        """Increment activation_count. Called every time a node is retrieved."""
        self._run(
            "MATCH (c:Concept {node_id: $nid}) "
            "SET c.activation_count = c.activation_count + 1, "
            "    c.updated_at = $now",
            {"nid": node_id, "now": time.time()}
        )

    # ── Vector search ─────────────────────────────────────────────────

    def vector_search(
        self,
        query_embedding : list[float],
        top_k           : int = 10,
        concept_type    : Optional[str] = None,
    ) -> list[tuple[ConceptNode, float]]:
        """
        ANN search using the vector index.
        Returns list of (ConceptNode, similarity_score) sorted by score desc.

        concept_type filter is applied post-retrieval — Neo4j Community doesn't
        support filtered vector search natively. For large graphs consider
        over-fetching (top_k * 3) then filtering in Python.
        """
        query = """
        CALL db.index.vector.queryNodes('concept_embedding', $top_k, $embedding)
        YIELD node, score
        RETURN node AS c, score
        ORDER BY score DESC
        """
        fetch_k = top_k * 3 if concept_type else top_k
        # Suppress Neo4j deprecation warning for vector search
        # (db.index.vector.queryNodes still works, just deprecated in >= 5.19)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self._run(query, {
                "embedding": query_embedding,
                "top_k"    : fetch_k,
            })

        out = []
        for row in results:
            node = self._row_to_concept(row["c"])
            if concept_type and node.concept_type != concept_type:
                continue
            out.append((node, float(row["score"])))
            if len(out) >= top_k:
                break

        return out

    # ── Hebbian edge CRUD ─────────────────────────────────────────────

    def upsert_edge(self, edge: HebbianEdge):
        """
        Create an ASSOCIATED_WITH relationship, or re-assert an existing one
        without ever weakening it: hebb_weight = max(existing, new),
        co_activation_count + 1, eligibility and causal_score kept.
        MERGE on (src, dst, layer) — distinct hippocampal/cortical edges.

        (It used to overwrite everything: once concepts are reused, re-extracting
        a relation would knock a trained 4.8 edge back to 1.5 and zero its
        causal score.)
        """
        query = """
        MATCH (src:Concept {node_id: $src_id})
        MATCH (dst:Concept {node_id: $dst_id})
        MERGE (src)-[r:ASSOCIATED_WITH {layer: $layer}]->(dst)
        ON CREATE SET
            r.hebb_weight         = $hebb_weight,
            r.eligibility         = $eligibility,
            r.causal_score        = $causal_score,
            r.co_activation_count = $co_activation_count,
            r.last_updated        = $now
        ON MATCH SET
            r.hebb_weight         = CASE WHEN $hebb_weight > r.hebb_weight
                                         THEN $hebb_weight ELSE r.hebb_weight END,
            r.co_activation_count = r.co_activation_count + 1,
            r.last_updated        = $now
        """
        self._run(query, {
            "src_id"             : edge.src_id,
            "dst_id"             : edge.dst_id,
            "hebb_weight"        : edge.hebb_weight,
            "eligibility"        : edge.eligibility,
            "causal_score"       : edge.causal_score,
            "co_activation_count": edge.co_activation_count,
            "layer"              : edge.layer,
            "now"                : time.time(),
        })

    def update_hebb_weights(self, weight_updates: list[tuple[str, str, float, float, str]]):
        """
        Batch update Hebbian weights after a GNN step.
        Faster than calling upsert_edge in a loop for large subgraphs.

        weight_updates: list of (src_id, dst_id, new_weight, new_eligibility, layer)
        """
        query = """
        UNWIND $updates AS u
        MATCH (src:Concept {node_id: u.src_id})-[r:ASSOCIATED_WITH {layer: u.layer}]
              ->(dst:Concept {node_id: u.dst_id})
        SET r.hebb_weight   = u.weight,
            r.eligibility   = u.eligibility,
            r.last_updated  = $now,
            r.co_activation_count = r.co_activation_count + 1
        """
        updates = [
            {"src_id": s, "dst_id": d, "weight": w,
             "eligibility": e, "layer": l}
            for s, d, w, e, l in weight_updates
        ]
        self._run(query, {"updates": updates, "now": time.time()})

    def update_causal_scores(
        self,
        score_updates: list[tuple[str, str, float]],
        layer        : str = "cortical",
    ):
        """
        Batch update causal scores after cortical GNN consolidation.
        score_updates: list of (src_id, dst_id, causal_score)

        Consolidation passes layer="hippocampal" — the edges it trained on.
        (It used to match only 'cortical' edges, which nothing creates, so
        causal was always 0.) last_updated is left alone: a consolidation
        write is not a use of the edge.
        """
        query = """
        UNWIND $updates AS u
        MATCH (src:Concept {node_id: u.src_id})-[r:ASSOCIATED_WITH {layer: $layer}]
              ->(dst:Concept {node_id: u.dst_id})
        SET r.causal_score = u.score
        """
        updates = [{"src_id": s, "dst_id": d, "score": sc}
                   for s, d, sc in score_updates]
        self._run(query, {"updates": updates, "layer": layer})

    def reinforce_co_retrieved(
        self,
        node_ids : list[str],
        m_t      : float = 1.0,
        eta      : float = 0.02,
        lam      : float = 0.008,
        w_floor  : float = 0.1,
        w_ceil   : float = 5.0,
        layer    : str   = "hippocampal",
    ):
        """
        Co-retrieval reinforcement — the temporal Hebbian signal.

        Nodes retrieved TOGETHER for the same query get their pairwise
        edges strengthened, regardless of embedding similarity. This is
        what makes the graph capture "fires together → wires together"
        over time, instead of recapitulating cosine similarity.

        Update rule per co-retrieval:  w += m_t·eta − lam·w  (clamped).
        New pairs start at 1.0 (same as bootstrap edges).

        Called once per chat turn with the retrieved top-k node ids.
        """
        pairs = [{"src": a, "dst": b}
                 for a in node_ids for b in node_ids if a != b]
        if not pairs:
            return

        query = """
        UNWIND $pairs AS p
        MATCH (a:Concept {node_id: p.src})
        MATCH (b:Concept {node_id: p.dst})
        MERGE (a)-[r:ASSOCIATED_WITH {layer: $layer}]->(b)
        ON CREATE SET
            r.hebb_weight         = 1.0,
            r.eligibility         = 0.0,
            r.causal_score        = 0.0,
            r.co_activation_count = 1,
            r.last_updated        = $now
        ON MATCH SET
            r.hebb_weight = CASE
                WHEN r.hebb_weight + $hebb - $lam * r.hebb_weight > $w_ceil
                    THEN $w_ceil
                WHEN r.hebb_weight + $hebb - $lam * r.hebb_weight < $w_floor
                    THEN $w_floor
                ELSE r.hebb_weight + $hebb - $lam * r.hebb_weight
            END,
            r.co_activation_count = r.co_activation_count + 1,
            r.last_updated        = $now
        """
        self._run(query, {
            "pairs"  : pairs,
            "hebb"   : m_t * eta,
            "lam"    : lam,
            "w_floor": w_floor,
            "w_ceil" : w_ceil,
            "layer"  : layer,
            "now"    : time.time(),
        })

    # ── Subgraph retrieval for GNN ────────────────────────────────────

    def get_local_subgraph(
        self,
        seed_node_ids : list[str],
        hops          : int   = 2,
        max_nodes     : int   = 100,
        layer         : str   = "hippocampal",
        min_weight    : float = 0.0,
    ) -> Subgraph:
        """
        Pull a local neighborhood for hippocampal GNN processing.

        Returns nodes and edges within `hops` of the seed nodes,
        filtered by layer and minimum Hebbian weight.
        Max nodes capped at max_nodes to keep GNN subgraphs manageable.

        Why hops=2 default: the hippocampal GNN operates on local context.
        Going deeper captures too much of the graph and kills the speed
        advantage of local Hebbian updates.
        """
        query = """
        MATCH (seed:Concept)
        WHERE seed.node_id IN $seed_ids
        CALL apoc.path.subgraphAll(seed, {
            relationshipFilter: 'ASSOCIATED_WITH>',
            maxLevel: $hops
        })
        YIELD nodes, relationships
        WITH nodes, relationships
        UNWIND nodes AS n
        WITH collect(DISTINCT n) AS all_nodes, relationships
        UNWIND relationships AS r
        WITH all_nodes,
             collect(DISTINCT r) AS all_rels
        RETURN all_nodes, all_rels
        LIMIT 1
        """
        # Fallback if APOC not available: manual 2-hop query
        query_no_apoc = """
        MATCH (seed:Concept)
        WHERE seed.node_id IN $seed_ids
        OPTIONAL MATCH (seed)-[r1:ASSOCIATED_WITH {layer: $layer}]->(n1:Concept)
        WHERE r1.hebb_weight >= $min_weight
        OPTIONAL MATCH (n1)-[r2:ASSOCIATED_WITH {layer: $layer}]->(n2:Concept)
        WHERE r2.hebb_weight >= $min_weight
        WITH collect(DISTINCT seed) + collect(DISTINCT n1) + collect(DISTINCT n2)
             AS all_nodes,
             collect(DISTINCT r1) + collect(DISTINCT r2) AS all_rels
        RETURN all_nodes, all_rels
        """

        try:
            rows = self._run(query, {
                "seed_ids": seed_node_ids,
                "hops"    : hops,
            })
        except Exception:
            # APOC not installed — use manual query
            rows = self._run(query_no_apoc, {
                "seed_ids": seed_node_ids,
                "hops"    : hops,
                "layer"   : layer,
                "min_weight": min_weight,
            })

        if not rows:
            return Subgraph(nodes=[], edges=[])

        row      = rows[0]
        raw_nodes = row.get("all_nodes", [])
        raw_rels  = row.get("all_rels", [])

        # Filter edges FIRST (layer + weight floor), then keep only
        # seed nodes and endpoints of surviving edges. The APOC path
        # returns edges of every layer, so filtering nodes by
        # activation_count before this step could let wrong-layer
        # neighbors evict real ones from the max_nodes cap.
        kept_edges = []
        for r in raw_rels:
            props = dict(r)
            if (props.get("layer") == layer
                    and props.get("hebb_weight", 0) >= min_weight):
                kept_edges.append((props, r))

        seed_set = set(seed_node_ids)
        endpoint_ids = {str(r.start_node["node_id"]) for _, r in kept_edges} \
                     | {str(r.end_node["node_id"]) for _, r in kept_edges}

        raw_nodes = [
            n for n in raw_nodes
            if n["node_id"] in seed_set or n["node_id"] in endpoint_ids
        ]
        # Cap at max_nodes — sort by activation_count descending (most active first)
        raw_nodes = sorted(
            raw_nodes,
            key=lambda n: n.get("activation_count", 0),
            reverse=True
        )[:max_nodes]

        node_ids_in_graph = {n["node_id"] for n in raw_nodes}
        kept_edges = [(p, r) for p, r in kept_edges
                      if str(r.start_node["node_id"]) in node_ids_in_graph
                      and str(r.end_node["node_id"]) in node_ids_in_graph]

        nodes = [self._row_to_concept(n) for n in raw_nodes]
        edges = []
        for props, r in kept_edges:
            edges.append(HebbianEdge(
                src_id              = str(r.start_node["node_id"]),
                dst_id              = str(r.end_node["node_id"]),
                hebb_weight         = float(props.get("hebb_weight", 1.0)),
                eligibility         = float(props.get("eligibility", 0.0)),
                causal_score        = float(props.get("causal_score", 0.0)),
                co_activation_count = int(props.get("co_activation_count", 0)),
                last_updated        = float(props.get("last_updated", 0.0)),
                layer               = props.get("layer", layer),
            ))

        return Subgraph(nodes=nodes, edges=edges)

    def get_global_graph(
        self,
        layer      : str   = "hippocampal",
        min_weight : float = 0.5,
        limit      : int   = 5000,
    ) -> Subgraph:
        """
        Pull a large subgraph for cortical GNN consolidation.
        Filtered to edges above min_weight to keep it manageable.
        Called on schedule (hourly/nightly), not at ingest time.
        """
        query = """
        MATCH (src:Concept)-[r:ASSOCIATED_WITH {layer: $layer}]->(dst:Concept)
        WHERE r.hebb_weight >= $min_weight
        WITH src, r, dst
        ORDER BY r.hebb_weight DESC
        LIMIT $limit
        WITH collect(DISTINCT src) + collect(DISTINCT dst) AS all_nodes,
             collect(r) AS all_rels
        RETURN all_nodes, all_rels
        """
        rows = self._run(query, {
            "layer"     : layer,
            "min_weight": min_weight,
            "limit"     : limit,
        })
        if not rows:
            return Subgraph(nodes=[], edges=[])

        row = rows[0]
        node_map = {}
        for n in row.get("all_nodes", []):
            nid = n["node_id"]
            if nid not in node_map:
                node_map[nid] = self._row_to_concept(n)

        edges = []
        for r in row.get("all_rels", []):
            props = dict(r)
            edges.append(HebbianEdge(
                src_id              = str(r.start_node["node_id"]),
                dst_id              = str(r.end_node["node_id"]),
                hebb_weight         = float(props.get("hebb_weight", 1.0)),
                eligibility         = float(props.get("eligibility", 0.0)),
                causal_score        = float(props.get("causal_score", 0.0)),
                co_activation_count = int(props.get("co_activation_count", 0)),
                last_updated        = float(props.get("last_updated", 0.0)),
                layer               = props.get("layer", layer),
            ))

        return Subgraph(nodes=list(node_map.values()), edges=edges)

    def list_concepts(self, limit: int = 200) -> list[ConceptNode]:
        """
        Most recently updated concepts, regardless of edges.
        Used by the dashboard to show isolated nodes — concepts that
        exist but have no edges above the current min_weight filter
        (e.g. freshly ingested nodes before their first co-retrieval).
        """
        query = """
        MATCH (c:Concept)
        RETURN c
        ORDER BY c.updated_at DESC
        LIMIT $limit
        """
        return [self._row_to_concept(r["c"])
                for r in self._run(query, {"limit": limit})]

    # ── ARM-E stats updates ───────────────────────────────────────────

    def update_concept_arme_stats(
        self,
        node_id   : str,
        m_t       : float,
        dominance : float,
        r_t       : float,
    ):
        """
        Update lifetime running stats (Option B) and circular buffer (Option C).

        Running mean uses Welford's online algorithm — no need to store history.
        Circular buffer capped at RECENT_EVENTS_MAX — oldest entry dropped.

        Called after every ARM-E step for a node.
        """
        # Read current stats
        result = self._run(
            "MATCH (c:Concept {node_id: $nid}) "
            "RETURN c.mean_m_t AS mm, c.mean_dominance AS md, "
            "       c.mean_r_t AS mr, c.activation_count AS cnt, "
            "       c.recent_events AS re",
            {"nid": node_id}
        )
        if not result:
            return

        row = result[0]
        cnt = max(int(row["cnt"]), 1)

        # Welford online mean update: mean_new = mean + (x - mean) / n
        new_mm = float(row["mm"]) + (m_t       - float(row["mm"])) / cnt
        new_md = float(row["md"]) + (dominance - float(row["md"])) / cnt
        new_mr = float(row["mr"]) + (r_t       - float(row["mr"])) / cnt

        # Circular buffer update
        events = json.loads(row["re"] or "[]")
        events.append({
            "m_t"      : round(m_t, 4),
            "dominance": round(dominance, 4),
            "r_t"      : round(r_t, 4),
            "ts"       : round(time.time(), 1),
        })
        if len(events) > self.RECENT_EVENTS_MAX:
            events = events[-self.RECENT_EVENTS_MAX:]

        self._run(
            "MATCH (c:Concept {node_id: $nid}) "
            "SET c.mean_m_t       = $mm, "
            "    c.mean_dominance = $md, "
            "    c.mean_r_t       = $mr, "
            "    c.recent_events  = $re",
            {
                "nid": node_id,
                "mm" : new_mm,
                "md" : new_md,
                "mr" : new_mr,
                "re" : json.dumps(events),
            }
        )

    # ── Session management ────────────────────────────────────────────

    def start_session(self, trigger: str = "time_window") -> str:
        """Create a new Session node. Returns session_id."""
        session_id = str(uuid.uuid4())
        self._run(
            "CREATE (:Session {"
            "  session_id: $sid, trigger: $trigger, "
            "  started_at: $now, ended_at: null, total_activations: 0"
            "})",
            {"sid": session_id, "trigger": trigger, "now": time.time()}
        )
        return session_id

    def end_session(self, session_id: str):
        """Close a session by writing ended_at."""
        self._run(
            "MATCH (s:Session {session_id: $sid}) SET s.ended_at = $now",
            {"sid": session_id, "now": time.time()}
        )

    def increment_session_activations(self, session_id: str):
        self._run(
            "MATCH (s:Session {session_id: $sid}) "
            "SET s.total_activations = s.total_activations + 1",
            {"sid": session_id}
        )

    def upsert_session_stats(
        self,
        session_id : str,
        node_id    : str,
        m_t        : float,
        dominance  : float,
        r_t        : float,
        delta_w    : float,
    ):
        """
        Upsert SessionStats for one (session, concept) pair.
        Uses Welford online mean — safe to call repeatedly.

        delta_w: mean Hebbian weight change this step — key for w_d regression.
        """
        query = """
        MERGE (ss:SessionStats {session_id: $sid, node_id: $nid})
        ON CREATE SET
            ss.stat_id          = $stat_id,
            ss.mean_m_t         = $m_t,
            ss.mean_dominance   = $dominance,
            ss.mean_r_t         = $r_t,
            ss.activation_count = 1,
            ss.delta_w_mean     = $delta_w
        ON MATCH SET
            ss.mean_m_t       = ss.mean_m_t
                                + ($m_t - ss.mean_m_t) / (ss.activation_count + 1),
            ss.mean_dominance = ss.mean_dominance
                                + ($dominance - ss.mean_dominance) / (ss.activation_count + 1),
            ss.mean_r_t       = ss.mean_r_t
                                + ($r_t - ss.mean_r_t) / (ss.activation_count + 1),
            ss.delta_w_mean   = ss.delta_w_mean
                                + ($delta_w - ss.delta_w_mean) / (ss.activation_count + 1),
            ss.activation_count = ss.activation_count + 1
        WITH ss
        MATCH (s:Session {session_id: $sid}), (c:Concept {node_id: $nid})
        MERGE (s)-[:AGGREGATES]->(ss)
        MERGE (ss)-[:TRACKED_IN]->(c)
        """
        self._run(query, {
            "sid"      : session_id,
            "nid"      : node_id,
            "stat_id"  : str(uuid.uuid4()),
            "m_t"      : m_t,
            "dominance": dominance,
            "r_t"      : r_t,
            "delta_w"  : delta_w,
        })

    # ── LLM retrieval ─────────────────────────────────────────────────

    def retrieve_for_llm(
        self,
        query_embedding : list[float],
        seed_node_ids   : list[str],
        top_k           : int   = 5,
        alpha           : float = 0.6,   # vector similarity weight
        beta            : float = 0.3,   # Hebbian association weight
        gamma           : float = 0.1,   # causal score weight
        w_ceil          : float = 5.0,   # hebb_weight scale — assoc normalized by this
        use_spreading   : bool  = True,  # multi-hop activation diffusion
        delta_spread    : float = 0.3,   # weight of normalized spread activation
        spread_extra    : Optional[int] = None,  # max spread-only candidates (None → top_k, 0 → off)
    ) -> list[dict]:
        """
        Combined retrieval for LLM context assembly.

        Score = α·cos_sim + β·assoc_norm + γ·causal_score

        assoc_norm = clamp(direct/w_ceil + δ·spread, 0, 1)

          direct  — average Hebbian edge weight from session seeds (1-hop)
          spread  — multi-hop activation diffusion through the Hebbian
                    graph from the same seeds, normalized to [0,1]
                    (spreading_activation). Lets indirectly-associated
                    concepts surface even when their cosine similarity to
                    the query is low.

        cos_sim and each assoc component live in [0, 1]; without the
        w_ceil normalization a strongly-wired node's association term
        could contribute β·5 ≈ 1.5 — dwarfing α·cos_sim ≤ 0.6.

        Returns list of dicts with 'label', 'text_summary', 'text_raw',
        'concept_type', 'source_type', 'source_uri', 'score'.

        The LLM receives text_summary by default.
        text_raw is included so the caller can optionally expand
        high-priority concepts (two-stage fetch, DeepSeek MLA style).
        """
        # Step 1: vector candidates
        vector_hits = self.vector_search(query_embedding, top_k=top_k * 3)
        cos_scores    = {n.node_id: s for n, s in vector_hits}

        # Step 1b: spreading activation, computed BEFORE the association query so
        # nodes the graph reaches strongly but vector search missed can join the
        # pool — the "surface even when cosine similarity is low" promised above.
        # (Previously spread was filtered to vector candidates, so the graph could
        # only rerank.) Extras are scored by the same formula as everything else.
        raw_spread: dict[str, float] = {}
        if use_spreading and seed_node_ids:
            raw_spread = self.spreading_activation(seed_node_ids, min_weight=0.3)
            n_extra = top_k if spread_extra is None else spread_extra
            ranked = sorted(raw_spread.items(), key=lambda kv: kv[1], reverse=True)
            for nid, act in ranked:
                if n_extra <= 0:
                    break
                if nid in cos_scores or act < SPREAD_CANDIDATE_MIN:
                    continue
                node = self.get_concept(nid)
                cos = _neo4j_cosine(query_embedding, node.embedding) if node else None
                if cos is None:
                    continue
                vector_hits.append((node, cos))
                cos_scores[nid] = cos
                n_extra -= 1
        candidate_ids = [n.node_id for n, _ in vector_hits]

        # Step 2a: Hebbian association scores from seed nodes (direct)
        if seed_node_ids and candidate_ids:
            assoc_query = """
            UNWIND $cand_ids AS cid
            MATCH (seed:Concept)-[r:ASSOCIATED_WITH]->(cand:Concept {node_id: cid})
            WHERE seed.node_id IN $seed_ids
            RETURN cand.node_id AS nid,
                   avg(r.hebb_weight) AS assoc,
                   avg(r.causal_score) AS causal
            """
            assoc_rows = self._run(assoc_query, {
                "cand_ids": candidate_ids,
                "seed_ids": seed_node_ids,
            })
            assoc_scores  = {r["nid"]: float(r["assoc"] or 0)  for r in assoc_rows}
            causal_scores = {r["nid"]: float(r["causal"] or 0) for r in assoc_rows}
        else:
            assoc_scores  = {}
            causal_scores = {}

        # Step 2b: multi-hop spreading activation (indirect association)
        spread_scores = {nid: act for nid, act in raw_spread.items()
                         if nid in cos_scores}

        # Step 3: combine scores
        scored = []
        for node, cos in vector_hits:
            nid       = node.node_id
            assoc_raw = assoc_scores.get(nid, 0.0)
            spread    = spread_scores.get(nid, 0.0)
            assoc_norm = max(0.0, min(1.5,
                assoc_raw / w_ceil + delta_spread * spread))
            score  = (alpha  * cos
                    + beta   * assoc_norm
                    + gamma  * causal_scores.get(nid, 0.0))
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "node_id"     : n.node_id,
                "label"       : n.label,
                "text_summary": n.text_summary,   # primary LLM input
                "text_raw"    : n.text_raw,        # available for two-stage expand
                "concept_type": n.concept_type,
                "source_type" : n.source_type,
                "source_uri"  : n.source_uri,
                "score"       : round(score, 4),
                "cos_sim"     : round(cos_scores.get(n.node_id, 0), 4),
                "assoc"       : round(assoc_scores.get(n.node_id, 0), 4),
                "spread"      : round(spread_scores.get(n.node_id, 0), 4),
                "causal"      : round(causal_scores.get(n.node_id, 0), 4),
            }
            for n, score in scored[:top_k]
        ]

    # ── Practice: fading concepts ─────────────────────────────────────

    def get_fading_concepts(
        self,
        limit           : int   = 5,
        min_activations : int   = 1,
        w_ceil          : float = 5.0,
        candidate_mult  : int   = 4,
    ) -> list[dict]:
        """
        Concepts whose memory is fading — the practice queue.

        A concept is a good practice target when it WAS learned (has
        activations) but its Hebbian wiring has decayed and it hasn't
        been touched recently:

            fading_score = (1 − strength) · staleness · log(1 + activations)

            strength  = clamp(avg_hippocampal_weight / w_ceil, 0, 1)
                        isolated nodes count as strength 0
            staleness = type-conditioned, 1 − exp(−Δt / τ_type)
                        (vocabulary goes stale in days; grammar rules and
                        cultural frameworks take weeks — see type_profiles)
            log term keeps rarely-seen noise below once-important concepts

        Answering a practice exercise correctly calls increment_activation
        + reinforce_co_retrieved on the target, which refreshes updated_at
        and strengthens edges — so a successful retrieval literally moves
        the concept to the back of the forgetting curve.

        Returns dicts: {concept (ConceptNode), avg_w, degree, fading_score}.
        """
        query = """
        MATCH (c:Concept)
        WHERE c.activation_count >= $min_act
        OPTIONAL MATCH (c)-[r:ASSOCIATED_WITH {layer: 'hippocampal'}]-()
        WITH c, avg(r.hebb_weight) AS avg_w, count(r) AS degree
        RETURN c, avg_w, degree
        ORDER BY c.updated_at ASC
        LIMIT $limit
        """
        rows = self._run(query, {
            "min_act": min_activations,
            "limit"  : limit * candidate_mult,
        })

        now = time.time()
        out = []
        for row in rows:
            node  = self._row_to_concept(row["c"])
            avg_w = float(row["avg_w"]) if row["avg_w"] is not None else 0.0
            degree = int(row["degree"])

            strength  = max(0.0, min(1.0, avg_w / w_ceil))
            staleness = _type_staleness(now - node.updated_at, node.concept_type)
            fading    = (1.0 - strength) * staleness \
                        * math.log1p(node.activation_count)

            out.append({
                "concept"      : node,
                "avg_w"        : avg_w,
                "degree"       : degree,
                "strength"     : round(strength, 4),
                "staleness"    : round(staleness, 4),
                "fading_score" : round(fading, 4),
            })

        out.sort(key=lambda d: d["fading_score"], reverse=True)
        return out[:limit]

    def spreading_activation(
        self,
        seed_node_ids : list[str],
        layer         : str   = "hippocampal",
        depth         : int   = 2,
        decay         : float = 0.45,
        min_weight    : float = 0.3,
        max_paths     : int   = 2000,
    ) -> dict[str, float]:
        """
        Multi-hop activation diffusion through the Hebbian graph
        (spreading-activation retrieval; cf. HeLa-Mem, ACL 2026).

        Activation flows from seed nodes along ASSOCIATED_WITH edges:
        each path contributes decay^hops · Π(edge weights), capped per
        path at w_ceil so single strong edges can't dominate. Results are
        normalized to [0, 1] by the maximum activation.

        This captures indirect association — a grammar concept two hops
        from the current topic lights up even when its cosine similarity
        is low — which pure vector search cannot see.

        Returns {node_id: activation in [0, 1]} (seeds excluded).
        """
        depth = max(1, min(int(depth), 3))   # interpolated — keep it an int
        query = f"""
        MATCH (seed:Concept)-[rels:ASSOCIATED_WITH*1..{depth}]->(n:Concept)
        WHERE seed.node_id IN $seeds
          AND all(r IN rels WHERE r.layer = $layer AND r.hebb_weight >= $min_w)
          AND n.node_id <> seed.node_id
        RETURN n.node_id AS nid,
               size(rels) AS hops,
               reduce(w = 1.0, x IN rels | w * x.hebb_weight) AS path_w
        LIMIT $max_paths
        """
        try:
            rows = self._run(query, {
                "seeds"     : seed_node_ids,
                "layer"     : layer,
                "min_w"     : min_weight,
                "max_paths" : max_paths,
            })
        except Exception as e:
            print(f"[graph] spreading_activation failed ({e}) — skipping")
            return {}

        activations: dict[str, float] = {}
        for row in rows:
            nid = row["nid"]
            if nid in seed_node_ids:
                continue
            contribution = (decay ** int(row["hops"])) \
                           * min(float(row["path_w"] or 0.0), self.W_CEIL)
            activations[nid] = activations.get(nid, 0.0) + contribution

        if not activations:
            return {}
        peak = max(activations.values())
        if peak <= 0:
            return {}
        return {nid: act / peak for nid, act in activations.items()}

    # ── Phase 4: w_d regression data ─────────────────────────────────

    def get_dominance_regression_data(
        self, min_activations: int = 5
    ) -> list[dict]:
        """
        Pull data for tuning w_dom in Phase 4.

        Returns rows of (mean_dominance, delta_w_mean, activation_count)
        from SessionStats — enough to fit a simple linear regression.

        Filter min_activations: exclude nodes seen too rarely to be meaningful.
        """
        query = """
        MATCH (s:Session)-[:AGGREGATES]->(ss:SessionStats)-[:TRACKED_IN]->(c:Concept)
        WHERE ss.activation_count >= $min_act
        RETURN ss.mean_dominance   AS dominance,
               ss.delta_w_mean     AS delta_w,
               ss.activation_count AS count,
               ss.mean_m_t         AS m_t,
               c.concept_type      AS ctype,
               s.session_id        AS session_id
        ORDER BY s.started_at DESC
        """
        return self._run(query, {"min_act": min_activations})

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_concept(row: Any) -> ConceptNode:
        """Convert a Neo4j node record to a ConceptNode dataclass."""
        # Handle both dict-like node objects and plain dicts
        def get(key, default=None):
            try:
                return row[key]
            except (KeyError, TypeError):
                return default

        recent_raw = get("recent_events", "[]") or "[]"
        try:
            recent = json.loads(recent_raw)
        except (json.JSONDecodeError, TypeError):
            recent = []

        emb = get("embedding", [])
        if emb is None:
            emb = []

        return ConceptNode(
            node_id          = str(get("node_id", "")),
            label            = str(get("label", "")),
            text_raw         = str(get("text_raw", "")),
            text_summary     = str(get("text_summary", "")),
            concept_type     = str(get("concept_type", "fact")),
            source_type      = str(get("source_type", "document")),
            source_uri       = get("source_uri"),
            created_at       = float(get("created_at", 0)),
            updated_at       = float(get("updated_at", 0)),
            activation_count = int(get("activation_count", 0)),
            embedding        = list(emb),
            mean_m_t         = float(get("mean_m_t", 1.0)),
            mean_dominance   = float(get("mean_dominance", 0.5)),
            mean_r_t         = float(get("mean_r_t", 0.5)),
            recent_events    = recent,
        )

    @staticmethod
    def make_node_id() -> str:
        return str(uuid.uuid4())


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=== GraphStore demo ===")
    print("Connecting to Neo4j... (make sure Neo4j Desktop is running)")

    store = GraphStore(
        uri      = "bolt://localhost:7687",
        user     = "neo4j",
        password = "password",    # change to your Neo4j password
    )

    print("\n1. Setting up schema...")
    store.setup_schema()

    print("\n2. Creating concept nodes...")
    dim = 1024

    def fake_emb(seed: int) -> list[float]:
        random.seed(seed)
        v = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x*x for x in v))
        return [x / norm for x in v]

    concepts = [
        ConceptNode(
            node_id="c001", label="Hebbian plasticity",
            text_raw="Neurons that fire together wire together. Hebbian learning strengthens synaptic connections between co-activated neurons.",
            text_summary="Hebbian plasticity: co-activated neurons strengthen their synaptic connections.",
            concept_type="fact", source_type="document",
            source_uri="https://doi.org/10.1037/h0042519",
            created_at=0, updated_at=0, activation_count=0,
            embedding=fake_emb(1),
        ),
        ConceptNode(
            node_id="c002", label="Oja's rule",
            text_raw="Oja's rule extends Hebbian learning with a decay term to prevent weight explosion: Δw = η(hᵢhⱼ - λwᵢⱼhᵢ²).",
            text_summary="Oja's rule adds weight decay to Hebbian learning, preventing unbounded growth.",
            concept_type="fact", source_type="web",
            source_uri="https://neurophysics.ucsd.edu/courses/physics_171/Oja_1982.pdf",
            created_at=0, updated_at=0, activation_count=0,
            embedding=fake_emb(2),
        ),
        ConceptNode(
            node_id="c003", label="ARM-E modulation",
            text_raw="ARM-E computes a scalar m_t from retrieval coherence, usage, novelty, emotion, and dominance to gate Hebbian updates.",
            text_summary="ARM-E scales Hebbian learning rate based on emotional and contextual signals.",
            concept_type="procedure", source_type="synthesized",
            source_uri=None,
            created_at=0, updated_at=0, activation_count=0,
            embedding=fake_emb(3),
        ),
    ]

    for c in concepts:
        store.upsert_concept(c)
        print(f"   upserted: {c.label}")

    print("\n3. Creating Hebbian edges...")
    edges = [
        HebbianEdge("c001","c002", hebb_weight=1.35, eligibility=0.42,
                    causal_score=0.0, co_activation_count=5,
                    last_updated=time.time(), layer="hippocampal"),
        HebbianEdge("c002","c003", hebb_weight=0.98, eligibility=0.21,
                    causal_score=0.0, co_activation_count=2,
                    last_updated=time.time(), layer="hippocampal"),
        HebbianEdge("c001","c003", hebb_weight=1.12, eligibility=0.33,
                    causal_score=0.71, co_activation_count=4,
                    last_updated=time.time(), layer="hippocampal"),
    ]
    for e in edges:
        store.upsert_edge(e)
        print(f"   edge: {e.src_id} → {e.dst_id}  w={e.hebb_weight:.3f}")

    print("\n4. Testing vector search...")
    query_emb = fake_emb(1)    # should match c001 most closely
    results = store.vector_search(query_emb, top_k=3)
    for node, score in results:
        print(f"   {node.label:<30} score={score:.4f}  type={node.concept_type}")

    print("\n5. Testing subgraph retrieval (for GNN)...")
    sg = store.get_local_subgraph(["c001"], hops=2, layer="hippocampal")
    print(f"   nodes: {[n.label for n in sg.nodes]}")
    print(f"   edges: {len(sg.edges)}")

    print("\n6. Testing session + ARM-E stats...")
    sid = store.start_session(trigger="time_window")
    print(f"   session: {sid}")

    for nid, m_t, dom, r_t, dw in [
        ("c001", 1.5, 0.72, 0.81, 0.04),
        ("c001", 1.3, 0.68, 0.75, 0.03),
        ("c002", 0.8, 0.35, 0.42, -0.01),
    ]:
        store.update_concept_arme_stats(nid, m_t, dom, r_t)
        store.upsert_session_stats(sid, nid, m_t, dom, r_t, dw)
        store.increment_session_activations(sid)
        store.increment_activation(nid)

    store.end_session(sid)
    print("   session closed")

    print("\n7. Testing LLM retrieval...")
    llm_results = store.retrieve_for_llm(
        query_embedding=query_emb,
        seed_node_ids=["c001"],
        top_k=3,
    )
    for r in llm_results:
        print(f"   [{r['concept_type']}] {r['label']:<30} "
              f"score={r['score']:.3f}  "
              f"(cos={r['cos_sim']:.3f} assoc={r['assoc']:.3f} causal={r['causal']:.3f})")
        print(f"     summary: {r['text_summary'][:70]}...")
        print(f"     source:  {r['source_type']} — {r['source_uri']}")

    print("\n8. Dominance regression data (for Phase 4 w_d tuning)...")
    rows = store.get_dominance_regression_data(min_activations=1)
    print(f"   {len(rows)} rows available for regression")
    for row in rows:
        print(f"   dominance={row['dominance']:.3f}  "
              f"delta_w={row['delta_w']:.4f}  "
              f"m_t={row['m_t']:.3f}  "
              f"type={row['ctype']}")

    store.close()
    print("\nDone.")
"""
check_identity.py — offline checks for concept identity (no Neo4j needed).

1. label_key / resolve cases, mirrored from mobileRAG's ConceptIdentityTest.kt
   (keep the two lists in sync — they define cross-platform parity).
2. The real IngestPipeline over an in-memory fake store: re-ingesting a concept
   reuses the node, and re-asserted edges never weaken.

3. Optional --explain: compiles the new Cypher against your Neo4j with EXPLAIN
   (plans only, nothing executes or writes). Uses NEO4J_URI/USER/PASSWORD.

Run from "Hebbian Memory/" with an interpreter that has torch + torch_geometric:
  /opt/anaconda3/envs/RAG/bin/python tools/check_identity.py [--explain]
"""
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

import concept_identity as ci
from graph_store import ConceptNode, HebbianEdge, Subgraph

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


def node(nid, label, ctype="vocabulary", activations=1, created=0.0, emb=(1.0, 0.0)):
    return ConceptNode(nid, label, label, label, ctype, "user", None, created, created,
                       activations, list(emb))


# ── 1. label_key + resolve (mirror of ConceptIdentityTest.kt) ─────────
LABEL_CASES = [
    ("  食べる ", "食べる"),
    ("「食べる」", "食べる"),
    ("Taberu", "taberu"),
    ("ＡＢＣ　ｄｅｆ", "abc def"),
    ("te  form\tverbs", "te form verbs"),
    ("て-form.", "て-form"),
    ("(passive voice)", "passive voice"),
    ("C#", "c#"),
    ("C++", "c++"),
    ("〜ている", "〜ている"),
    ("!!!", ""),
]
for raw, expected in LABEL_CASES:
    check(f"label_key({raw!r}) == {expected!r}", ci.label_key(raw) == expected)

a, b = node("a", "食べる", activations=2), node("b", " 食べる ", activations=5)
check("same label + type → most activated", ci.resolve("食べる", "vocabulary", True, [a, b], []) is b)
old, new = node("old", "食べる", activations=3, created=100), node("new", "食べる", activations=3, created=200)
check("activation tie → earliest", ci.resolve("食べる", "vocabulary", True, [new, old], []) is old)
check("same label, other type → none",
      ci.resolve("て-form", "example", True, [node("g", "て-form", "grammar")], []) is None)
n = node("n", "to eat")
check("embedding 0.98 same type → reuse", ci.resolve("食べる", "vocabulary", True, [], [(n, 0.98)]) is n)
check("embedding 0.97 → none", ci.resolve("食べる", "vocabulary", True, [], [(n, 0.97)]) is None)
check("embedding other type → none", ci.resolve("食べる", "grammar", True, [], [(n, 0.99)]) is None)
g, v = node("g", "x", "grammar"), node("v", "y")
check("skips wrong type to right one", ci.resolve("z", "vocabulary", True, [], [(g, 0.99), (v, 0.98)]) is v)
check("unvalidated → none", ci.resolve("食べる", "vocabulary", False, [n], [(n, 1.0)]) is None)
e = node("e", "User: hi", "event")
check("event → none", ci.resolve("User: hi", "event", True, [e], [(e, 1.0)]) is None)
check("punctuation-only label → no label match",
      ci.resolve("!!!", "vocabulary", True, [node("p", "!!!")], []) is None)


# ── 2. IngestPipeline reuse over a fake store ─────────────────────────
class FakeStore:
    """Just enough GraphStore for IngestPipeline, with the new edge semantics."""

    def __init__(self):
        self.nodes, self.edges = {}, {}

    def vector_search(self, query_embedding, top_k=10, concept_type=None):
        import math
        def score(nd):
            dot = sum(x * y for x, y in zip(query_embedding, nd.embedding))
            na = math.sqrt(sum(x * x for x in query_embedding))
            nb = math.sqrt(sum(y * y for y in nd.embedding))
            return (1 + dot / (na * nb)) / 2
        hits = sorted(((nd, score(nd)) for nd in self.nodes.values()), key=lambda t: -t[1])
        return hits[:top_k]

    def find_by_label_key(self, key):
        return [nd for nd in self.nodes.values() if key and ci.label_key(nd.label) == key]

    def upsert_concept(self, c):
        self.nodes[c.node_id] = c
        return c.node_id

    def increment_activation(self, nid):
        self.nodes[nid].activation_count += 1
        self.nodes[nid].updated_at = time.time()

    def upsert_edge(self, edge):
        key = (edge.src_id, edge.dst_id, edge.layer)
        cur = self.edges.get(key)
        if cur is None:
            self.edges[key] = edge
        else:  # never weaken; keep eligibility/causal
            cur.hebb_weight = max(cur.hebb_weight, edge.hebb_weight)
            cur.co_activation_count += 1

    def get_local_subgraph(self, seed_node_ids, hops=2, max_nodes=100, layer="hippocampal", min_weight=0.0):
        seen, frontier, kept = set(seed_node_ids), list(seed_node_ids), []
        for _ in range(hops):
            nxt = []
            for nid in frontier:
                for (s, d, l), ed in self.edges.items():
                    if s == nid and l == layer and ed.hebb_weight >= min_weight:
                        kept.append(ed)
                        if d not in seen:
                            seen.add(d)
                            nxt.append(d)
            frontier = nxt
        ids = set(seed_node_ids) | {e.src_id for e in kept} | {e.dst_id for e in kept}
        return Subgraph([self.nodes[i] for i in ids if i in self.nodes], kept)

    def update_hebb_weights(self, updates):
        for s, d, w, el, layer in updates:
            ed = self.edges[(s, d, layer)]
            ed.hebb_weight, ed.eligibility = w, el

    def update_concept_arme_stats(self, *a, **k): pass
    def upsert_session_stats(self, *a, **k): pass
    def increment_session_activations(self, *a, **k): pass
    def start_session(self, trigger="x"): return "s1"
    def end_session(self, sid): pass


try:
    from ingest_pipeline import IngestPipeline
except Exception as exc:  # torch / torch_geometric missing
    print(f"SKIP ingest checks ({exc}) — run with the RAG env")
else:
    stub_emotion = lambda text: types.SimpleNamespace(valence=0.0, arousal=0.5, dominance=0.5)
    vocab = {"食べる": [1.0, 0.1, 0.0, 0.0], "飲む": [0.1, 1.0, 0.0, 0.0], "見る": [0.0, 0.2, 1.0, 0.0]}
    embed = lambda text: next((v for k, v in vocab.items() if k in text), [0.0, 0.0, 0.0, 1.0])
    store = FakeStore()
    pipe = IngestPipeline(store, embed, summary_fn=None, feature_dim=4, emotion_classifier=stub_emotion)
    pipe.start_session()
    r1 = pipe.ingest("食べる: to eat", "食べる", concept_type="vocabulary", user_text="hi")
    pipe.ingest("飲む: to drink", "飲む", concept_type="vocabulary", user_text="hi")
    pipe.ingest("見る: to see", "見る", concept_type="vocabulary", user_text="hi")
    r2 = pipe.ingest("「食べる」: eat (verb)", "「食べる」", concept_type="vocabulary", user_text="hi")
    check("re-ingest reuses the node", r2.reused and r2.node_id == r1.node_id)
    check("no duplicate node", sum(1 for nd in store.nodes.values() if ci.label_key(nd.label) == "食べる") == 1)
    check("activation count is 2", store.nodes[r1.node_id].activation_count == 2)
    check("reused node keeps its first description", store.nodes[r1.node_id].text_raw == "食べる: to eat")
    strong = next(iter(store.edges.values()))
    strong.hebb_weight = 4.8
    store.upsert_edge(HebbianEdge(strong.src_id, strong.dst_id, 1.5, 0.0, 0.0, 1, time.time(), strong.layer))
    check("re-asserted edge never weakens", strong.hebb_weight == 4.8)
    r3 = pipe.ingest("食べる", "食べる", concept_type="grammar", user_text="hi")
    check("same label, other type → new node", not r3.reused and r3.node_id != r1.node_id)

# ── 3. Optional: compile the new Cypher (EXPLAIN — nothing runs) ─────
if "--explain" in sys.argv:
    from neo4j import GraphDatabase
    queries = {
        "upsert_edge never weakens": """
            MATCH (src:Concept {node_id: $src_id}) MATCH (dst:Concept {node_id: $dst_id})
            MERGE (src)-[r:ASSOCIATED_WITH {layer: $layer}]->(dst)
            ON CREATE SET r.hebb_weight = $w
            ON MATCH SET r.hebb_weight = CASE WHEN $w > r.hebb_weight THEN $w ELSE r.hebb_weight END,
                         r.co_activation_count = r.co_activation_count + 1""",
        "find_by_label_key": "MATCH (c:Concept {label_key: $key}) RETURN c",
        "update_concept_embeddings":
            "UNWIND $u AS u MATCH (c:Concept {node_id: u.node_id}) SET c.embedding = u.embedding",
        "update_causal_scores(layer)": """
            UNWIND $u AS u MATCH (s:Concept {node_id: u.src_id})-[r:ASSOCIATED_WITH {layer: $layer}]
            ->(d:Concept {node_id: u.dst_id}) SET r.causal_score = u.score""",
    }
    params = {"src_id": "x", "dst_id": "y", "layer": "hippocampal", "w": 1.0, "key": "k", "u": []}
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password"))
    with GraphDatabase.driver(os.environ.get("NEO4J_URI", "neo4j://localhost:7687"), auth=auth) as drv:
        drv.verify_connectivity()  # fail once on bad credentials instead of per query
        with drv.session() as session:
            for name, q in queries.items():
                try:
                    session.run("EXPLAIN " + q, params).consume()
                    check(f"cypher compiles: {name}", True)
                except Exception as exc:
                    print(f"   {exc}")
                    check(f"cypher compiles: {name}", False)

print(f"\n{'ALL PASS' if not failures else str(len(failures)) + ' FAILED'}")
sys.exit(1 if failures else 0)

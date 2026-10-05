"""
merge_duplicates.py — opt-in merge of concepts duplicated before concept identity.

New duplicates are prevented at ingest (core/concept_identity.py). This folds the
old ones together, with the same rules as mobileRAG's ConceptMerge.kt:

  groups   same label_key + same concept_type (events excluded); survivor = most
           activations, ties → earliest created
  node     activations summed; created_at min, updated_at max; ARM-E means weighted
           by activations; recent_events merged by ts (latest 20); label/text/
           embedding stay the survivor's
  edges    re-pointed to the survivor; self loops dropped; collisions: weight,
           eligibility, causal max; co_activation_count summed; last_updated max
  sessions SessionStats re-pointed; on a (session, concept) collision the row with
           more activations survives
  practice practice_state/scheduler.json last_practiced ids folded into survivors

DRY RUN by default. Back up first, then apply (one Neo4j transaction):
  neo4j-admin database dump neo4j --to-path=<backup dir>     (database stopped)
  /opt/anaconda3/envs/RAG/bin/python tools/merge_duplicates.py --apply

  --selftest  runs the pure merge rules on fixtures (mirror of ConceptMergeTest.kt)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

from concept_identity import label_key, EVENT_TYPE

RECENT_EVENTS_MAX = 20


# ── Pure rules (mirror of ConceptMerge.kt) ────────────────────────────

def plan(concepts):
    """concepts: dicts with node_id, label, concept_type, activation_count, created_at.
    Returns [(keep, [drops])] and the remap {dropped_id: keep_id}."""
    groups = {}
    for c in concepts:
        if c["concept_type"] == EVENT_TYPE:
            continue
        key = label_key(c["label"] or "")
        if key:
            groups.setdefault((key, c["concept_type"]), []).append(c)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda c: (c["activation_count"], -c["created_at"]))
        out.append((keep, [c for c in members if c is not keep]))
    out.sort(key=lambda g: g[0]["node_id"])
    remap = {d["node_id"]: keep["node_id"] for keep, drops in out for d in drops}
    return out, remap


def merged_node(keep, drops):
    allc = [keep] + drops
    w = [max(c["activation_count"], 1) for c in allc]
    total = sum(w)
    def weighted(field):
        return sum(c[field] * wi for c, wi in zip(allc, w)) / total
    events = [e for c in allc for e in c["recent_events"]]
    events.sort(key=lambda e: e.get("ts", 0.0) if isinstance(e, dict) else 0.0)
    return {
        "node_id": keep["node_id"],
        "activation_count": sum(c["activation_count"] for c in allc),
        "created_at": min(c["created_at"] for c in allc),
        "updated_at": max(c["updated_at"] for c in allc),
        "mean_m_t": weighted("mean_m_t"),
        "mean_dominance": weighted("mean_dominance"),
        "mean_r_t": weighted("mean_r_t"),
        "recent_events": events[-RECENT_EVENTS_MAX:],
    }


def merge_edges(edges, remap):
    out = {}
    for e in edges:
        s = remap.get(e["src"], e["src"])
        d = remap.get(e["dst"], e["dst"])
        if s == d:
            continue
        key = (s, d, e["layer"])
        cur = out.get(key)
        if cur is None:
            out[key] = dict(e, src=s, dst=d)
        else:
            cur["w"] = max(cur["w"], e["w"])
            cur["elig"] = max(cur["elig"], e["elig"])
            cur["causal"] = max(cur["causal"], e["causal"])
            cur["co"] = cur["co"] + e["co"]
            cur["ts"] = max(cur["ts"], e["ts"])
    return list(out.values())


def merge_session_stats(rows, remap):
    best = {}
    for r in rows:
        r = dict(r, node_id=remap.get(r["node_id"], r["node_id"]))
        key = (r["session_id"], r["node_id"])
        if key not in best or r["activation_count"] > best[key]["activation_count"]:
            best[key] = r
    return list(best.values())


# ── Neo4j I/O ─────────────────────────────────────────────────────────

def _events(raw):
    try:
        v = json.loads(raw or "[]")
        return v if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


def load_concepts(tx):
    rows = tx.run(
        "MATCH (c:Concept) RETURN c.node_id AS node_id, c.label AS label, "
        "c.concept_type AS concept_type, coalesce(c.activation_count, 0) AS activation_count, "
        "coalesce(c.created_at, 0.0) AS created_at, coalesce(c.updated_at, 0.0) AS updated_at, "
        "coalesce(c.mean_m_t, 1.0) AS mean_m_t, coalesce(c.mean_dominance, 0.5) AS mean_dominance, "
        "coalesce(c.mean_r_t, 0.5) AS mean_r_t, c.recent_events AS recent_events"
    )
    out = []
    for r in rows:
        d = r.data()
        d["recent_events"] = _events(d["recent_events"])
        out.append(d)
    return out


def apply_merge(tx, groups, remap):
    ids = sorted({g[0]["node_id"] for g in groups} | set(remap))

    edges = [r.data() for r in tx.run(
        "MATCH (a:Concept)-[r:ASSOCIATED_WITH]->(b:Concept) "
        "WHERE a.node_id IN $ids OR b.node_id IN $ids "
        "RETURN a.node_id AS src, b.node_id AS dst, r.layer AS layer, "
        "coalesce(r.hebb_weight, 1.0) AS w, coalesce(r.eligibility, 0.0) AS elig, "
        "coalesce(r.causal_score, 0.0) AS causal, coalesce(r.co_activation_count, 0) AS co, "
        "coalesce(r.last_updated, 0.0) AS ts", ids=ids)]
    tx.run("MATCH (a:Concept)-[r:ASSOCIATED_WITH]->(b:Concept) "
           "WHERE a.node_id IN $ids OR b.node_id IN $ids DELETE r", ids=ids)
    tx.run("UNWIND $edges AS e "
           "MATCH (a:Concept {node_id: e.src}), (b:Concept {node_id: e.dst}) "
           "CREATE (a)-[:ASSOCIATED_WITH {layer: e.layer, hebb_weight: e.w, eligibility: e.elig, "
           "causal_score: e.causal, co_activation_count: e.co, last_updated: e.ts}]->(b)",
           edges=merge_edges(edges, remap))

    stats = [r.data() for r in tx.run(
        "MATCH (ss:SessionStats) WHERE ss.node_id IN $ids "
        "RETURN ss.stat_id AS stat_id, ss.session_id AS session_id, ss.node_id AS node_id, "
        "coalesce(ss.activation_count, 0) AS activation_count", ids=ids)]
    survivors = merge_session_stats(stats, remap)
    keep_ids = {r["stat_id"] for r in survivors}
    losers = [r["stat_id"] for r in stats if r["stat_id"] not in keep_ids]
    tx.run("MATCH (ss:SessionStats) WHERE ss.stat_id IN $ids DETACH DELETE ss", ids=losers)
    original = {r["stat_id"]: r["node_id"] for r in stats}
    moved = [{"stat_id": r["stat_id"], "node_id": r["node_id"]} for r in survivors
             if original[r["stat_id"]] != r["node_id"]]
    tx.run("UNWIND $moved AS m MATCH (ss:SessionStats {stat_id: m.stat_id}) "
           "OPTIONAL MATCH (ss)-[t:TRACKED_IN]->() DELETE t "
           "WITH DISTINCT ss, m SET ss.node_id = m.node_id "
           "WITH ss, m MATCH (c:Concept {node_id: m.node_id}) MERGE (ss)-[:TRACKED_IN]->(c)", moved=moved)

    tx.run("UNWIND $nodes AS n MATCH (c:Concept {node_id: n.node_id}) "
           "SET c.activation_count = n.activation_count, c.created_at = n.created_at, "
           "c.updated_at = n.updated_at, c.mean_m_t = n.mean_m_t, c.mean_dominance = n.mean_dominance, "
           "c.mean_r_t = n.mean_r_t, c.recent_events = n.recent_events",
           nodes=[dict(m, recent_events=json.dumps(m["recent_events"]))
                  for m in (merged_node(k, d) for k, d in groups)])
    tx.run("MATCH (c:Concept) WHERE c.node_id IN $ids DETACH DELETE c", ids=list(remap))


def remap_practice(practice_dir, remap):
    path = os.path.join(practice_dir, "scheduler.json")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        data = json.load(f)
    lp = data.get("last_practiced", {})
    n = 0
    for src, dst in remap.items():
        if src in lp:
            lp[dst] = max(lp.get(dst, 0.0), lp.pop(src))
            n += 1
    if n:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    return n


# ── Self-test (mirror of ConceptMergeTest.kt) ─────────────────────────

def selftest():
    fails = []
    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)
    def c(nid, label, ctype="vocabulary", act=1, created=0.0, updated=0.0, mt=1.0, ts=()):
        return {"node_id": nid, "label": label, "concept_type": ctype, "activation_count": act,
                "created_at": created, "updated_at": updated, "mean_m_t": mt, "mean_dominance": 0.5,
                "mean_r_t": 0.5, "recent_events": [{"ts": t, "m_t": 1.0} for t in ts]}

    groups, remap = plan([c("a", "食べる", act=1), c("b", "「食べる」", act=4), c("c", "食べる", "grammar"),
                          c("d", "飲む"), c("e", "User: hi", "event"), c("f", "User: hi", "event")])
    check("one group, most activated survives", len(groups) == 1 and groups[0][0]["node_id"] == "b")
    check("remap a→b", remap == {"a": "b"})
    groups, _ = plan([c("new", "x", act=2, created=20), c("old", "x", act=2, created=10)])
    check("tie keeps earliest", groups[0][0]["node_id"] == "old")

    m = merged_node(c("k", "x", act=3, created=50, updated=60, mt=2.0, ts=(5.0, 7.0)),
                    [c("d", "x", act=1, created=10, updated=90, mt=1.0, ts=(6.0,))])
    check("activations summed", m["activation_count"] == 4)
    check("created min / updated max", m["created_at"] == 10 and m["updated_at"] == 90)
    check("means weighted", abs(m["mean_m_t"] - 7 / 4) < 1e-12)
    check("events by ts", [e["ts"] for e in m["recent_events"]] == [5.0, 6.0, 7.0])
    m = merged_node(c("k", "x", act=2, ts=range(1, 16)), [c("d", "x", ts=range(16, 31))])
    check("latest 20 events", [e["ts"] for e in m["recent_events"]] == list(range(11, 31)))

    E = lambda s, d, w, elig=0.0, causal=0.0, co=1, ts=0: {"src": s, "dst": d, "layer": "hippocampal",
                                                             "w": w, "elig": elig, "causal": causal, "co": co, "ts": ts}
    out = {(e["src"], e["dst"]): e for e in merge_edges(
        [E("k", "n", 1.0, 0.5, 0.1, 2, 5), E("d", "n", 3.0, 0.2, 0.4, 3, 9), E("d", "k", 2.0), E("m", "d", 1.5)],
        {"d": "k"})}
    check("edge keys", set(out) == {("k", "n"), ("m", "k")})
    kn = out[("k", "n")]
    check("edge combine", (kn["w"], kn["elig"], kn["causal"], kn["co"], kn["ts"]) == (3.0, 0.5, 0.4, 5, 9))

    rows = [{"stat_id": "s1", "session_id": "sess", "node_id": "k", "activation_count": 2},
            {"stat_id": "s2", "session_id": "sess", "node_id": "d", "activation_count": 5},
            {"stat_id": "s3", "session_id": "other", "node_id": "d", "activation_count": 1}]
    out = merge_session_stats(rows, {"d": "k"})
    check("session stats", sorted(r["stat_id"] for r in out) == ["s2", "s3"]
          and all(r["node_id"] == "k" for r in out))
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="merge (default is a dry run)")
    ap.add_argument("--selftest", action="store_true", help="check the merge rules and exit")
    ap.add_argument("--practice-dir", default="./practice_state")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    from neo4j import GraphDatabase
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password"))
    with GraphDatabase.driver(os.environ.get("NEO4J_URI", "neo4j://localhost:7687"), auth=auth) as drv:
        drv.verify_connectivity()
        with drv.session() as session:
            concepts = session.execute_read(load_concepts)
            groups, remap = plan(concepts)
            print(f"{len(concepts)} concepts; {len(groups)} duplicate groups; "
                  f"{len(remap)} nodes would be merged away")
            for keep, drops in groups[:20]:
                print(f"  {keep['label']!r} [{keep['concept_type']}] ← {len(drops)} duplicate(s)")
            if len(groups) > 20:
                print(f"  … and {len(groups) - 20} more groups")
            if not args.apply or not groups:
                print("\nDry run — nothing changed. Back up Neo4j, then re-run with --apply.")
                return
            session.execute_write(apply_merge, groups, remap)
    n = remap_practice(args.practice_dir, remap)
    print(f"Merged {len(remap)} duplicates into {len(groups)} concepts; "
          f"practice history remapped for {n} ids.")


if __name__ == "__main__":
    main()

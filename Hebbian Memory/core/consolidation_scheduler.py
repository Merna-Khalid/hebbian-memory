"""
consolidation_scheduler.py
Cortical GNN consolidation scheduler for the Hebbian RAG system.

Runs cortical consolidation:
    - Every N ingests (default: 20)
    - At session end (always)
    - Manually on demand

Handles:
    - Pulling global subgraph from Neo4j
    - Running CorticalGNN.consolidate()
    - Writing updated embeddings back to Neo4j
    - Persisting cortical GNN weights + z_prev to disk
    - Loading persisted state on startup

Cortical GNN is slow and runs on large graphs — it should NOT
block the ingest loop. This scheduler either:
    - Runs synchronously at session end (acceptable latency)
    - Runs in a background thread during the session

Usage:
    scheduler = ConsolidationScheduler(
        store        = graph_store,
        cortical_gnn = CorticalGNN(...),
        model_dir    = "./cortical_state",
        every_n      = 20,
    )
    scheduler.load()          # load persisted state on startup

    # in ingest loop:
    scheduler.on_ingest()     # triggers if count % every_n == 0

    # at session end:
    scheduler.on_session_end()
"""

import os
import time
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from threading import Thread

from graph_store import GraphStore, HebbianEdge
from cortical_gnn import CorticalGNN, consolidate


class ConsolidationScheduler:
    """
    Manages when and how the cortical GNN consolidates.

    State persisted to disk (model_dir/):
        cortical_model/     — CorticalGNN encoder weights
        head.pt             — regression head (if VAD model used)
        z_prev.pt           — previous cortical embeddings [N, D]
        node_id_order.pt    — list of node_ids matching z_prev rows
        meta.pt             — ingest count, last consolidation time
    """

    def __init__(
        self,
        store           : GraphStore,
        cortical_gnn    : CorticalGNN,
        model_dir       : str   = "./cortical_state",
        every_n         : int   = 20,      # consolidate every N ingests
        feature_dim     : int   = 1024,    # must match embedding model
        min_nodes       : int   = 5,       # skip if graph too small
        min_edges       : int   = 3,       # skip if too few edges
        consolidate_epochs: int = 30,      # epochs per consolidation run
        consolidate_lr  : float = 5e-4,
        min_hebb_weight : float = 0.5,     # pull edges above this weight
        background      : bool  = False,   # run in background thread
        verbose         : bool  = True,
    ):
        self.store              = store
        self.cortical_gnn       = cortical_gnn
        self.model_dir          = model_dir
        self.every_n            = every_n
        self.feature_dim        = feature_dim
        self.min_nodes          = min_nodes
        self.min_edges          = min_edges
        self.consolidate_epochs = consolidate_epochs
        self.consolidate_lr     = consolidate_lr
        self.min_hebb_weight    = min_hebb_weight
        self.background         = background
        self.verbose            = verbose

        # Runtime state
        self._ingest_count      : int           = 0
        self._last_consolidation: float         = 0.0
        self._z_prev            : Optional[Tensor] = None
        self._node_id_order     : list          = []   # node_ids matching z_prev rows
        self._running           : bool          = False

        os.makedirs(model_dir, exist_ok=True)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self):
        """Save cortical GNN weights and z_prev to disk."""
        self.cortical_gnn.gat1.state_dict()  # ensure params exist

        torch.save(
            self.cortical_gnn.state_dict(),
            os.path.join(self.model_dir, "cortical_gnn.pt"),
        )
        if self._z_prev is not None:
            torch.save(self._z_prev,
                       os.path.join(self.model_dir, "z_prev.pt"))
            torch.save(self._node_id_order,
                       os.path.join(self.model_dir, "node_id_order.pt"))
        torch.save(
            {"ingest_count": self._ingest_count,
             "last_consolidation": self._last_consolidation},
            os.path.join(self.model_dir, "meta.pt"),
        )
        if self.verbose:
            print(f"[consolidation] state saved → {self.model_dir}")

    def load(self):
        """Load persisted cortical state on startup. Safe to call if nothing saved yet."""
        model_path = os.path.join(self.model_dir, "cortical_gnn.pt")
        if os.path.exists(model_path):
            self.cortical_gnn.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            print(f"[consolidation] loaded cortical GNN from {self.model_dir}")

        z_path = os.path.join(self.model_dir, "z_prev.pt")
        if os.path.exists(z_path):
            self._z_prev        = torch.load(z_path, map_location="cpu")
            self._node_id_order = torch.load(
                os.path.join(self.model_dir, "node_id_order.pt"),
                map_location="cpu",
            )
            print(f"  z_prev loaded: {self._z_prev.shape}  "
                  f"({len(self._node_id_order)} nodes)")

        meta_path = os.path.join(self.model_dir, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            self._ingest_count       = meta.get("ingest_count", 0)
            self._last_consolidation = meta.get("last_consolidation", 0.0)
            last_str = (
                time.strftime("%H:%M:%S", time.localtime(self._last_consolidation))
                if self._last_consolidation else "never"
            )
            print(f"  ingest count: {self._ingest_count}  "
                  f"last consolidation: {last_str}")

    # ── Triggers ──────────────────────────────────────────────────────

    def on_ingest(self):
        """
        Call after every successful ingest.
        Triggers consolidation if count % every_n == 0.
        """
        self._ingest_count += 1
        if self._ingest_count % self.every_n == 0:
            if self.verbose:
                print(f"\n[consolidation] trigger: {self._ingest_count} ingests")
            self._run(reason="scheduled")

    def on_session_end(self):
        """
        Call at session end. Always consolidates if graph is large enough.
        Saves state to disk.
        """
        if self.verbose:
            print("\n[consolidation] session end — running final consolidation...")
        self._run(reason="session_end")
        self.save()

    def consolidate_now(self):
        """Manual trigger — call from anywhere."""
        self._run(reason="manual")
        self.save()

    # ── Core consolidation ────────────────────────────────────────────

    def _run(self, reason: str = "scheduled"):
        """Run one cortical consolidation cycle."""
        if self._running:
            print("[consolidation] already running — skipping")
            return

        if self.background:
            t = Thread(target=self._consolidate_sync, args=(reason,), daemon=True)
            t.start()
        else:
            self._consolidate_sync(reason)

    def _consolidate_sync(self, reason: str):
        """Synchronous consolidation — pulls graph, runs GNN, writes back."""
        self._running = True
        t0 = time.time()

        try:
            # ── 1. Pull global subgraph ───────────────────────────────
            if self.verbose:
                print(f"[consolidation] pulling global graph "
                      f"(min_weight={self.min_hebb_weight})...")

            subgraph = self.store.get_global_graph(
                layer      = "hippocampal",
                min_weight = self.min_hebb_weight,
                limit      = 5000,
            )

            n_nodes = len(subgraph.nodes)
            n_edges = len(subgraph.edges)

            if n_nodes < self.min_nodes or n_edges < self.min_edges:
                if self.verbose:
                    print(f"[consolidation] skipped — graph too small "
                          f"({n_nodes} nodes, {n_edges} edges)")
                return

            if self.verbose:
                print(f"[consolidation] graph: {n_nodes} nodes, {n_edges} edges")

            # ── 2. Build PyG Data ─────────────────────────────────────
            node_id_list = [n.node_id for n in subgraph.nodes]
            id_to_idx    = {nid: i for i, nid in enumerate(node_id_list)}

            H = torch.tensor(
                [n.embedding for n in subgraph.nodes], dtype=torch.float
            )
            H = F.normalize(H, dim=1)

            valid_edges = [
                e for e in subgraph.edges
                if e.src_id in id_to_idx and e.dst_id in id_to_idx
            ]

            if not valid_edges:
                if self.verbose:
                    print("[consolidation] no valid edges — skipping")
                return

            src_idx    = [id_to_idx[e.src_id] for e in valid_edges]
            dst_idx    = [id_to_idx[e.dst_id] for e in valid_edges]
            weights    = [e.hebb_weight        for e in valid_edges]

            from torch_geometric.data import Data
            edge_index  = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            edge_weight = torch.tensor(weights, dtype=torch.float)

            data = Data(
                x          = H,
                edge_index = edge_index,
                edge_attr  = edge_weight.unsqueeze(1),
                num_nodes  = n_nodes,
            )

            # ── 3. Build z_prev aligned to current node order ─────────
            # z_prev may be from a previous run with a different node set.
            # Align by node_id: reuse embeddings for known nodes,
            # use current H for new nodes.
            if self._z_prev is not None and len(self._node_id_order) > 0:
                prev_id_to_z = {
                    nid: self._z_prev[i]
                    for i, nid in enumerate(self._node_id_order)
                }
                z_prev_aligned = []
                for nid in node_id_list:
                    if nid in prev_id_to_z:
                        z_prev_aligned.append(prev_id_to_z[nid])
                    else:
                        # New node — anchor to current hippocampal embedding
                        z_prev_aligned.append(H[id_to_idx[nid]])
                z_prev = torch.stack(z_prev_aligned)
            else:
                z_prev = None   # first consolidation — anchor to H

            # ── 4. Run consolidation ──────────────────────────────────
            if self.verbose:
                print(f"[consolidation] running {self.consolidate_epochs} epochs...")

            z_cort, history = consolidate(
                data         = data,
                cortical_gnn = self.cortical_gnn,
                z_prev       = z_prev,
                epochs       = self.consolidate_epochs,
                lr           = self.consolidate_lr,
                verbose      = self.verbose,
            )

            final_loss = history[-1]["loss_total"]
            elapsed    = time.time() - t0

            if self.verbose:
                print(f"[consolidation] done in {elapsed:.1f}s  "
                      f"loss={final_loss:.4f}  reason={reason}")

            # ── 5. Write updated embeddings back to Neo4j ─────────────
            # Update each Concept node's embedding with the cortical
            # stabilised version. This is what the vector search uses —
            # cortical embeddings are more semantically coherent than
            # the raw hippocampal ones. updated_at is NOT touched: it means
            # "last seen" to the practice queue, and consolidation is not an
            # exposure (setting it here made every consolidated node look
            # freshly practiced).
            if self.verbose:
                print(f"[consolidation] writing {n_nodes} embeddings to Neo4j...")

            self.store.update_concept_embeddings({
                node_id_list[i]: z_cort[i].tolist() for i in range(n_nodes)
            })

            # ── 6. Update causal scores on edges ─────────────────────
            # Compute cosine similarity between cortical embeddings of
            # each connected pair — use as a proxy for causal alignment.
            # Written to the hippocampal edges trained on here: 'cortical'
            # edges are never created, so causal used to stay 0.
            causal_updates = []
            for k, e in enumerate(valid_edges):
                i = id_to_idx[e.src_id]
                j = id_to_idx[e.dst_id]
                causal = F.cosine_similarity(
                    z_cort[i].unsqueeze(0),
                    z_cort[j].unsqueeze(0),
                ).item()
                causal_updates.append((e.src_id, e.dst_id, float(causal)))

            if causal_updates:
                self.store.update_causal_scores(causal_updates, layer="hippocampal")

            # ── 7. Store z_prev for next run ──────────────────────────
            self._z_prev        = z_cort.detach()
            self._node_id_order = node_id_list
            self._last_consolidation = time.time()

            if self.verbose:
                print(f"[consolidation] ✓ complete — "
                      f"{n_nodes} embeddings updated, "
                      f"{len(causal_updates)} causal scores updated")

        except Exception as e:
            print(f"[consolidation] ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            self._running = False

    # ── Stats ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "ingest_count"      : self._ingest_count,
            "last_consolidation": self._last_consolidation,
            "z_prev_shape"      : list(self._z_prev.shape)
                                   if self._z_prev is not None else None,
            "n_nodes_tracked"   : len(self._node_id_order),
            "every_n"           : self.every_n,
            "next_at"           : (
                self._ingest_count +
                (self.every_n - self._ingest_count % self.every_n)
            ),
        }
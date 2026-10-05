# HippocampalGNN rewritten with PyG Data + MessagePassing base class.
# Behaviorally identical to raw_hippocampal_gnn.py — same math, better infrastructure.

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import os

# ── What is PyG Data? ─────────────────────────────────────────────────
#
# Data is just a dict-like container with dot access.
# Standard fields:
#   data.x          : [N, D]  node features
#   data.edge_index : [2, E]  directed edges (src, dst)
#   data.edge_attr  : [E, F]  edge features (we use F=1: the Hebbian weight)
#   data.num_nodes  : int     needed when graph has isolated nodes
#
# You can add ANY custom field:
#   data.node_ids = [10, 11, 12, ...]   # global Neo4j IDs
#   data.m_t      = 1.4                 # ARM-E modulation for this batch
#
# PyG never breaks on unknown fields — it just carries them along.
# This is why it's better than our Subgraph dataclass: it composes
# with the entire PyG ecosystem (DataLoader, transforms, batching).


def make_data(
    node_ids   : list,
    H          : Tensor,
    edge_index : Tensor,
    edge_weight: Tensor,
    m_t        : float = 1.0,
) -> Data:
    """
    Build a PyG Data object from our familiar ingredients.
    edge_attr shape: [E, 1] — PyG convention for scalar edge features.
    """
    return Data(
        x          = H,
        edge_index = edge_index,
        edge_attr  = edge_weight.unsqueeze(1),  # [E] → [E, 1]
        node_ids   = node_ids,
        m_t        = m_t,
        num_nodes  = H.shape[0],
    )


# ── HebbianConv: one message-passing layer ────────────────────────────
#
# Subclassing MessagePassing is the PyG way.
# You define THREE methods:
#
#   forward()   — called by you; sets up the graph and calls propagate()
#   message()   — called by PyG for each edge; returns the message vector
#   update()    — called by PyG for each node after aggregation
#
# PyG handles: neighbor lookup, aggregation (sum/mean/max), batching.
# You handle: what the message IS, and what to do with the aggregation.
#
# flow="source_to_target" means edge (i→j): i is source, j is target.
# x_i = features of target node j (what receives)
# x_j = features of source node i (what sends)
# (yes, the naming is confusing — PyG calls the sender _j)

class HebbianConv(MessagePassing):
    def __init__(self, feature_dim: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.D = feature_dim
        # Buffer, not a plain attribute: follows .to(device)/.cuda()
        # without appearing in parameters() (it's fixed identity — no grad).
        self.register_buffer("W", torch.eye(feature_dim))
 
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        N   = x.shape[0]
        dst = edge_index[1]
 
        # Fix: expand in_deg from [N] to [E] before propagate().
        # PyG >= 2.4 fails to broadcast a 1D [N] tensor as a generic kwarg.
        # in_deg[dst] is identical math to what PyG's _i suffix was attempting.
        in_deg      = torch.zeros(N).scatter_add(0, dst, edge_weight).clamp(min=1)
        in_deg_edge = in_deg[dst]   # [E]
 
        agg = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            in_deg_edge=in_deg_edge,   # already [E], no broadcasting needed
        )
        return torch.tanh(agg @ self.W)
 
    def message(self, x_j: Tensor, edge_weight: Tensor, in_deg_edge: Tensor) -> Tensor:
        return (edge_weight / in_deg_edge).unsqueeze(1) * x_j
 
    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


# ── HippocampalGNN ────────────────────────────────────────────────────

class HippocampalGNN(torch.nn.Module):
    """
    PyG-native hippocampal GNN.

    Same math as before, now using:
    - PyG Data objects for graph representation
    - MessagePassing base class for propagation
    - torch.nn.Module for future .parameters() / optimizer compatibility

    The Hebbian update is NOT in the computation graph (no autograd).
    It runs in torch.no_grad() — this is intentional.
    Backprop is the cortical GNN's job.
    """

    def __init__(
        self,
        feature_dim : int   = 768,
        eta         : float = 0.05,
        lam         : float = 0.008,
        w_floor     : float = 0.1,
        w_ceil      : float = 5.0,
        num_layers  : int   = 1,
    ):
        super().__init__()
        self.eta       = eta
        self.lam       = lam
        self.w_floor   = w_floor
        self.w_ceil    = w_ceil
        self.num_layers = num_layers
        self.conv      = HebbianConv(feature_dim)

    def _hebbian_update(
        self,
        data  : Data,
        H     : Tensor,
        m_t   : float,
    ) -> Tensor:
        """
        Directed Oja's rule. Runs outside autograd.
        Returns new edge_weight [E] (not yet stored in data).

        H is the POST-propagation feature matrix — co-activation is
        computed on conv outputs, so message passing actually shapes
        the weight update (with raw input features the conv would be
        decorative at num_layers=1).
        """
        with torch.no_grad():
            src = data.edge_index[0]
            dst = data.edge_index[1]
            w   = data.edge_attr.squeeze(1)   # [E, 1] → [E]

            h_src = H[src]                              # [E, D]
            h_dst = H[dst]                              # [E, D]
            coact = (h_src * h_dst).sum(dim=1)          # [E]
            hebb  = m_t * self.eta * coact
            decay = self.lam * w * (h_src ** 2).sum(1)
            new_w = (w + hebb - decay).clamp(self.w_floor, self.w_ceil)

        return new_w

    def propagate(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        """
        Message passing only — returns propagated node features.
        Used by the ingest pipeline, which runs the weight update
        separately through ARME.modulated_hebbian_update (eligibility
        gating + anti-Hebbian decay).
        """
        for _ in range(self.num_layers):
            x = self.conv(x, edge_index, edge_weight)
        return x

    def forward(self, data: Data) -> Tuple[Data, dict]:
        """
        One hippocampal pass: message passing → Hebbian update.

        Accepts a PyG Data object, returns an updated one.
        ARM-E modulation is read from data.m_t (default 1.0).
        """
        m_t = getattr(data, 'm_t', 1.0)
        ei  = data.edge_index
        ew  = data.edge_attr.squeeze(1)   # [E]

        x = self.propagate(data.x, ei, ew)
        new_w = self._hebbian_update(data, x, m_t)

        coact_mean = (
            (x[ei[0]] * x[ei[1]]).sum(1).mean().item()
        )
        dw = (new_w - ew).abs()
        stats = {
            "max_dw"    : dw.max().item(),
            "mean_dw"   : dw.mean().item(),
            "mean_w"    : new_w.mean().item(),
            "m_t"       : m_t,
            "coact_mean": coact_mean,
        }

        # build updated Data — PyG Data is immutable-ish so we clone
        data = Data(
            x          = x,
            edge_index = ei,
            edge_attr  = new_w.unsqueeze(1),
            node_ids   = data.node_ids,
            m_t        = m_t,
            num_nodes  = data.num_nodes,
        )

        return data, stats


# ── Visualization (same as before, reads from Data fields) ────────────

def visualize(data: Data, step: int, save_path: Optional[str] = None):
    fig, (ax_g, ax_e) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [1.3, 1]}
    )
    fig.patch.set_facecolor("#F8F8F6")

    N        = data.num_nodes
    D        = data.x.shape[1]
    ei       = data.edge_index
    ew       = data.edge_attr.squeeze(1)
    node_ids = data.node_ids
    m_t      = getattr(data, 'm_t', 1.0)

    # graph
    ax_g.set_facecolor("#F8F8F6")
    ax_g.set_title(f"HippocampalGNN (PyG) — step {step}  m_t={m_t:.2f}",
                   fontsize=12, pad=10)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for k in range(ei.shape[1]):
        s, d, w = ei[0,k].item(), ei[1,k].item(), ew[k].item()
        G.add_edge(s, d, weight=w)

    pos    = nx.spring_layout(G, seed=7, weight=None)
    w_vals = ew.tolist()
    w_min, w_max = min(w_vals), max(w_vals)
    w_rng  = max(w_max - w_min, 1e-6)

    edges      = list(G.edges())
    widths     = [1.0 + 5.0*(G[s][d]['weight']-w_min)/w_rng for s,d in edges]
    alphas     = [0.3  + 0.65*(G[s][d]['weight']-w_min)/w_rng for s,d in edges]
    colors     = [plt.cm.Blues(0.35+0.6*(G[s][d]['weight']-w_min)/w_rng) for s,d in edges]

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths,
                           alpha=alphas, edge_color=colors, arrows=True,
                           arrowsize=14, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08", ax=ax_g)
    nx.draw_networkx_nodes(G, pos,
                           node_color=plt.cm.Set2([i/max(N-1,1) for i in range(N)]),
                           node_size=700, linewidths=1.2, edgecolors="#555", ax=ax_g)
    nx.draw_networkx_labels(G, pos,
                            labels={i: f"N{node_ids[i]}" for i in range(N)},
                            font_size=8, font_weight="bold", ax=ax_g)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels={(s,d): f"{G[s][d]['weight']:.2f}"
                                              for s,d in edges},
                                 font_size=7, font_color="#444", ax=ax_g)
    ax_g.axis("off")

    # embeddings
    ax_e.set_facecolor("#F8F8F6")
    ax_e.set_title("Node embeddings", fontsize=12, pad=10)
    H_np = data.x.detach().numpy()
    im   = ax_e.imshow(H_np, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_e.set_xticks(range(D))
    ax_e.set_xticklabels([f"dim{d}" for d in range(D)], fontsize=8)
    ax_e.set_yticks(range(N))
    ax_e.set_yticklabels([f"N{node_ids[i]}" for i in range(N)], fontsize=8)
    for r in range(N):
        for c in range(D):
            v = H_np[r, c]
            ax_e.text(c, r, f"{v:.2f}", ha="center", va="center",
                      fontsize=7.5, color="white" if abs(v) > 0.5 else "#333")
    plt.colorbar(im, ax=ax_e, fraction=0.03, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#F8F8F6")
        plt.close(fig)
        print(f"  → saved {save_path}")
    else:
        plt.show()


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    H_init = F.normalize(torch.tensor([
        [0.05, 0.05, 0.95, 0.05],
        [0.05, 0.10, 0.90, 0.05],
        [0.50, 0.50, 0.50, 0.50],
        [0.95, 0.05, 0.05, 0.05],
        [0.90, 0.10, 0.05, 0.05],
    ]), dim=1)

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 3, 4, 2, 4, 3],
    ], dtype=torch.long)

    edge_weight = torch.ones(edge_index.shape[1])

    # ── Key difference from before: wrap everything in Data ──────────
    data = make_data(
        node_ids    = [10, 11, 12, 13, 14],
        H           = H_init,
        edge_index  = edge_index,
        edge_weight = edge_weight,
        m_t         = 1.0,
    )

    gnn = HippocampalGNN(feature_dim=4, eta=0.05, lam=0.008)

    print(f"{'Step':<6} {'m_t':<8} {'mean_w':<10} {'max_dw':<10} {'coact'}")
    print("-" * 50)

    scenarios = [(5, 1.0, "neutral"), (5, 1.4, "curiosity"), (5, 0.6, "confusion")]
    step = 0

    for n_steps, m_t, label in scenarios:
        print(f"\n--- {label} (m_t={m_t}) ---")
        data.m_t = m_t                        # set ARM-E signal on the Data object

        for _ in range(n_steps):
            step += 1
            data, stats = gnn.forward(data)
            print(f"{step:<6} {stats['m_t']:<8.2f} {stats['mean_w']:<10.4f} "
                  f"{stats['max_dw']:<10.5f} {stats['coact_mean']:.4f}")

        visualize(data, step=step,
                  save_path=f"outputs/pyg_step_{step:03d}_{label}.png")

    # ── Show what's now inside data ──────────────────────────────────
    print("\n=== PyG Data object contents ===")
    print(data)
    print(f"\ndata.x.shape        : {data.x.shape}")
    print(f"data.edge_index.shape: {data.edge_index.shape}")
    print(f"data.edge_attr.shape : {data.edge_attr.shape}")
    print(f"data.node_ids        : {data.node_ids}")
    print(f"data.m_t             : {data.m_t}")
    print(f"data.num_nodes       : {data.num_nodes}")
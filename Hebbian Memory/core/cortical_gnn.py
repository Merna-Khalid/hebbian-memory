"""
cortical_gnn.py
CorticalGNN — the slow, scheduled, semantic consolidation layer.

Reads accumulated hippocampal edge weights as supervision.
Trains stable node embeddings via GATConv + three-part loss.
Runs periodically (not on every ingest).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import os


# ── What GATConv does (in plain terms) ────────────────────────────────
#
# Standard GNN:  h_i' = σ( W · mean_j( w_ij · h_j ) )
#                        fixed scalar weight per edge
#
# GATConv:       h_i' = σ( Σ_j  α_ij · W · h_j )
#                        α_ij = softmax( LeakyReLU( a^T [Wh_i ‖ Wh_j] ) )
#
# α_ij is computed from node CONTENT, not just co-occurrence frequency.
# Two nodes that co-occurred often (high w_ij hippocampally) but have
# semantically incoherent features will get low α_ij — the cortex
# down-weights that association during consolidation.
#
# Multi-head: run K attention heads independently, concatenate outputs.
# Each head can specialize — one for topic similarity, one for causal
# direction, one for temporal proximity.


# ── CorticalGNN ───────────────────────────────────────────────────────

class CorticalGNN(nn.Module):
    """
    Two-layer GATConv network for slow semantic consolidation.

    Architecture:
        Input [N, D_in]
            ↓  GATConv(heads=4, concat=True)   → [N, D_hidden * 4]
            ↓  ELU + Dropout
            ↓  GATConv(heads=1, concat=False)  → [N, D_out]
            ↓  Output embeddings

    Loss = L_reconstruct + λ1·L_causal + λ2·L_consistency

    L_reconstruct : can we predict which edges exist from the embeddings?
                    (graph autoencoder style — dot product of endpoint embeddings)
    L_causal      : penalize embeddings where strong hippocampal edges
                    point in semantically inconsistent directions
    L_consistency : keep new embeddings close to previous cortical state
                    (prevents catastrophic forgetting of old knowledge)
    """

    def __init__(
        self,
        in_dim      : int   = 4,      # match your embedding dim
        hidden_dim  : int   = 16,
        out_dim     : int   = 4,      # same as in_dim — keeps shape stable
        heads       : int   = 2,
        dropout     : float = 0.1,
        lambda_causal: float = 0.3,
        lambda_consist: float = 0.5,
    ):
        super().__init__()

        self.lambda_causal  = lambda_causal
        self.lambda_consist = lambda_consist

        # Layer 1: multi-head attention, concat outputs
        self.gat1 = GATConv(
            in_channels  = in_dim,
            out_channels = hidden_dim,
            heads        = heads,
            dropout      = dropout,
            concat       = True,       # output: [N, hidden_dim * heads]
        )

        # Layer 2: single-head, average (not concat) → final embedding
        self.gat2 = GATConv(
            in_channels  = hidden_dim * heads,
            out_channels = out_dim,
            heads        = 1,
            dropout      = dropout,
            concat       = False,      # output: [N, out_dim]
        )

        self.dropout = nn.Dropout(dropout)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass through both GAT layers.
        Returns stable cortical embeddings H_cort: [N, out_dim]
        """
        # Layer 1: attend + ELU + dropout
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        # Layer 2: final embedding
        h = self.gat2(h, edge_index)
        return h

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        Reconstruct edge existence from embeddings.
        Score(i,j) = dot(z_i, z_j) — positive = edge should exist.
        Used in L_reconstruct.
        """
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)   # [E]

    def loss(
        self,
        z           : Tensor,   # current cortical embeddings [N, D]
        z_prev      : Tensor,   # previous cortical embeddings [N, D]
        edge_index  : Tensor,   # graph edges [2, E]
        edge_weight : Tensor,   # hippocampal weights [E] — supervision signal
        neg_edge_index: Optional[Tensor] = None,  # negative samples for recon loss
    ) -> Tuple[Tensor, dict]:
        """
        Three-part loss.

        L_reconstruct:
            Positive edges (exist in graph): push dot(z_i, z_j) → 1
            Negative edges (don't exist):    push dot(z_i, z_j) → 0
            Weighted by hippocampal w_ij — strong associations matter more.

        L_causal:
            Penalize edges where the embedding similarity CONTRADICTS the
            hippocampal weight direction. If w_ij is high but z_i·z_j is
            low (or negative), something is semantically wrong — force alignment.

        L_consistency:
            Keep z close to z_prev (previous cortical state).
            Prevents each consolidation run from forgetting old knowledge.
            This is a soft version of Elastic Weight Consolidation (EWC).
        """

        # ── L_reconstruct ─────────────────────────────────────────────
        pos_score = self.decode(z, edge_index)              # [E]

        # weight positive samples by hippocampal strength
        # strong edges should be reconstructed more faithfully
        w_norm    = edge_weight / (edge_weight.max() + 1e-8)  # normalize to [0,1]
        pos_loss  = (w_norm * F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score), reduction='none'
        )).mean()

        # negative edges: random pairs not in the graph
        if neg_edge_index is None:
            neg_edge_index = _sample_negative_edges(
                edge_index, z.shape[0], num_neg=edge_index.shape[1]
            )
        neg_score = self.decode(z, neg_edge_index)
        neg_loss  = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )
        l_recon = pos_loss + neg_loss

        # ── L_causal ──────────────────────────────────────────────────
        # If hippocampal weight is high but embedding similarity is low,
        # we're encoding an association the cortex doesn't understand.
        # Penalize the gap.
        emb_sim  = F.cosine_similarity(z[edge_index[0]], z[edge_index[1]])  # [E]
        expected = w_norm * 2 - 1          # map [0,1] weights → [-1,1] similarity
        l_causal = F.mse_loss(emb_sim, expected.clamp(-1, 1))

        # ── L_consistency ─────────────────────────────────────────────
        # Soft anchor to previous cortical state.
        # MSE between new and old embeddings.
        l_consist = F.mse_loss(z, z_prev.detach())  # detach: no grad through prev

        # ── Combined loss ─────────────────────────────────────────────
        total = l_recon + self.lambda_causal * l_causal + self.lambda_consist * l_consist

        return total, {
            "loss_total"  : total.item(),
            "loss_recon"  : l_recon.item(),
            "loss_causal" : l_causal.item(),
            "loss_consist": l_consist.item(),
        }


def _sample_negative_edges(edge_index: Tensor, N: int, num_neg: int) -> Tensor:
    """
    Sample random node pairs that are NOT in edge_index.
    Used as negative examples for reconstruction loss.
    Simple rejection sampling — fine for small graphs.
    """
    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    negs = []
    attempts = 0
    while len(negs) < num_neg and attempts < num_neg * 20:
        i = torch.randint(0, N, (1,)).item()
        j = torch.randint(0, N, (1,)).item()
        if i != j and (i, j) not in existing:
            negs.append([i, j])
        attempts += 1

    if not negs:   # fallback: just use reversed existing edges
        negs = edge_index.flip(0).T.tolist()[:num_neg]

    return torch.tensor(negs, dtype=torch.long).T   # [2, num_neg]


# ── Consolidation runner ──────────────────────────────────────────────

def consolidate(
    data        : Data,
    cortical_gnn: CorticalGNN,
    z_prev      : Optional[Tensor] = None,
    epochs      : int   = 50,
    lr          : float = 1e-3,
    verbose     : bool  = True,
) -> Tuple[Tensor, list]:
    """
    Run one consolidation session.

    Takes a graph with accumulated hippocampal edge weights (data.edge_attr)
    and trains the cortical GNN to produce stable embeddings.

    Parameters
    ----------
    data         : PyG Data with x, edge_index, edge_attr (hippocampal weights)
    cortical_gnn : CorticalGNN instance
    z_prev       : previous cortical embeddings — if None, use data.x as anchor
    epochs       : gradient steps per consolidation run
    lr           : learning rate

    Returns
    -------
    z_final : [N, D] stable cortical embeddings
    history : list of loss dicts per epoch
    """
    optimizer = torch.optim.Adam(cortical_gnn.parameters(), lr=lr)
    edge_weight = data.edge_attr.squeeze(1)   # [E]

    # anchor: if no previous cortical state, use hippocampal embeddings
    if z_prev is None:
        z_prev = data.x.clone().detach()

    history = []
    cortical_gnn.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        z = cortical_gnn.encode(data.x, data.edge_index)
        loss, metrics = cortical_gnn.loss(z, z_prev, data.edge_index, edge_weight)

        loss.backward()
        optimizer.step()
        history.append(metrics)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>3} | "
                  f"total={metrics['loss_total']:.4f}  "
                  f"recon={metrics['loss_recon']:.4f}  "
                  f"causal={metrics['loss_causal']:.4f}  "
                  f"consist={metrics['loss_consist']:.4f}")

    cortical_gnn.eval()
    with torch.no_grad():
        z_final = cortical_gnn.encode(data.x, data.edge_index)

    return z_final, history


# ── Visualization ─────────────────────────────────────────────────────

def visualize_consolidation(
    data      : Data,
    z_hipp    : Tensor,   # hippocampal (volatile) embeddings
    z_cort    : Tensor,   # cortical (stable) embeddings after consolidation
    history   : list,
    save_path : Optional[str] = None,
):
    """
    Three-panel figure:
      Left:   hippocampal embeddings heatmap
      Center: cortical embeddings heatmap
      Right:  loss curves over consolidation epochs
    """
    fig = plt.figure(figsize=(15, 5), facecolor="#F8F8F6")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_h = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])
    ax_l = fig.add_subplot(gs[2])

    N        = data.num_nodes
    D        = z_hipp.shape[1]
    node_ids = data.node_ids

    def draw_heatmap(ax, Z, title):
        ax.set_facecolor("#F8F8F6")
        ax.set_title(title, fontsize=11, pad=8)
        Z_np = Z.detach().numpy()
        im   = ax.imshow(Z_np, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(D))
        ax.set_xticklabels([f"d{d}" for d in range(D)], fontsize=8)
        ax.set_yticks(range(N))
        ax.set_yticklabels([f"N{node_ids[i]}" for i in range(N)], fontsize=8)
        for r in range(N):
            for c in range(D):
                v = Z_np[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(v) > 0.5 else "#333")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    draw_heatmap(ax_h, z_hipp, "Hippocampal H\n(volatile, post-Hebbian)")
    draw_heatmap(ax_c, z_cort, "Cortical H\n(stable, post-consolidation)")

    # loss curves
    ax_l.set_facecolor("#F8F8F6")
    ax_l.set_title("Consolidation loss", fontsize=11, pad=8)
    epochs  = range(len(history))
    colors  = {"loss_total":"#333","loss_recon":"#378ADD",
                "loss_causal":"#D85A30","loss_consist":"#1D9E75"}
    labels  = {"loss_total":"total","loss_recon":"reconstruct",
                "loss_causal":"causal","loss_consist":"consistency"}

    for key, col in colors.items():
        vals = [h[key] for h in history]
        lw   = 2.5 if key == "loss_total" else 1.5
        ax_l.plot(epochs, vals, color=col, linewidth=lw,
                  label=labels[key], alpha=0.9)

    ax_l.set_xlabel("Epoch", fontsize=9)
    ax_l.set_ylabel("Loss", fontsize=9)
    ax_l.legend(fontsize=8, framealpha=0.7)
    ax_l.set_facecolor("#F8F8F6")
    ax_l.spines[['top','right']].set_visible(False)

    plt.suptitle("Cortical Consolidation — Hippocampal → Cortical",
                 fontsize=12, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#F8F8F6")
        plt.close(fig)
        print(f"  → saved {save_path}")
    else:
        plt.show()


# ── Demo: full hippocampal → cortical pipeline ────────────────────────

if __name__ == "__main__":
    from hippocampal_gnn_pyg import HippocampalGNN, make_data
    os.makedirs("outputs", exist_ok=True)
    torch.manual_seed(42)

    # ── Step 1: build graph and run hippocampal GNN ───────────────────
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

    data = make_data([10,11,12,13,14], H_init,
                     edge_index, torch.ones(8))

    hipp_gnn = HippocampalGNN(feature_dim=4, eta=0.05, lam=0.008)

    print("=== Phase 1: Hippocampal encoding (15 steps, mixed ARM-E) ===")
    for m_t, n in [(1.0,5),(1.4,5),(0.6,5)]:
        data.m_t = m_t
        for _ in range(n):
            data, _ = hipp_gnn.forward(data)

    z_hipp = data.x.clone()
    print(f"Hippocampal edge weights after 15 steps:")
    ew = data.edge_attr.squeeze(1)
    for k in range(edge_index.shape[1]):
        s,d = edge_index[0,k].item(), edge_index[1,k].item()
        print(f"  N{s}→N{d}: {ew[k].item():.4f}")

    # ── Step 2: consolidate into cortical GNN ─────────────────────────
    print("\n=== Phase 2: Cortical consolidation (50 epochs) ===")

    cort_gnn = CorticalGNN(
        in_dim        = 4,
        hidden_dim    = 8,
        out_dim       = 4,
        heads         = 2,
        lambda_causal = 0.3,
        lambda_consist= 0.5,
    )

    z_cort, history = consolidate(
        data        = data,
        cortical_gnn= cort_gnn,
        z_prev      = None,     # first consolidation — anchor to hippocampal state
        epochs      = 50,
        lr          = 1e-3,
    )

    # ── Step 3: compare hippocampal vs cortical embeddings ───────────
    print("\n=== Hippocampal vs Cortical embeddings ===")
    print(f"{'Node':<8} {'Hipp sim to N13':<20} {'Cort sim to N13'}")
    print("-" * 48)
    for i in range(5):
        hs = F.cosine_similarity(z_hipp[i].unsqueeze(0), z_hipp[3].unsqueeze(0)).item()
        cs = F.cosine_similarity(z_cort[i].unsqueeze(0), z_cort[3].unsqueeze(0)).item()
        print(f"N{data.node_ids[i]:<7} {hs:>+.3f}               {cs:>+.3f}")

    print("\nN13 is the physics node. Cortical should cluster N13/N14 together")
    print("and separate them from N10/N11 (history) more cleanly than hippocampal.")

    visualize_consolidation(
        data, z_hipp, z_cort, history,
        save_path="outputs/cortical_consolidation.png"
    )

    # ── Step 4: show what a second consolidation looks like ──────────
    # (simulates the nightly replay cycle)
    print("\n=== Phase 3: Second consolidation (nightly replay) ===")
    print("Using previous cortical state as consistency anchor...")

    # run 5 more hippocampal steps (new daytime learning)
    data.m_t = 1.2
    for _ in range(5):
        data, _ = hipp_gnn.forward(data)

    z_cort2, history2 = consolidate(
        data         = data,
        cortical_gnn = cort_gnn,
        z_prev       = z_cort,     # anchor to PREVIOUS cortical state
        epochs       = 50,
        lr           = 5e-4,       # smaller lr for refinement
        verbose      = False,
    )
    print(f"Final loss: {history2[-1]['loss_total']:.4f}  "
          f"(consistency term protects old knowledge)")

    print("\nDone. The two GNNs are now wired together.")
    print("Next: ARM-E computes m_t from real signals → feeds into hippocampal GNN")
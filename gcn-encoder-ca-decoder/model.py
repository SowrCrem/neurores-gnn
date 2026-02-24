# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNLayer(nn.Module):
    """
    Simple dense GCN layer for weighted graphs:
      H' = sigma( D^-1/2 (A+I) D^-1/2 H W )
    Works in inductive setting; no transductive embeddings.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # A: (B, N, N), H: (B, N, Fin)
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)  # (1,N,N)
        A_hat = A + I

        # degree and normalization
        deg = A_hat.sum(dim=-1)  # (B, N)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)  # avoid divide-by-0
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # (B,N,N)

        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt  # (B,N,N)
        H = F.dropout(H, p=self.dropout, training=self.training)
        return F.relu(A_norm @ self.lin(H))  # (B,N,Fout)

class LR2HRGenerator(nn.Module):
    """
    Baseline graph super-resolution:
      LR adjacency -> LR node embeddings (dense GCN)
      HR nodes are learned query embeddings (268 x d)
      Cross-attention: HR queries attend to LR node embeddings
      Decode HR adjacency via bilinear form, enforce symmetry, output upper-tri vector
    """
    def __init__(
        self,
        n_lr: int,
        n_hr: int,
        d_model: int,
        gcn_layers: int,
        attn_heads: int,
        dropout: float,
        in_node_feat_dim: int = 2
    ):
        super().__init__()
        self.n_lr = n_lr
        self.n_hr = n_hr
        self.d_model = d_model

        # Project node features -> model dimension
        self.node_in = nn.Linear(in_node_feat_dim, d_model)

        # LR encoder (dense GCN stack)
        gcn = []
        for _ in range(gcn_layers):
            gcn.append(DenseGCNLayer(d_model, d_model, dropout))
        self.gcn = nn.ModuleList(gcn)

        # HR learnable node "queries" (shared across all subjects => inductive)
        self.hr_queries = nn.Parameter(torch.randn(n_hr, d_model) * 0.02)

        # Cross-attention upsampling (HR queries attend to LR nodes)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_ln = nn.LayerNorm(d_model)

        # Bilinear decoder parameter
        self.W = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def forward(self, A_lr: torch.Tensor, X_lr: torch.Tensor) -> torch.Tensor:
        """
        A_lr: (B, n_lr, n_lr) weighted symmetric
        X_lr: (B, n_lr, node_feat_dim)
        returns: pred_hr_vec (B, n_hr*(n_hr-1)/2)
        """
        # Encode LR nodes
        H = self.node_in(X_lr)  # (B, n_lr, d)
        for layer in self.gcn:
            H = layer(A_lr, H)   # (B, n_lr, d)

        # Upsample to HR nodes via cross-attention
        B = H.shape[0]
        Q = self.hr_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_hr, d)
        HR, _ = self.attn(query=Q, key=H, value=H)          # (B, n_hr, d)
        HR = self.attn_ln(HR + Q)                           # residual + norm

        # Decode HR adjacency (bilinear), then symmetrize
        # A = (HR W) HR^T
        HW = HR @ self.W                   # (B, n_hr, d)
        A = HW @ HR.transpose(1, 2)        # (B, n_hr, n_hr)
        A = 0.5 * (A + A.transpose(1, 2))  # enforce symmetry

        # return upper triangle without diagonal
        idx = torch.triu_indices(self.n_hr, self.n_hr, offset=1, device=A.device)
        return A[:, idx[0], idx[1]]
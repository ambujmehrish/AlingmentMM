import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class GraphAwarePooling(nn.Module):
    """Attention-based pooling for graph construction.

    Reduces sequence length from L to N and dimension from D to dgraph
    using cross-attention with learnable query vectors.
    """

    def __init__(self, input_dim=768, target_length=32, graph_dim=256):
        super().__init__()
        self.target_length = target_length
        self.graph_dim = graph_dim

        # Learnable query vectors
        self.queries = nn.Parameter(torch.randn(target_length, input_dim))
        trunc_normal_(self.queries, std=0.02)

        # Key/Value projections
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

        # Dimension reduction to graph space
        self.dim_reduce = nn.Sequential(
            nn.Linear(input_dim, graph_dim),
            nn.LayerNorm(graph_dim),
        )

    def forward(self, features):
        """
        Args:
            features: [B, L, D] encoded features from any modality
        Returns:
            pooled: [B, N, dgraph] graph-optimized features
        """
        B, L, D = features.shape

        # Compute keys and values
        K = self.key_proj(features)   # [B, L, D]
        V = self.value_proj(features) # [B, L, D]

        # Expand queries for batch
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5)  # [B, N, L]
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to values
        pooled = torch.bmm(attn, V)  # [B, N, D]

        # Dimension reduction to graph space
        graph_features = self.dim_reduce(pooled)  # [B, N, dgraph]

        return graph_features

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationshipGraphBuilder(nn.Module):
    """Constructs relationship graphs from pooled features using cosine similarity."""

    def __init__(self):
        super().__init__()

    def forward(self, features):
        """
        Args:
            features: [B, N, dgraph] pooled features
        Returns:
            R: [B, N, N] relationship matrix (cosine similarity)
        """
        # L2 normalize features
        features_norm = F.normalize(features, p=2, dim=-1)  # [B, N, dgraph]

        # Compute pairwise cosine similarity
        R = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, N, N]

        return R

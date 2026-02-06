import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class CrossModalGraphFusion(nn.Module):
    """Fuses expanded graphs from two modalities using learned weights.

    Computes: G_fused = sum_{p,q} A_{pq} * (R^p_M1 ⊙ R^q_M2)
    where A_{pq} are learnable scalar weights.
    """

    def __init__(self, order=3):
        super().__init__()
        self.order = order

        # Learnable fusion weights A_pq of shape [P+1, P+1]
        self.fusion_weights = nn.Parameter(torch.zeros(order + 1, order + 1))
        # Initialize with small values; diagonal gets slightly higher
        nn.init.uniform_(self.fusion_weights, -0.1, 0.1)
        with torch.no_grad():
            for i in range(order + 1):
                self.fusion_weights[i, i] += 1.0 / (order + 1)

    def forward(self, G_M1, G_M2):
        """
        Args:
            G_M1: [B, P+1, N, N] expanded graph for modality 1
            G_M2: [B, P+1, N, N] expanded graph for modality 2
        Returns:
            G_fused: [B, N, N] fused graph
        """
        B, P_plus_1, N, _ = G_M1.shape

        # Compute all cross-order Hadamard products at once using einsum
        # G_M1[:, p] has shape [B, N, N], G_M2[:, q] has shape [B, N, N]
        # We want sum over p, q of A[p,q] * (G_M1[:,p] * G_M2[:,q])

        # Expand for broadcasting: [B, P+1, 1, N, N] * [B, 1, P+1, N, N]
        G1_exp = G_M1.unsqueeze(2)   # [B, P+1, 1, N, N]
        G2_exp = G_M2.unsqueeze(1)   # [B, 1, P+1, N, N]

        # Element-wise product of all (p, q) pairs
        cross_products = G1_exp * G2_exp  # [B, P+1, P+1, N, N]

        # Apply learned weights and sum
        weights = self.fusion_weights.view(1, P_plus_1, P_plus_1, 1, 1)  # broadcast
        G_fused = (weights * cross_products).sum(dim=(1, 2))  # [B, N, N]

        return G_fused


class FusionClassifier(nn.Module):
    """MLP classifier for fused graph representations."""

    def __init__(self, N=32, num_classes=1000):
        super().__init__()
        self.N = N

        self.mlp = nn.Sequential(
            nn.Linear(N * N, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, G_fused):
        """
        Args:
            G_fused: [B, N, N] fused graph
        Returns:
            logits: [B, num_classes]
        """
        h = G_fused.reshape(G_fused.size(0), -1)  # [B, N*N]
        logits = self.mlp(h)
        return logits

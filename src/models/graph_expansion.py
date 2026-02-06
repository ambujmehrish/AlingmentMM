import torch
import torch.nn as nn


class GraphExpansion(nn.Module):
    """Expands relationship graphs via element-wise (Hadamard) powers.

    Given R, produces [R^0, R^1, R^2, ..., R^P] where R^p is the
    element-wise p-th power of R.
    """

    def __init__(self, order=3):
        super().__init__()
        self.order = order

    def forward(self, R):
        """
        Args:
            R: [B, N, N] relationship matrix
        Returns:
            expanded: [B, P+1, N, N] expanded graphs [R^0, R^1, ..., R^P]
        """
        B, N, _ = R.shape
        P = self.order

        expanded = []

        # R^0 = Identity matrix
        R_identity = torch.eye(N, device=R.device, dtype=R.dtype).unsqueeze(0).expand(B, -1, -1)
        expanded.append(R_identity)

        # R^1 = R itself
        expanded.append(R)

        # R^p = R ⊙ R ⊙ ... (element-wise p-th power)
        R_p = R
        for p in range(2, P + 1):
            R_p = R_p * R  # Hadamard product (element-wise)
            expanded.append(R_p)

        # Stack into [B, P+1, N, N]
        expanded = torch.stack(expanded, dim=1)

        return expanded

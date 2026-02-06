import torch
import torch.nn.functional as F


def graph_similarity(R_i, R_j, eps=1e-8):
    """Compute normalized Frobenius inner product between graph matrices.

    Args:
        R_i: [B, N, N] relationship matrix batch i
        R_j: [B, N, N] relationship matrix batch j
    Returns:
        sim: [B] similarity scores in [-1, 1]
    """
    B = R_i.size(0)
    R_i_flat = R_i.reshape(B, -1)  # [B, N*N]
    R_j_flat = R_j.reshape(B, -1)  # [B, N*N]

    # Frobenius inner product
    inner = (R_i_flat * R_j_flat).sum(dim=-1)  # [B]

    # Frobenius norms
    norm_i = torch.norm(R_i_flat, p=2, dim=-1)  # [B]
    norm_j = torch.norm(R_j_flat, p=2, dim=-1)  # [B]

    sim = inner / (norm_i * norm_j + eps)
    return sim


def graph_similarity_matrix(R_M1, R_M2, eps=1e-8):
    """Compute pairwise graph similarity matrix between two batches.

    Args:
        R_M1: [B, N, N] graphs from modality 1
        R_M2: [B, N, N] graphs from modality 2
    Returns:
        sim_matrix: [B, B] pairwise similarity matrix
    """
    B, N, _ = R_M1.shape

    # Flatten graphs
    R1_flat = R_M1.reshape(B, -1)  # [B, N*N]
    R2_flat = R_M2.reshape(B, -1)  # [B, N*N]

    # Normalize
    R1_norm = F.normalize(R1_flat, p=2, dim=-1)  # [B, N*N]
    R2_norm = F.normalize(R2_flat, p=2, dim=-1)  # [B, N*N]

    # Pairwise cosine similarity
    sim_matrix = R1_norm @ R2_norm.T  # [B, B]

    return sim_matrix


def graph_contrastive_loss(R_M1, R_M2, temperature=0.07):
    """Graph-level InfoNCE contrastive loss.

    Paired samples (same index) are positives, all others are negatives.

    Args:
        R_M1: [B, N, N] graphs from modality 1
        R_M2: [B, N, N] graphs from modality 2
        temperature: temperature scaling parameter
    Returns:
        loss: scalar contrastive loss
    """
    B = R_M1.size(0)

    # Compute full pairwise similarity matrix [B, B]
    sim_matrix = graph_similarity_matrix(R_M1, R_M2) / temperature

    # InfoNCE: positive pairs on diagonal
    labels = torch.arange(B, device=R_M1.device)

    # Symmetric loss
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.T, labels)

    return (loss_12 + loss_21) / 2


def soft_graph_binding_loss(R_M1, R_M2, y_M1, y_M2, eps=1e-8):
    """Soft binding loss using graph similarity distributions and label overlap.

    Aligns graph similarity distribution to label-based similarity distribution
    via KL divergence.

    Args:
        R_M1: [B, N, N] graphs from modality 1
        R_M2: [B, N, N] graphs from modality 2
        y_M1: [B] class labels for modality 1
        y_M2: [B] class labels for modality 2
    Returns:
        loss: scalar loss
    """
    # Compute graph similarity matrices
    S_graph_12 = graph_similarity_matrix(R_M1, R_M2)  # [B, B]

    # Compute label similarity (binary: same class = 1, different = 0)
    S_label_12 = (y_M1.unsqueeze(1) == y_M2.unsqueeze(0)).float()  # [B, B]

    # Normalize label similarities to probability distribution per row
    row_sums = S_label_12.sum(dim=-1, keepdim=True)
    # Handle rows with no matching labels (uniform distribution)
    no_match = (row_sums == 0).squeeze(-1)
    S_label_12 = S_label_12 / (row_sums + eps)
    if no_match.any():
        S_label_12[no_match] = 1.0 / S_label_12.size(1)

    # KL divergence: align graph similarity to label similarity
    loss = F.kl_div(
        F.log_softmax(S_graph_12, dim=-1),
        S_label_12,
        reduction='batchmean'
    )

    return loss


def anchor_distillation_loss(R_source, R_anchor_1, R_anchor_2):
    """Anchor-based graph distillation.

    The source modality's cross-modal similarity should match
    that of the anchor modalities.

    Args:
        R_source: [B, N, N] source modality graphs (e.g., point cloud)
        R_anchor_1: [B, N, N] anchor modality 1 graphs (e.g., text)
        R_anchor_2: [B, N, N] anchor modality 2 graphs (e.g., image)
    Returns:
        loss: scalar loss
    """
    # Source -> Anchor1 similarity should match Anchor2 -> Anchor1
    sim_source_a1 = graph_similarity_matrix(R_source, R_anchor_1)   # [B, B]
    sim_a2_a1 = graph_similarity_matrix(R_anchor_2, R_anchor_1)     # [B, B]

    loss1 = F.kl_div(
        F.log_softmax(sim_source_a1, dim=-1),
        F.softmax(sim_a2_a1, dim=-1),
        reduction='batchmean'
    )

    # Source -> Anchor2 similarity should match Anchor1 -> Anchor2
    sim_source_a2 = graph_similarity_matrix(R_source, R_anchor_2)   # [B, B]
    sim_a1_a2 = graph_similarity_matrix(R_anchor_1, R_anchor_2)     # [B, B]

    loss2 = F.kl_div(
        F.log_softmax(sim_source_a2, dim=-1),
        F.softmax(sim_a1_a2, dim=-1),
        reduction='batchmean'
    )

    return loss1 + loss2


def graph_knowledge_distillation_loss(R_teacher, R_student):
    """Distill graph structure from teacher to student modality.

    Minimizes the Frobenius norm of the difference between teacher
    and student relationship graphs.

    Args:
        R_teacher: [B, N, N] teacher graphs (detached)
        R_student: [B, N, N] student graphs
    Returns:
        loss: scalar loss
    """
    diff = R_teacher.detach() - R_student
    loss = torch.norm(diff.reshape(diff.size(0), -1), p=2, dim=-1).mean()
    return loss


def graph_regularization_loss(R, lambda_sparse=0.01, lambda_cluster=0.01, lambda_rank=0.01):
    """Regularization losses for encouraging good graph structure.

    Includes sparsity, clustering, and low-rank penalties.

    Args:
        R: [B, N, N] relationship matrix
        lambda_sparse: weight for L1 sparsity
        lambda_cluster: weight for clustering (maximize diagonal)
        lambda_rank: weight for nuclear norm (low-rank)
    Returns:
        loss: scalar combined regularization loss
    """
    B, N, _ = R.shape

    loss = torch.tensor(0.0, device=R.device, dtype=R.dtype)

    # Sparsity: L1 norm of off-diagonal elements
    if lambda_sparse > 0:
        mask = 1.0 - torch.eye(N, device=R.device, dtype=R.dtype)
        L_sparse = (torch.abs(R) * mask.unsqueeze(0)).mean()
        loss = loss + lambda_sparse * L_sparse

    # Clustering: encourage high self-similarity (diagonal should be maximal)
    if lambda_cluster > 0:
        diag = torch.diagonal(R, dim1=-2, dim2=-1)  # [B, N]
        L_cluster = -diag.mean()
        loss = loss + lambda_cluster * L_cluster

    # Low-rank: nuclear norm approximation via sum of singular values
    if lambda_rank > 0:
        # Use SVD; for efficiency, only compute singular values
        try:
            s = torch.linalg.svdvals(R)  # [B, N]
            L_rank = s.sum(dim=-1).mean()
            loss = loss + lambda_rank * L_rank
        except Exception:
            pass  # Skip if SVD fails for numerical reasons

    return loss

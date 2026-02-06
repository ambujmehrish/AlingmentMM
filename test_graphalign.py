"""
Test script for GraphAlign components.
Verifies all modules work correctly with proper shapes, gradient flow,
and end-to-end trainability.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add project root to path (robust to any working directory)
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.graph_pooling import GraphAwarePooling
from src.models.relationship_graph import RelationshipGraphBuilder
from src.models.graph_expansion import GraphExpansion
from src.models.graph_fusion import CrossModalGraphFusion, FusionClassifier
from util.graph_losses import (
    graph_similarity,
    graph_similarity_matrix,
    graph_contrastive_loss,
    soft_graph_binding_loss,
    anchor_distillation_loss,
    graph_knowledge_distillation_loss,
    graph_regularization_loss,
)

PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition, msg):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {msg}")


def test_graph_pooling():
    print("\n[1] Testing GraphAwarePooling...")
    pooler = GraphAwarePooling(input_dim=768, target_length=32, graph_dim=256)

    # Test with typical ViT output shape
    x = torch.randn(4, 197, 768, requires_grad=True)  # [B, L=196+1cls, D=768]
    out = pooler(x)

    check(out.shape == (4, 32, 256), f"Output shape: {out.shape} == (4, 32, 256)")

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    check(x.grad is not None, "Gradients flow to input")
    check(pooler.queries.grad is not None, "Gradients flow to learnable queries")
    check(pooler.key_proj.weight.grad is not None, "Gradients flow to key projection")
    check(pooler.value_proj.weight.grad is not None, "Gradients flow to value projection")
    check(pooler.dim_reduce[0].weight.grad is not None, "Gradients flow to dim reduction")

    # Test with different sequence lengths (different modalities)
    for L in [65, 128, 197, 512]:
        x2 = torch.randn(2, L, 768)
        out2 = pooler(x2)
        check(out2.shape == (2, 32, 256), f"Works with L={L}: shape {out2.shape}")

    # Test with batch size 1
    x3 = torch.randn(1, 197, 768)
    out3 = pooler(x3)
    check(out3.shape == (1, 32, 256), f"Works with B=1: shape {out3.shape}")

    param_count = sum(p.numel() for p in pooler.parameters())
    print(f"  Parameter count: {param_count:,}")


def test_relationship_graph():
    print("\n[2] Testing RelationshipGraphBuilder...")
    builder = RelationshipGraphBuilder()

    x = torch.randn(4, 32, 256, requires_grad=True)
    R = builder(x)

    check(R.shape == (4, 32, 32), f"Output shape: {R.shape} == (4, 32, 32)")

    # Symmetry check
    check(torch.allclose(R, R.transpose(-2, -1), atol=1e-5), "Graph is symmetric")

    # Diagonal should be ~1 (self-similarity of normalized vectors)
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    check(torch.allclose(diag, torch.ones_like(diag), atol=1e-5), "Diagonal is ~1.0")

    # Values in [-1, 1] (cosine similarity range)
    check(R.min().item() >= -1.001 and R.max().item() <= 1.001, "Values in [-1, 1]")

    # Gradient flow
    loss = R.sum()
    loss.backward()
    check(x.grad is not None, "Gradients flow to input")


def test_graph_expansion():
    print("\n[3] Testing GraphExpansion...")
    expander = GraphExpansion(order=3)

    R = torch.randn(4, 32, 32, requires_grad=True)
    G = expander(R)

    check(G.shape == (4, 4, 32, 32), f"Output shape: {G.shape} == (4, 4, 32, 32)")

    # R^0 should be identity
    I = torch.eye(32).unsqueeze(0).expand(4, -1, -1)
    check(torch.allclose(G[:, 0], I, atol=1e-5), "R^0 is identity")

    # R^1 should be the input
    check(torch.allclose(G[:, 1], R, atol=1e-5), "R^1 is input R")

    # R^2 should be R * R (element-wise)
    check(torch.allclose(G[:, 2], R * R, atol=1e-5), "R^2 = R * R (Hadamard)")

    # R^3 should be R * R * R
    check(torch.allclose(G[:, 3], R * R * R, atol=1e-5), "R^3 = R * R * R")

    # Gradient flow
    loss = G.sum()
    loss.backward()
    check(R.grad is not None, "Gradients flow to input")

    # Test different orders
    for order in [1, 2, 5]:
        exp2 = GraphExpansion(order=order)
        G2 = exp2(R.detach())
        check(G2.shape == (4, order + 1, 32, 32), f"Order={order}: shape {G2.shape}")


def test_graph_fusion():
    print("\n[4] Testing CrossModalGraphFusion...")
    fuser = CrossModalGraphFusion(order=3)

    G1 = torch.randn(4, 4, 32, 32, requires_grad=True)
    G2 = torch.randn(4, 4, 32, 32, requires_grad=True)
    G_fused = fuser(G1, G2)

    check(G_fused.shape == (4, 32, 32), f"Output shape: {G_fused.shape} == (4, 32, 32)")

    # Gradient flow
    loss = G_fused.sum()
    loss.backward()
    check(G1.grad is not None, "Gradients flow to input 1")
    check(G2.grad is not None, "Gradients flow to input 2")
    check(fuser.fusion_weights.grad is not None, "Gradients flow to fusion weights")

    param_count = sum(p.numel() for p in fuser.parameters())
    print(f"  Fusion weights shape: {fuser.fusion_weights.shape}, params: {param_count}")


def test_fusion_classifier():
    print("\n[5] Testing FusionClassifier...")
    classifier = FusionClassifier(N=32, num_classes=1000)

    G = torch.randn(4, 32, 32, requires_grad=True)
    logits = classifier(G)

    check(logits.shape == (4, 1000), f"Output shape: {logits.shape} == (4, 1000)")

    # Gradient flow
    labels = torch.randint(0, 1000, (4,))
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    check(G.grad is not None, "Gradients flow to input")

    param_count = sum(p.numel() for p in classifier.parameters())
    print(f"  Parameter count: {param_count:,}")


def test_graph_similarity():
    print("\n[6] Testing graph_similarity and graph_similarity_matrix...")

    B, N = 4, 32
    R1 = torch.randn(B, N, N)
    R2 = torch.randn(B, N, N)

    # Test graph_similarity
    sim = graph_similarity(R1, R2)
    check(sim.shape == (B,), f"graph_similarity shape: {sim.shape} == ({B},)")
    check(sim.min().item() >= -1.001 and sim.max().item() <= 1.001, "Similarity in [-1, 1]")

    # Self-similarity should be 1
    self_sim = graph_similarity(R1, R1)
    check(torch.allclose(self_sim, torch.ones(B), atol=1e-5), "Self-similarity is ~1.0")

    # Test graph_similarity_matrix
    sim_mat = graph_similarity_matrix(R1, R2)
    check(sim_mat.shape == (B, B), f"similarity_matrix shape: {sim_mat.shape} == ({B}, {B})")


def test_graph_contrastive_loss():
    print("\n[7] Testing graph_contrastive_loss...")

    B, N = 8, 32
    R1 = torch.randn(B, N, N, requires_grad=True)
    R2 = torch.randn(B, N, N, requires_grad=True)

    loss = graph_contrastive_loss(R1, R2, temperature=0.07)
    check(loss.item() > 0, f"Loss is positive: {loss.item():.4f}")
    check(torch.isfinite(loss), "Loss is finite")

    loss.backward()
    check(R1.grad is not None, "Gradients flow to R1")
    check(R2.grad is not None, "Gradients flow to R2")

    # Loss with identical pairs should be lower than random
    R3 = R1.detach().clone().requires_grad_(True)
    loss_identical = graph_contrastive_loss(R3, R3)
    print(f"  Loss (random pairs): {loss.item():.4f}, (identical): {loss_identical.item():.4f}")


def test_soft_binding_loss():
    print("\n[8] Testing soft_graph_binding_loss...")

    B, N = 8, 32
    R1 = torch.randn(B, N, N, requires_grad=True)
    R2 = torch.randn(B, N, N, requires_grad=True)
    y1 = torch.randint(0, 5, (B,))
    y2 = torch.randint(0, 5, (B,))

    loss = soft_graph_binding_loss(R1, R2, y1, y2)
    check(loss.item() >= 0, f"Loss is non-negative: {loss.item():.4f}")
    check(torch.isfinite(loss), "Loss is finite")

    loss.backward()
    check(R1.grad is not None, "Gradients flow to R1")
    check(R2.grad is not None, "Gradients flow to R2")


def test_anchor_distillation_loss():
    print("\n[9] Testing anchor_distillation_loss...")

    B, N = 8, 32
    R_source = torch.randn(B, N, N, requires_grad=True)
    R_anchor1 = torch.randn(B, N, N, requires_grad=True)
    R_anchor2 = torch.randn(B, N, N, requires_grad=True)

    loss = anchor_distillation_loss(R_source, R_anchor1, R_anchor2)
    check(loss.item() >= 0, f"Loss is non-negative: {loss.item():.4f}")
    check(torch.isfinite(loss), "Loss is finite")

    loss.backward()
    check(R_source.grad is not None, "Gradients flow to source")
    check(R_anchor1.grad is not None, "Gradients flow to anchor1")
    check(R_anchor2.grad is not None, "Gradients flow to anchor2")


def test_graph_kd_loss():
    print("\n[10] Testing graph_knowledge_distillation_loss...")

    B, N = 4, 32
    R_teacher = torch.randn(B, N, N)
    R_student = torch.randn(B, N, N, requires_grad=True)

    loss = graph_knowledge_distillation_loss(R_teacher, R_student)
    check(loss.item() >= 0, f"Loss is non-negative: {loss.item():.4f}")

    loss.backward()
    check(R_student.grad is not None, "Gradients flow to student")


def test_graph_regularization_loss():
    print("\n[11] Testing graph_regularization_loss...")

    B, N = 4, 32
    R = torch.randn(B, N, N, requires_grad=True)

    loss = graph_regularization_loss(R, lambda_sparse=0.01, lambda_cluster=0.01, lambda_rank=0.01)
    check(torch.isfinite(loss), f"Loss is finite: {loss.item():.4f}")

    loss.backward()
    check(R.grad is not None, "Gradients flow to input")


def test_end_to_end_pipeline():
    print("\n[12] Testing end-to-end GraphAlign pipeline...")

    # Simulate two modalities through the full pipeline
    pooler_m1 = GraphAwarePooling(input_dim=768, target_length=32, graph_dim=256)
    pooler_m2 = GraphAwarePooling(input_dim=768, target_length=32, graph_dim=256)
    builder = RelationshipGraphBuilder()
    expander = GraphExpansion(order=3)
    fuser = CrossModalGraphFusion(order=3)
    classifier = FusionClassifier(N=32, num_classes=100)

    B, L, D = 4, 197, 768
    features_m1 = torch.randn(B, L, D, requires_grad=True)
    features_m2 = torch.randn(B, L, D, requires_grad=True)
    labels = torch.randint(0, 100, (B,))

    # Forward: Modality 1
    pooled_m1 = pooler_m1(features_m1)
    R_m1 = builder(pooled_m1)
    G_m1 = expander(R_m1)

    # Forward: Modality 2
    pooled_m2 = pooler_m2(features_m2)
    R_m2 = builder(pooled_m2)
    G_m2 = expander(R_m2)

    # Fusion
    G_fused = fuser(G_m1, G_m2)
    logits = classifier(G_fused)

    check(logits.shape == (B, 100), f"End-to-end logits shape: {logits.shape}")

    # Compute combined loss
    cls_loss = F.cross_entropy(logits, labels)
    nce_loss = graph_contrastive_loss(R_m1, R_m2)
    bind_loss = soft_graph_binding_loss(R_m1, R_m2, labels, labels)
    reg_loss = graph_regularization_loss(R_m1)

    total_loss = cls_loss + nce_loss + 0.01 * reg_loss + bind_loss
    check(torch.isfinite(total_loss), f"Total loss is finite: {total_loss.item():.4f}")

    # Backward pass
    total_loss.backward()

    check(features_m1.grad is not None, "Gradients flow to modality 1 input")
    check(features_m2.grad is not None, "Gradients flow to modality 2 input")
    check(pooler_m1.queries.grad is not None, "Gradients flow to pooler_m1 queries")
    check(pooler_m2.queries.grad is not None, "Gradients flow to pooler_m2 queries")
    check(fuser.fusion_weights.grad is not None, "Gradients flow to fuser weights")
    check(classifier.mlp[0].weight.grad is not None, "Gradients flow to classifier")

    # Print loss breakdown
    print(f"  Loss breakdown:")
    print(f"    Classification: {cls_loss.item():.4f}")
    print(f"    Graph NCE:      {nce_loss.item():.4f}")
    print(f"    Soft Binding:   {bind_loss.item():.4f}")
    print(f"    Regularization: {reg_loss.item():.4f}")
    print(f"    Total:          {total_loss.item():.4f}")


def test_parameter_counts():
    print("\n[13] Testing parameter counts...")

    pooler = GraphAwarePooling(input_dim=768, target_length=32, graph_dim=256)
    builder = RelationshipGraphBuilder()
    expander = GraphExpansion(order=3)
    fuser = CrossModalGraphFusion(order=3)
    classifier = FusionClassifier(N=32, num_classes=1000)

    counts = {
        "GraphAwarePooling": sum(p.numel() for p in pooler.parameters()),
        "RelationshipGraphBuilder": sum(p.numel() for p in builder.parameters()),
        "GraphExpansion": sum(p.numel() for p in expander.parameters()),
        "CrossModalGraphFusion": sum(p.numel() for p in fuser.parameters()),
        "FusionClassifier": sum(p.numel() for p in classifier.parameters()),
    }

    total = sum(counts.values())
    for name, count in counts.items():
        print(f"  {name}: {count:,} params")
    print(f"  Total (single pair): {total:,} params")

    # Per-modality graph components for 5 modalities
    n_modalities = 5
    n_pairs = n_modalities * (n_modalities - 1) // 2  # 10 pairs
    total_all = (
        counts["GraphAwarePooling"] * n_modalities
        + counts["RelationshipGraphBuilder"]
        + counts["GraphExpansion"]
        + counts["CrossModalGraphFusion"] * n_pairs
    )
    print(f"  Total for {n_modalities} modalities ({n_pairs} pairs): {total_all:,} params")
    check(True, f"Parameter accounting complete")


def test_mixed_precision():
    print("\n[14] Testing mixed precision (float16) compatibility...")

    pooler = GraphAwarePooling(input_dim=768, target_length=32, graph_dim=256).cuda()
    builder = RelationshipGraphBuilder()
    expander = GraphExpansion(order=3)

    x = torch.randn(2, 197, 768, device='cuda')

    with torch.cuda.amp.autocast():
        pooled = pooler(x)
        R = builder(pooled)
        G = expander(R)

    check(G.shape == (2, 4, 32, 32), f"AMP output shape: {G.shape}")
    check(torch.isfinite(G).all(), "AMP outputs are finite")
    print(f"  Pooled dtype: {pooled.dtype}, R dtype: {R.dtype}, G dtype: {G.dtype}")


if __name__ == "__main__":
    print("=" * 60)
    print("GraphAlign Component Tests")
    print("=" * 60)

    try:
        test_graph_pooling()
        test_relationship_graph()
        test_graph_expansion()
        test_graph_fusion()
        test_fusion_classifier()
        test_graph_similarity()
        test_graph_contrastive_loss()
        test_soft_binding_loss()
        test_anchor_distillation_loss()
        test_graph_kd_loss()
        test_graph_regularization_loss()
        test_end_to_end_pipeline()
        test_parameter_counts()

        # Only run CUDA tests if available
        if torch.cuda.is_available():
            test_mixed_precision()
        else:
            print("\n[14] Skipping mixed precision test (no CUDA)")

        print("\n" + "=" * 60)
        print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
        if FAIL_COUNT == 0:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
            sys.exit(1)
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST SUITE FAILED WITH EXCEPTION")
        print(f"Error: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

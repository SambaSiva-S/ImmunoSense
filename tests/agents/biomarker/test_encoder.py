"""Tests for Layer 2 Pillar A — DiseaseEncoder + NTXentLoss."""

import numpy as np
import pytest
import torch

from immunosense.agents.biomarker.layer2.encoder import (
    DiseaseEncoder,
    NTXentLoss,
    compute_centroids,
    get_similarities,
    train_contrastive_encoder,
)


def test_encoder_output_is_unit_norm():
    """All forward-pass outputs must lie on the unit sphere."""
    encoder = DiseaseEncoder(input_dim=28, embedding_dim=128)
    encoder.eval()
    x = torch.randn(10, 28)
    with torch.no_grad():
        z = encoder(x)
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones(10), atol=1e-5)


def test_encoder_output_shape():
    """Output shape is (batch, embedding_dim)."""
    encoder = DiseaseEncoder(input_dim=28, embedding_dim=64)
    encoder.eval()
    x = torch.randn(5, 28)
    with torch.no_grad():
        z = encoder(x)
    assert z.shape == (5, 64)


def test_encoder_encode_single_returns_1d_unit_vector():
    encoder = DiseaseEncoder(input_dim=28, embedding_dim=128)
    encoder.eval()
    x = np.random.randn(28).astype(np.float32)
    emb = encoder.encode_single(x)
    assert emb.shape == (128,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_encoder_encode_single_accepts_2d():
    encoder = DiseaseEncoder(input_dim=28)
    encoder.eval()
    x = np.random.randn(1, 28).astype(np.float32)
    emb = encoder.encode_single(x)
    assert emb.shape == (128,)


def test_ntxent_loss_returns_scalar():
    loss_fn = NTXentLoss(temperature=0.1)
    embeddings = torch.randn(8, 64)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    labels = torch.LongTensor([0, 0, 1, 1, 2, 2, 0, 1])
    loss = loss_fn(embeddings, labels)
    assert loss.dim() == 0  # scalar


def test_ntxent_loss_positive():
    """NTXent should produce non-negative loss."""
    loss_fn = NTXentLoss(temperature=0.1)
    embeddings = torch.randn(8, 64)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    labels = torch.LongTensor([0, 0, 1, 1, 2, 2, 0, 1])
    loss = loss_fn(embeddings, labels)
    assert loss.item() >= 0.0


def test_train_contrastive_encoder_returns_trained_model():
    """Training reduces loss over epochs (with a clearly separable signal)."""
    np.random.seed(0)
    torch.manual_seed(0)
    # 3 classes, well-separated in feature space
    X = np.vstack([
        np.random.randn(80, 10) + np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(80, 10) + np.array([0, 3, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(80, 10) + np.array([0, 0, 3, 0, 0, 0, 0, 0, 0, 0]),
    ])
    y = np.repeat([0, 1, 2], 80)
    encoder, losses = train_contrastive_encoder(
        X, y, embedding_dim=32, epochs=10, batch_size=16, verbose=False,
    )
    assert isinstance(encoder, DiseaseEncoder)
    assert len(losses) == 10
    # Loss should decrease meaningfully
    assert losses[-1] < losses[0]


def test_compute_centroids_returns_unit_vectors():
    encoder = DiseaseEncoder(input_dim=10, embedding_dim=16)
    encoder.eval()
    X = np.random.randn(60, 10).astype(np.float32)
    y = np.repeat([0, 1, 2], 20)
    class_names = ["A", "B", "C"]
    centroids = compute_centroids(encoder, X, y, class_names)
    assert set(centroids.keys()) == {"A", "B", "C"}
    for name, c in centroids.items():
        norm = np.linalg.norm(c)
        # Zero-class case allowed, otherwise unit length
        if norm > 1e-9:
            assert abs(norm - 1.0) < 1e-5


def test_compute_centroids_handles_empty_class():
    """A class with no training samples gets a zero centroid."""
    encoder = DiseaseEncoder(input_dim=10, embedding_dim=16)
    encoder.eval()
    X = np.random.randn(20, 10).astype(np.float32)
    y = np.zeros(20, dtype=int)  # all class 0
    centroids = compute_centroids(encoder, X, y, ["A", "B"])
    assert "B" in centroids
    assert np.allclose(centroids["B"], np.zeros(16))


def test_get_similarities_returns_per_class_values():
    centroids = {"A": np.array([1, 0, 0]), "B": np.array([0, 1, 0])}
    emb = np.array([1, 0, 0])
    sims = get_similarities(emb, centroids)
    assert abs(sims["A"] - 1.0) < 1e-9
    assert abs(sims["B"]) < 1e-9


def test_get_similarities_handles_zero_vectors():
    centroids = {"A": np.array([0.0, 0.0, 0.0])}
    emb = np.array([1.0, 0.0, 0.0])
    sims = get_similarities(emb, centroids)
    assert sims["A"] == 0.0

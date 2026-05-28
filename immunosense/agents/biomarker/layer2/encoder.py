"""Pillar A: Spatial — Contrastive Encoder (PyTorch).

DiseaseEncoder: 3-layer MLP that projects biomarker vectors onto the unit
sphere as 128-dim embeddings. Trained with NT-Xent contrastive loss to
cluster same-disease patients together.

After training, disease centroids are computed by averaging training
embeddings per class and renormalizing to unit length. Inference: cosine
similarity to each centroid -> the disease whose centroid is most similar
becomes the Pillar A prediction.

This module requires PyTorch. If torch is not installed, importing this
module will raise ImportError with installation instructions.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError(
        "PyTorch is required for Agent 1 Pillar A (contrastive encoder). "
        "Install with: pip install immunosense[biomarker]"
    ) from e

import numpy as np

from immunosense.agents.biomarker.constants import (
    LAYER2_EMBEDDING_DIM,
    LAYER2_HYPERPARAMS,
)


class DiseaseEncoder(nn.Module):
    """3-layer MLP encoder mapping biomarker vectors to unit-sphere embeddings.

    Architecture:
        input -> Linear(64) + BN + ReLU + Dropout(0.2)
              -> Linear(128) + BN + ReLU + Dropout(0.2)
              -> Linear(embedding_dim)
              -> Linear(embedding_dim)  # projection head
              -> L2-normalize (unit sphere)

    Args:
        input_dim: number of biomarker features (typically 28: 14 values + 14 missing flags)
        embedding_dim: output embedding dimension (default 128)
    """

    def __init__(self, input_dim: int, embedding_dim: int = LAYER2_EMBEDDING_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """x: (batch, input_dim) -> (batch, embedding_dim) on the unit sphere."""
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

    def encode_single(self, x: np.ndarray) -> np.ndarray:
        """Encode one already-scaled feature vector to its embedding.

        Args:
            x: shape (input_dim,) or (1, input_dim)

        Returns:
            np.ndarray of shape (embedding_dim,) on the unit sphere.
        """
        self.eval()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        with torch.no_grad():
            t = torch.FloatTensor(x)
            # BatchNorm in eval mode uses running stats — safe for batch=1
            z = self(t).cpu().numpy()
        return z[0]


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss for supervised contrastive learning.

    Pulls together embeddings with the same label, pushes apart embeddings
    with different labels. Self-similarity (diagonal) is masked out.

    Args:
        temperature: softmax temperature; lower = sharper contrasts.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: "torch.Tensor", labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Args:
            embeddings: (batch, embedding_dim), L2-normalized
            labels: (batch,) integer class labels

        Returns:
            scalar loss tensor
        """
        # Cosine similarity matrix (batch, batch)
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Positive-pair mask: same label, excluding self
        labels_2d = labels.unsqueeze(1)
        positive_mask = (labels_2d == labels_2d.t()).float()
        identity = torch.eye(len(embeddings), device=embeddings.device)
        positive_mask = positive_mask - identity

        # Mask out self-similarity for numerical stability
        sim_matrix = sim_matrix - identity * 1e9

        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Mean log-prob over positive pairs (per row)
        positive_log_prob = (
            (log_prob * positive_mask).sum(dim=1)
            / positive_mask.sum(dim=1).clamp(min=1)
        )

        return -positive_log_prob.mean()


def train_contrastive_encoder(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    embedding_dim: int = LAYER2_EMBEDDING_DIM,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    weight_decay: float = None,
    temperature: float = None,
    random_seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Train the DiseaseEncoder on scaled features + class labels.

    Args:
        X_train_scaled: (n, input_dim) StandardScaler-transformed features
        y_train: (n,) integer class labels
        embedding_dim: output embedding dimension
        epochs, batch_size, lr, weight_decay, temperature: training hyperparams
            (defaults from LAYER2_HYPERPARAMS)
        random_seed: torch.manual_seed for reproducibility
        verbose: print progress every 20 epochs

    Returns:
        Tuple (trained_encoder, list_of_epoch_losses).
    """
    epochs = epochs or LAYER2_HYPERPARAMS["encoder_epochs"]
    batch_size = batch_size or LAYER2_HYPERPARAMS["encoder_batch_size"]
    lr = lr or LAYER2_HYPERPARAMS["encoder_lr"]
    weight_decay = weight_decay or LAYER2_HYPERPARAMS["encoder_weight_decay"]
    temperature = temperature or LAYER2_HYPERPARAMS["ntxent_temperature"]

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    input_dim = X_train_scaled.shape[1]
    encoder = DiseaseEncoder(input_dim=input_dim, embedding_dim=embedding_dim)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.FloatTensor(X_train_scaled)
    y_t = torch.LongTensor(y_train)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    losses = []
    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            embeddings = encoder(batch_x)
            loss = criterion(embeddings, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    return encoder, losses


def compute_centroids(
    encoder: DiseaseEncoder,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    class_names: list,
) -> dict:
    """Compute one centroid per disease class on the unit sphere.

    Args:
        encoder: trained DiseaseEncoder
        X_train_scaled: (n, input_dim) scaled features used for training
        y_train: (n,) integer labels matching class_names ordering
        class_names: list of disease class name strings

    Returns:
        dict {class_name: np.ndarray of shape (embedding_dim,)}, L2-normalized.
    """
    encoder.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_train_scaled)
        embeddings = encoder(X_t).cpu().numpy()

    centroids = {}
    for i, disease in enumerate(class_names):
        mask = y_train == i
        if mask.sum() == 0:
            # No training samples for this class — initialize with zeros
            centroids[disease] = np.zeros(embeddings.shape[1])
            continue
        centroid = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        centroids[disease] = centroid

    return centroids


def get_similarities(
    embedding: np.ndarray, centroids: dict,
) -> dict:
    """Cosine similarity between one embedding and each disease centroid.

    Args:
        embedding: shape (embedding_dim,), assumed L2-normalized
        centroids: dict {disease: L2-normalized centroid vector}

    Returns:
        dict {disease: cosine_similarity in [-1, 1]}
    """
    sims = {}
    emb_norm = np.linalg.norm(embedding)
    for disease, centroid in centroids.items():
        cen_norm = np.linalg.norm(centroid)
        if emb_norm < 1e-9 or cen_norm < 1e-9:
            sims[disease] = 0.0
            continue
        sim = float(np.dot(embedding, centroid) / (emb_norm * cen_norm))
        sims[disease] = sim
    return sims

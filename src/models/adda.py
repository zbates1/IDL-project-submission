"""Adversarial Discriminative Domain Adaptation (ADDA) for sim-to-real transfer.

Implements the ADDA framework from Tzeng et al. 2017 (CVPR):
  1. Pre-train source encoder + classifier on labeled source data (already done)
  2. Initialize target encoder from source encoder weights
  3. Adversarial training: discriminator distinguishes source vs target features,
     target encoder fools the discriminator (GAN-style min-max)

Key difference from DANN: separate encoders for source/target, two-stage training,
and the source encoder is frozen during adversarial adaptation.

Usage in notebook::

    from src.models.adda import ADDA, adda_adapt

    adda = ADDA(source_backbone, feature_dim=512, num_classes=4)
    history = adda_adapt(adda, src_loader, tgt_loader, epochs=50, lr=1e-4)
    # Evaluate with target encoder:
    adda.target_encoder.eval()
    features = adda.target_encoder.get_features(real_x)
    logits = adda.classifier(features)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ADDADiscriminator(nn.Module):
    """Discriminator for ADDA: classifies features as source (1) or target (0)."""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ADDA(nn.Module):
    """Adversarial Discriminative Domain Adaptation.

    Args:
        source_backbone: Pre-trained source model with ``get_features()`` and ``fc``.
        feature_dim: Feature dimensionality. Default 512.
        num_classes: Number of classes. Default 4.
        disc_hidden: Discriminator hidden dim. Default 256.
    """

    def __init__(
        self,
        source_backbone: nn.Module,
        feature_dim: int = 512,
        num_classes: int = 4,
        disc_hidden: int = 256,
    ):
        super().__init__()
        # Source encoder is frozen
        self.source_encoder = source_backbone
        for p in self.source_encoder.parameters():
            p.requires_grad = False

        # Target encoder initialized from source weights
        self.target_encoder = copy.deepcopy(source_backbone)
        for p in self.target_encoder.parameters():
            p.requires_grad = True

        # Classifier from source (frozen)
        self.classifier = source_backbone.fc
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.discriminator = ADDADiscriminator(feature_dim, disc_hidden)

    def get_source_features(self, x: torch.Tensor) -> torch.Tensor:
        self.source_encoder.eval()
        with torch.no_grad():
            return self.source_encoder.get_features(x)

    def get_target_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder.get_features(x)


def adda_adapt(
    adda: ADDA,
    src_loader: torch.utils.data.DataLoader,
    tgt_x: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: torch.device | str = "cuda",
    verbose: bool = True,
) -> dict:
    """Run ADDA adversarial adaptation.

    Stage 1 (pre-training) is assumed done — source_backbone should already
    be trained on labeled source data.

    This function runs Stage 2: adversarial adaptation of the target encoder.

    Args:
        adda: ADDA model with pre-trained source encoder.
        src_loader: DataLoader for source (simulated) data.
        tgt_x: All target (real) input tensors ``(N, C, L)``.
        epochs: Number of adaptation epochs.
        lr: Learning rate for both target encoder and discriminator.
        batch_size: Batch size for target data sampling.
        device: Device to train on.
        verbose: Print progress.

    Returns:
        Dict with training history ('disc_loss', 'tgt_loss' per epoch).
    """
    adda = adda.to(device)
    tgt_x = tgt_x.to(device)

    # Optimizers: separate for discriminator and target encoder
    opt_disc = torch.optim.Adam(adda.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_tgt = torch.optim.Adam(adda.target_encoder.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()
    history = {"disc_loss": [], "tgt_loss": []}

    n_tgt = tgt_x.size(0)

    for epoch in range(1, epochs + 1):
        epoch_disc_loss = 0.0
        epoch_tgt_loss = 0.0
        n_steps = 0

        src_iter = iter(src_loader)

        for src_batch in src_iter:
            src_x_batch = src_batch[0].to(device)
            bs = src_x_batch.size(0)

            # Sample random target batch
            tgt_idx = torch.randint(0, n_tgt, (min(bs, n_tgt),))
            tgt_x_batch = tgt_x[tgt_idx]

            # --- Step 1: Train discriminator ---
            adda.discriminator.train()
            adda.target_encoder.eval()

            src_feats = adda.get_source_features(src_x_batch)
            with torch.no_grad():
                tgt_feats = adda.get_target_features(tgt_x_batch)

            src_pred = adda.discriminator(src_feats)
            tgt_pred = adda.discriminator(tgt_feats)

            d_loss = (
                criterion(src_pred, torch.ones_like(src_pred))
                + criterion(tgt_pred, torch.zeros_like(tgt_pred))
            ) / 2

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # --- Step 2: Train target encoder (fool discriminator) ---
            adda.target_encoder.train()
            tgt_feats = adda.get_target_features(tgt_x_batch)
            tgt_pred = adda.discriminator(tgt_feats)

            # Target encoder wants discriminator to think target = source (label=1)
            t_loss = criterion(tgt_pred, torch.ones_like(tgt_pred))

            opt_tgt.zero_grad()
            t_loss.backward()
            opt_tgt.step()

            epoch_disc_loss += d_loss.item()
            epoch_tgt_loss += t_loss.item()
            n_steps += 1

        avg_d = epoch_disc_loss / max(n_steps, 1)
        avg_t = epoch_tgt_loss / max(n_steps, 1)
        history["disc_loss"].append(avg_d)
        history["tgt_loss"].append(avg_t)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  ADDA epoch {epoch:>3d}/{epochs}  disc_loss={avg_d:.4f}  tgt_loss={avg_t:.4f}")

    return history

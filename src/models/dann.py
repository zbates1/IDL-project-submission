"""Domain-Adversarial Neural Network (DANN) for sim-to-real transfer.

Implements the DANN framework from Ganin & Lempitsky 2015:
  - Feature extractor: existing ResNet1D backbone (get_features → 512-dim)
  - Class classifier: existing FC head
  - Domain discriminator: MLP with gradient reversal layer (GRL)

The GRL negates gradients during backprop so the feature extractor learns
domain-invariant representations while the discriminator tries to distinguish
simulated from real data.

Usage in notebook::

    from src.models.dann import DANN, dann_train_step

    dann = DANN(backbone, feature_dim=512, num_classes=4)
    # In training loop:
    loss = dann_train_step(dann, src_batch, tgt_batch, optimizer, epoch, max_epochs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer (GRL) — negates gradients scaled by lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GRL as a module with configurable lambda scheduling."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """MLP domain classifier: features → P(source).

    Architecture follows Ganin 2015: two hidden layers with ReLU + dropout.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network.

    Wraps an existing backbone (ResNet1D with get_features + fc) and adds
    a GRL + domain discriminator head.

    Args:
        backbone: A model with ``get_features(x) → (B, feature_dim)`` and
            ``fc`` attribute (class classifier).
        feature_dim: Dimensionality of backbone features. Default 512.
        num_classes: Number of mechanism classes. Default 4.
        disc_hidden: Hidden dim for domain discriminator. Default 256.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 512,
        num_classes: int = 4,
        disc_hidden: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = backbone.fc
        self.grl = GradientReversalLayer()
        self.discriminator = DomainDiscriminator(feature_dim, disc_hidden)

    def forward(self, x: torch.Tensor, lambda_: float = 1.0):
        """Forward pass returning class logits and domain logits.

        Args:
            x: Input tensor ``(B, C, L)``.
            lambda_: GRL scaling factor (schedule this during training).

        Returns:
            Tuple of (class_logits, domain_logits).
        """
        features = self.backbone.get_features(x)
        class_logits = self.classifier(features)

        self.grl.set_lambda(lambda_)
        reversed_features = self.grl(features)
        domain_logits = self.discriminator(reversed_features)

        return class_logits, domain_logits


def schedule_lambda(epoch: int, max_epochs: int, gamma: float = 10.0) -> float:
    """Ganin 2015 progressive lambda schedule.

    Ramps from 0 → 1 over training using: λ = 2/(1+exp(-γp)) - 1
    where p = epoch/max_epochs.
    """
    import math
    p = epoch / max_epochs
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def dann_train_step(
    dann: DANN,
    src_x: torch.Tensor,
    src_y: torch.Tensor,
    tgt_x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_epochs: int,
    class_criterion: nn.Module | None = None,
    domain_weight: float = 1.0,
) -> dict:
    """One training step for DANN.

    Args:
        dann: DANN model.
        src_x: Source (simulated) input batch ``(B, C, L)``.
        src_y: Source labels ``(B,)``.
        tgt_x: Target (real) input batch ``(B', C, L)``.
        optimizer: Optimizer for all DANN parameters.
        epoch: Current epoch (for lambda schedule).
        max_epochs: Total epochs (for lambda schedule).
        class_criterion: Loss for classification. Default: CrossEntropyLoss.
        domain_weight: Weight for domain loss. Default 1.0.

    Returns:
        Dict with 'class_loss', 'domain_loss', 'total_loss', 'lambda'.
    """
    if class_criterion is None:
        class_criterion = nn.CrossEntropyLoss()

    domain_criterion = nn.BCEWithLogitsLoss()
    lambda_ = schedule_lambda(epoch, max_epochs)
    device = src_x.device

    dann.train()
    optimizer.zero_grad()

    # Source forward: class + domain
    src_class_logits, src_domain_logits = dann(src_x, lambda_=lambda_)
    class_loss = class_criterion(src_class_logits, src_y)
    src_domain_labels = torch.ones(src_x.size(0), 1, device=device)  # 1 = source
    src_domain_loss = domain_criterion(src_domain_logits, src_domain_labels)

    # Target forward: domain only (no class labels)
    _, tgt_domain_logits = dann(tgt_x, lambda_=lambda_)
    tgt_domain_labels = torch.zeros(tgt_x.size(0), 1, device=device)  # 0 = target
    tgt_domain_loss = domain_criterion(tgt_domain_logits, tgt_domain_labels)

    domain_loss = (src_domain_loss + tgt_domain_loss) / 2
    total_loss = class_loss + domain_weight * domain_loss

    total_loss.backward()
    optimizer.step()

    return {
        "class_loss": class_loss.item(),
        "domain_loss": domain_loss.item(),
        "total_loss": total_loss.item(),
        "lambda": lambda_,
    }

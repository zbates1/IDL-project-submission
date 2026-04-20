"""Few-shot fine-tuning for sim-to-real transfer.

Freezes the ResNet1D backbone and fine-tunes only the classification head
on a small number of labeled real examples per class. This is the simplest
transfer approach and serves as a strong baseline for DANN/ADDA.

Usage in notebook::

    from src.models.few_shot import few_shot_finetune, few_shot_sweep

    # Single run: 5 examples per class
    model, metrics = few_shot_finetune(backbone, real_X, real_y, n_per_class=5)

    # Full sweep: n = 1, 2, 5, 10, 20
    results = few_shot_sweep(backbone, real_X, real_y, class_names=MECHANISMS)
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def _split_few_shot(
    X: torch.Tensor,
    y: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> tuple:
    """Split data into n_per_class train examples + remaining test.

    Returns (train_X, train_y, test_X, test_y).
    """
    rng = np.random.default_rng(seed)
    classes = sorted(set(y.tolist()))
    train_idx, test_idx = [], []

    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) <= n_per_class:
            # Not enough samples — use all for training, none for test
            train_idx.extend(c_idx.tolist())
        else:
            chosen = rng.choice(c_idx, n_per_class, replace=False)
            train_idx.extend(chosen.tolist())
            test_idx.extend([i for i in c_idx if i not in chosen])

    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)

    return (
        X[train_idx],
        y[train_idx],
        X[test_idx] if test_idx else None,
        y[test_idx] if test_idx else None,
    )


def few_shot_finetune(
    backbone: nn.Module,
    real_X: torch.Tensor,
    real_y: np.ndarray,
    n_per_class: int = 5,
    num_classes: int = 4,
    feature_dim: int = 512,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | str = "cuda",
    verbose: bool = False,
) -> tuple:
    """Fine-tune only the FC head on n_per_class real examples.

    Args:
        backbone: Pre-trained model with ``get_features()`` method.
        real_X: All real input tensors ``(N, C, L)``.
        real_y: Integer labels ``(N,)``.
        n_per_class: Number of labeled examples per class for training.
        num_classes: Number of classes.
        feature_dim: Backbone feature dimensionality.
        epochs: Fine-tuning epochs.
        lr: Learning rate.
        seed: Random seed for split reproducibility.
        device: Device.
        verbose: Print progress.

    Returns:
        Tuple of (fine-tuned model, results dict with 'train_acc', 'test_acc',
        'n_per_class', 'n_train', 'n_test').
    """
    real_y = np.asarray(real_y)
    train_X, train_y, test_X, test_y = _split_few_shot(real_X, real_y, n_per_class, seed)

    # Deep copy backbone, freeze everything, replace FC
    model = copy.deepcopy(backbone).to(device)
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(feature_dim, num_classes).to(device)
    nn.init.kaiming_normal_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_X = train_X.to(device)
    train_y_t = torch.from_numpy(train_y).long().to(device)

    # Extract frozen features once
    model.eval()
    with torch.no_grad():
        train_feats = model.get_features(train_X)

    # Train FC head
    model.fc.train()
    for ep in range(1, epochs + 1):
        logits = model.fc(train_feats)
        loss = criterion(logits, train_y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_preds = model.fc(train_feats).argmax(1).cpu().numpy()
        train_acc = accuracy_score(train_y, train_preds)

        test_acc = None
        if test_X is not None:
            test_X = test_X.to(device)
            test_feats = model.get_features(test_X)
            test_preds = model.fc(test_feats).argmax(1).cpu().numpy()
            test_acc = accuracy_score(test_y, test_preds)

    results = {
        "n_per_class": n_per_class,
        "n_train": len(train_y),
        "n_test": len(test_y) if test_y is not None else 0,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }

    if verbose:
        print(f"  n={n_per_class}/class  train_acc={train_acc:.2%}  "
              f"test_acc={test_acc:.2%}" if test_acc else
              f"  n={n_per_class}/class  train_acc={train_acc:.2%}  (no holdout)")

    return model, results


def few_shot_sweep(
    backbone: nn.Module,
    real_X: torch.Tensor,
    real_y: np.ndarray,
    n_values: Optional[list] = None,
    class_names: Optional[list] = None,
    num_classes: int = 4,
    feature_dim: int = 512,
    epochs: int = 100,
    lr: float = 1e-3,
    n_trials: int = 5,
    device: torch.device | str = "cuda",
) -> list:
    """Sweep over different n_per_class values with multiple random seeds.

    Args:
        backbone: Pre-trained model.
        real_X: Real input tensors.
        real_y: Integer labels.
        n_values: List of n_per_class to try. Default [1, 2, 3, 5].
        class_names: Class names for display.
        num_classes: Number of classes.
        feature_dim: Backbone feature dim.
        epochs: Fine-tuning epochs per run.
        lr: Learning rate.
        n_trials: Number of random seeds per n value.
        device: Device.

    Returns:
        List of dicts with 'n_per_class', 'mean_test_acc', 'std_test_acc',
        'all_test_accs', 'mean_train_acc'.
    """
    if n_values is None:
        # Sensible defaults given 22 real samples
        n_values = [1, 2, 3, 5]

    real_y = np.asarray(real_y)
    min_class_count = min(np.bincount(real_y))

    results = []
    print(f"Few-shot sweep: {n_values}")
    print(f"  Real samples: {len(real_y)}, min class count: {min_class_count}")
    if class_names:
        print(f"  Classes: {class_names}")
    print()

    for n in n_values:
        if n > min_class_count:
            print(f"  n={n}: skipped (only {min_class_count} samples in smallest class)")
            continue

        test_accs = []
        train_accs = []

        for trial in range(n_trials):
            seed = 42 + trial
            _, res = few_shot_finetune(
                backbone, real_X, real_y,
                n_per_class=n, num_classes=num_classes,
                feature_dim=feature_dim, epochs=epochs, lr=lr,
                seed=seed, device=device, verbose=False,
            )
            train_accs.append(res["train_acc"])
            if res["test_acc"] is not None:
                test_accs.append(res["test_acc"])

        entry = {
            "n_per_class": n,
            "mean_train_acc": np.mean(train_accs),
            "mean_test_acc": np.mean(test_accs) if test_accs else None,
            "std_test_acc": np.std(test_accs) if test_accs else None,
            "all_test_accs": test_accs,
        }
        results.append(entry)

        test_str = (f"test={entry['mean_test_acc']:.2%} ± {entry['std_test_acc']:.2%}"
                    if test_accs else "no holdout")
        print(f"  n={n:>2d}/class ({n_trials} trials): "
              f"train={entry['mean_train_acc']:.2%}  {test_str}")

    return results

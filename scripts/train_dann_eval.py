"""DANN training + real-data evaluation.

Loads the pre-trained ResNet1D checkpoint, wraps it in DANN,
trains with adversarial domain adaptation using the 22 real samples
as unlabeled target data, then evaluates on real data (single-pass + TTA).

Usage:
    python -u scripts/train_dann_eval.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.cv_dataset import CVDataset, create_cv_dataloaders
from src.data.tta import tta_predict
from src.models.registry import get_model
from src.models.dann import DANN, schedule_lambda

import src.models.resnet1d  # noqa: F401

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = str(REPO_ROOT / "data" / "simulated")
REAL_DATA_PATH = str(REPO_ROOT / "data" / "real" / "processed" / "real_cvs.npz")
CKPT_DIR = str(REPO_ROOT / "checkpoints")
PRETRAINED_CKPT = os.path.join(CKPT_DIR, "best_resnet1d.pt")

# ── Mechanisms ──────────────────────────────────────────────────────────────
_sim_data = np.load(os.path.join(DATA_DIR, "simulated_cvs.npz"))
_labels_str = _sim_data["labels"]
MECHANISMS = sorted(set(_labels_str.tolist()))
MECHANISM_TO_IDX = {m: i for i, m in enumerate(MECHANISMS)}
N_CLASSES = len(MECHANISMS)
print(f"Mechanisms ({N_CLASSES}): {MECHANISMS}")

# ── Config ──────────────────────────────────────────────────────────────────
CNN_CONFIG = {
    "training": {"batch_size": 64},
    "data": {
        "root": DATA_DIR,
        "num_workers": 0,
        "pin_memory": False,
        "augmentation": {
            "noise_sigma": 0.1,
            "baseline_drift": 0.05,
            "scale_jitter": 0.05,
            "time_shift_max": 5,
            "channel_dropout_prob": 0.05,
            "sr_shuffle_prob": 0.1,
        },
    },
}

DANN_CONFIG = {
    "epochs": 30,
    "backbone_lr": 1e-5,   # very low LR for pretrained backbone
    "disc_lr": 1e-3,       # higher LR for discriminator (training from scratch)
    "weight_decay": 1e-4,
    "domain_weight": 0.3,  # gentle domain pressure
    "grad_clip_norm": 1.0,
    "label_smoothing": 0.1,
}


def compute_metrics(y_true, y_pred, class_names=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    if class_names is None:
        class_names = [
            str(l) for l in sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        ]
    per_class = [
        {
            "name": n,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, n in enumerate(class_names)
    ]
    return {"accuracy": acc, "per_class": per_class}


def load_real_data_tensor():
    """Load and preprocess real data, returning (tensor, labels, sources, norm_ds)."""
    real_data = np.load(REAL_DATA_PATH, allow_pickle=True)
    real_signals = real_data["signals"]
    real_labels_str = real_data["labels"]
    real_scan_rates = real_data["scan_rates"]
    real_sources = real_data["sources"] if "sources" in real_data else None

    # Filter to trained mechanisms
    mask = np.array([lbl in MECHANISMS for lbl in real_labels_str])
    r_signals = real_signals[mask]
    r_labels_str = real_labels_str[mask]
    r_scan_rates = real_scan_rates[mask]
    if real_sources is not None:
        real_sources = real_sources[mask]

    print(f"Real data: {len(r_labels_str)} samples — {dict(Counter(r_labels_str.tolist()))}")

    # Channel layout: (N, n_sr, L) -> (N, 18, 250)
    n, n_sr, length = r_signals.shape
    mid = length // 2
    real_tensor = np.empty((n, n_sr * 3, mid), dtype=np.float32)
    for sr_i in range(n_sr):
        real_tensor[:, sr_i * 3, :] = r_signals[:, sr_i, :mid]
        real_tensor[:, sr_i * 3 + 1, :] = r_signals[:, sr_i, mid:]
        real_tensor[:, sr_i * 3 + 2, :] = r_scan_rates[:, sr_i : sr_i + 1]

    # Normalize with training stats
    norm_ds = CVDataset(os.path.join(str(REPO_ROOT), "data", "simulated"), split="train")
    ch_mean = norm_ds.channel_mean[np.newaxis, :, np.newaxis]
    ch_std = norm_ds.channel_std[np.newaxis, :, np.newaxis]
    real_tensor = (real_tensor - ch_mean) / ch_std

    real_y = np.array([MECHANISM_TO_IDX[lbl] for lbl in r_labels_str])
    real_X = torch.from_numpy(real_tensor)

    return real_X, real_y, real_sources, r_labels_str


def eval_model_on_real(model, real_X, real_y, real_sources, label=""):
    """Run single-pass and TTA eval, print results."""
    model.eval()
    real_X_dev = real_X.to(DEVICE)

    with torch.no_grad():
        logits = model(real_X_dev)
        single_preds = logits.argmax(dim=1).cpu().numpy()
    single_acc = (single_preds == real_y).mean()

    tta_probs, tta_preds = tta_predict(model, real_X_dev, n_aug=20, device=DEVICE)
    tta_acc = (tta_preds == real_y).mean()
    tta_metrics = compute_metrics(real_y, tta_preds, class_names=MECHANISMS)

    print(f"\n{'='*60}")
    print(f"{label} Real Accuracy (single): {single_acc:.2%} ({(single_preds==real_y).sum()}/{len(real_y)})")
    print(f"{label} Real Accuracy (TTA):    {tta_acc:.2%} ({(tta_preds==real_y).sum()}/{len(real_y)})")
    print(f"{'='*60}")

    print(f"\n{'Class':<8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 42)
    for entry in tta_metrics["per_class"]:
        print(f"{entry['name']:<8} {entry['precision']:>8.4f} {entry['recall']:>8.4f} "
              f"{entry['f1']:>8.4f} {entry['support']:>8d}")

    # Per-sample table
    print(f"\n{'#':<4} {'True':<8} {'1x':<8} {'TTA':<8} {'Conf':>7}  Source")
    print("-" * 70)
    for i in range(len(real_y)):
        true_name = MECHANISMS[real_y[i]]
        p1 = MECHANISMS[single_preds[i]]
        pt = MECHANISMS[tta_preds[i]]
        conf = tta_probs[i].max()
        flip = " << FLIP" if single_preds[i] != tta_preds[i] else ""
        src = real_sources[i] if real_sources is not None else ""
        print(f"{i:<4} {true_name:<8} {p1:<8} {pt:<8} {conf:>7.4f}  {src}{flip}")

    return single_acc, tta_acc


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Load pretrained backbone ────────────────────────────────────────────
    assert os.path.exists(PRETRAINED_CKPT), f"Missing: {PRETRAINED_CKPT}"
    backbone = get_model("resnet1d_18", in_channels=18, num_classes=N_CLASSES, zero_init_residual=True)
    ckpt = torch.load(PRETRAINED_CKPT, map_location=DEVICE, weights_only=True)
    backbone.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained backbone (epoch {ckpt['epoch']}, val_acc={ckpt['val_accuracy']:.4f})")

    # ── Baseline eval (before DANN) ────────────────────────────────────────
    real_X, real_y, real_sources, real_labels_str = load_real_data_tensor()
    backbone.to(DEVICE)
    print("\n>>> BASELINE (pre-trained ResNet1D, no adaptation)")
    base_single, base_tta = eval_model_on_real(backbone, real_X, real_y, real_sources, label="Baseline")

    # ── Build DANN ──────────────────────────────────────────────────────────
    dann = DANN(backbone, feature_dim=512, num_classes=N_CLASSES).to(DEVICE)
    print(f"\nDANN total params:  {sum(p.numel() for p in dann.parameters()):,}")
    print(f"  Trainable:        {sum(p.numel() for p in dann.parameters() if p.requires_grad):,}")

    # ── Data loaders ────────────────────────────────────────────────────────
    train_dl, val_dl, _ = create_cv_dataloaders(CNN_CONFIG)
    real_X_dev = real_X.to(DEVICE)

    # ── DANN Training (custom loop, separate param groups) ──────────────────
    dcfg = DANN_CONFIG
    domain_criterion = nn.BCEWithLogitsLoss()
    class_criterion = nn.CrossEntropyLoss(label_smoothing=dcfg["label_smoothing"])
    grad_clip = dcfg["grad_clip_norm"]
    domain_weight = dcfg["domain_weight"]
    epochs = dcfg["epochs"]

    # Separate param groups: low LR for backbone, high LR for discriminator
    optimizer = torch.optim.AdamW([
        {"params": dann.backbone.parameters(), "lr": dcfg["backbone_lr"]},
        {"params": dann.discriminator.parameters(), "lr": dcfg["disc_lr"]},
    ], weight_decay=dcfg["weight_decay"])

    print(f"\n{'='*60}")
    print(f"DANN Training: {epochs} epochs")
    print(f"  backbone_lr={dcfg['backbone_lr']}, disc_lr={dcfg['disc_lr']}, domain_weight={domain_weight}")
    print(f"{'='*60}")

    header = f"{'Epoch':>6}  {'ClsLoss':>8}  {'DomLoss':>8}  {'Lambda':>7}  {'ValAcc':>7}  {'RealAcc':>8}"
    print(header)
    print("-" * len(header))

    best_real_acc = 0.0
    best_epoch = 0
    history = {"class_loss": [], "domain_loss": [], "val_acc": [], "real_acc": [], "lambda": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        dann.train()
        epoch_cls, epoch_dom, n_steps = 0.0, 0.0, 0
        lambda_ = schedule_lambda(epoch, epochs)

        for src_x, src_y in train_dl:
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)

            # Sample target batch (oversample real with replacement)
            tgt_idx = torch.randint(0, real_X_dev.size(0), (src_x.size(0),))
            tgt_x = real_X_dev[tgt_idx]

            optimizer.zero_grad()

            # Source: class loss + domain loss
            src_class_logits, src_domain_logits = dann(src_x, lambda_=lambda_)
            cls_loss = class_criterion(src_class_logits, src_y)
            src_dom_loss = domain_criterion(
                src_domain_logits, torch.ones(src_x.size(0), 1, device=DEVICE)
            )

            # Target: domain loss only (no labels)
            _, tgt_domain_logits = dann(tgt_x, lambda_=lambda_)
            tgt_dom_loss = domain_criterion(
                tgt_domain_logits, torch.zeros(tgt_x.size(0), 1, device=DEVICE)
            )

            dom_loss = (src_dom_loss + tgt_dom_loss) / 2
            total_loss = cls_loss + domain_weight * dom_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(dann.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_cls += cls_loss.item()
            epoch_dom += dom_loss.item()
            n_steps += 1

        avg_cls = epoch_cls / max(n_steps, 1)
        avg_dom = epoch_dom / max(n_steps, 1)

        # Val accuracy (sim val set, class head only)
        dann.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_dl:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                class_logits, _ = dann(vx, lambda_=0.0)
                val_correct += (class_logits.argmax(1) == vy).sum().item()
                val_total += vy.size(0)
        val_acc = val_correct / val_total

        # Quick real accuracy (single-pass)
        with torch.no_grad():
            class_logits, _ = dann(real_X_dev, lambda_=0.0)
            real_preds = class_logits.argmax(1).cpu().numpy()
        real_acc = (real_preds == real_y).mean()

        history["class_loss"].append(avg_cls)
        history["domain_loss"].append(avg_dom)
        history["val_acc"].append(val_acc)
        history["real_acc"].append(real_acc)
        history["lambda"].append(lambda_)

        best_flag = ""
        if real_acc > best_real_acc:
            best_real_acc = real_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "dann_state_dict": dann.state_dict(),
                "real_accuracy": float(real_acc),
                "val_accuracy": float(val_acc),
            }, os.path.join(CKPT_DIR, "best_dann.pt"))
            best_flag = " *"

        print(f"{epoch:>6d}  {avg_cls:>8.4f}  {avg_dom:>8.4f}  {lambda_:>7.4f}  "
              f"{val_acc:>6.2%}  {real_acc:>7.2%}{best_flag}")

    elapsed = time.time() - t0
    print("-" * len(header))
    print(f"Best real accuracy: {best_real_acc:.2%} (epoch {best_epoch})")
    print(f"DANN training time: {elapsed:.1f}s")

    # ── Reload best DANN and full eval ──────────────────────────────────────
    best_dann_ckpt = torch.load(
        os.path.join(CKPT_DIR, "best_dann.pt"), map_location=DEVICE, weights_only=True
    )
    dann.load_state_dict(best_dann_ckpt["dann_state_dict"])
    print(f"\nLoaded best DANN checkpoint (epoch {best_dann_ckpt['epoch']})")

    # Eval using the adapted backbone directly
    eval_backbone = dann.backbone
    eval_backbone.to(DEVICE)

    print("\n>>> DANN-ADAPTED MODEL")
    dann_single, dann_tta = eval_model_on_real(eval_backbone, real_X, real_y, real_sources, label="DANN")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Single':>10} {'TTA':>10}")
    print("-" * 47)
    print(f"{'Baseline (no adapt)':<25} {base_single:>9.2%} {base_tta:>9.2%}")
    print(f"{'DANN-adapted':<25} {dann_single:>9.2%} {dann_tta:>9.2%}")
    print(f"{'Improvement':<25} {dann_single-base_single:>+9.2%} {dann_tta-base_tta:>+9.2%}")

    # Save history
    with open(os.path.join(CKPT_DIR, "dann_history.json"), "w") as f:
        json.dump(history, f)
    print(f"\nHistory saved to checkpoints/dann_history.json")

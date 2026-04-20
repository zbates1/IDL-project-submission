"""Standalone ResNet1D-18 training + real-data evaluation with TTA.

Replicates the notebook training loop (Cell 33) and real eval (Cell 40)
as a single script that can be run from the command line.

Usage:
    conda run -n idl python scripts/retrain_and_eval.py
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
)

# Ensure the project root is on sys.path so `src.*` imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.cv_dataset import CVDataset, create_cv_dataloaders
from src.data.tta import tta_predict
from src.models.registry import get_model

# Force import of resnet1d so @register fires
import src.models.resnet1d  # noqa: F401

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR = str(REPO_ROOT / "data" / "simulated")
REAL_DATA_PATH = str(REPO_ROOT / "data" / "real" / "processed" / "real_cvs.npz")
CKPT_DIR = str(REPO_ROOT / "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Derive MECHANISMS from the simulated data ──────────────────────────────
_sim_data = np.load(os.path.join(DATA_DIR, "simulated_cvs.npz"))
_labels_str = _sim_data["labels"]
MECHANISMS = sorted(set(_labels_str.tolist()))
MECHANISM_TO_IDX = {m: i for i, m in enumerate(MECHANISMS)}
print(f"Mechanisms: {MECHANISMS}")

# ── CNN config (matches notebook Cell 30 exactly) ──────────────────────────
CNN_CONFIG = {
    "training": {
        "batch_size": 64,
        "epochs": 30,
        "lr": 1e-4,
        "max_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 10,
        "mixup_alpha": 0.0,
    },
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


# ── Helpers ─────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_metrics(y_true, y_pred, class_names=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    cm = sk_confusion_matrix(y_true, y_pred)
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
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "class_names": class_names,
        "per_class": per_class,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Training
# ═══════════════════════════════════════════════════════════════════════════
def run_training():
    cfg = copy.deepcopy(CNN_CONFIG)
    tcfg = cfg["training"]

    train_dl, val_dl, test_dl = create_cv_dataloaders(cfg)

    N_CLASSES = len(MECHANISMS)
    model = get_model(
        "resnet1d_18", in_channels=18, num_classes=N_CLASSES, zero_init_residual=True
    ).to(DEVICE)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape: (batch, 18, 250)")

    lr = float(tcfg.get("lr", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(tcfg.get("epochs", 200))
    max_lr = float(tcfg.get("max_lr", 3e-3))
    steps_per_epoch = len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=max_lr / lr,
        final_div_factor=1e4,
    )

    label_smoothing = float(tcfg.get("label_smoothing", 0.1))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 1.0))
    patience = int(tcfg.get("early_stopping_patience", 25))
    mixup_alpha = float(tcfg.get("mixup_alpha", 0.2))
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "lr": [],
    }
    best_val_acc = 0.0
    epochs_without_improvement = 0
    ckpt_path = os.path.join(CKPT_DIR, "best_resnet1d.pt")

    header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}"
    print(header)
    print("-" * len(header))

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if mixup_alpha > 0:
                x_mixed, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
            else:
                x_mixed, y_a, y_b, lam = x, y, y, 1.0

            optimizer.zero_grad()
            with torch.amp.autocast(DEVICE.type, enabled=use_amp):
                out = model(x_mixed)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast(DEVICE.type, enabled=use_amp):
                    out = model(x)
                    loss = val_criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        v_loss = val_loss / val_total
        v_acc = val_correct / val_total
        lr_now = optimizer.param_groups[0]["lr"]
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_acc)
        history["lr"].append(lr_now)

        best = v_acc > best_val_acc
        if best:
            best_val_acc = v_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": v_acc,
                    "val_loss": v_loss,
                },
                ckpt_path,
            )
            with open(ckpt_path.replace(".pt", "_history.json"), "w") as hf:
                json.dump(history, hf)
        else:
            epochs_without_improvement += 1

        flag = " *" if best else ""
        print(
            f"{epoch:>6d}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{v_loss:>8.4f}  {v_acc:>6.2%}  {lr_now:>8.2e}{flag}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    elapsed = time.time() - t0
    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    print("-" * len(header))
    print(f"Best val accuracy: {best_val_acc:.2%}  (epoch {best_epoch})")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Checkpoint saved to: {ckpt_path}")

    # Reload best checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    # ── Test set evaluation ───────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    cnn_preds = np.array(all_preds)
    cnn_labels = np.array(all_labels)
    cnn_test_acc = (cnn_preds == cnn_labels).mean()
    cnn_metrics = compute_metrics(cnn_labels, cnn_preds, class_names=MECHANISMS)

    print(f"\n{'='*60}")
    print(f"ResNet1D-18 Simulated Test Accuracy: {cnn_test_acc:.4f}")
    print(f"{'='*60}")
    print(f"\n{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    for entry in cnn_metrics["per_class"]:
        print(
            f"{entry['name']:<8} {entry['precision']:>10.4f} {entry['recall']:>10.4f} "
            f"{entry['f1']:>10.4f} {entry['support']:>10d}"
        )

    return model, history, cnn_test_acc


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Real Data Evaluation with TTA
# ═══════════════════════════════════════════════════════════════════════════
def run_real_eval(model, cnn_test_acc):
    print(f"\n\n{'#'*60}")
    print("# REAL DATA EVALUATION")
    print(f"{'#'*60}\n")

    assert os.path.exists(REAL_DATA_PATH), f"Real data not found: {REAL_DATA_PATH}"

    real_data = np.load(REAL_DATA_PATH, allow_pickle=True)
    real_signals = real_data["signals"]
    real_labels_str = real_data["labels"]
    real_scan_rates = real_data["scan_rates"]
    real_sources = real_data["sources"] if "sources" in real_data else None

    print(f"Real dataset: {len(real_labels_str)} samples")
    print(f"All labels:   {dict(Counter(real_labels_str.tolist()))}")

    # Filter to trained mechanisms (drop CE if present)
    mask = np.array([lbl in MECHANISMS for lbl in real_labels_str])
    r_signals = real_signals[mask]
    r_labels_str = real_labels_str[mask]
    r_scan_rates = real_scan_rates[mask]
    if real_sources is not None:
        real_sources = real_sources[mask]
    dropped = (~mask).sum()
    if dropped > 0:
        excluded = sorted(set(real_labels_str[~mask].tolist()))
        print(f"Excluded {dropped} samples with labels {excluded} (not in training set)")
    print(f"Kept:         {len(r_labels_str)} samples -- {dict(Counter(r_labels_str.tolist()))}")

    # Preprocess: same channel layout as training (N, 6, 500) -> (N, 18, 250)
    n, n_sr, length = r_signals.shape
    mid = length // 2
    real_tensor = np.empty((n, n_sr * 3, mid), dtype=np.float32)
    for sr_i in range(n_sr):
        real_tensor[:, sr_i * 3, :] = r_signals[:, sr_i, :mid]
        real_tensor[:, sr_i * 3 + 1, :] = r_signals[:, sr_i, mid:]
        real_tensor[:, sr_i * 3 + 2, :] = r_scan_rates[:, sr_i : sr_i + 1]

    # Normalize using training set statistics
    _norm_ds = CVDataset(
        os.path.join(str(REPO_ROOT), "data", "simulated"),
        split="train",
    )
    ch_mean = _norm_ds.channel_mean[np.newaxis, :, np.newaxis]
    ch_std = _norm_ds.channel_std[np.newaxis, :, np.newaxis]
    real_tensor = (real_tensor - ch_mean) / ch_std
    print("Applied training-set normalization (z-score current, min-max scan rate)")

    real_X = torch.from_numpy(real_tensor).to(DEVICE)
    real_y = np.array([MECHANISM_TO_IDX[lbl] for lbl in r_labels_str])

    # --- Single-pass inference ---
    model.eval()
    with torch.no_grad():
        logits = model(real_X)
        single_preds = logits.argmax(dim=1).cpu().numpy()

    single_acc = (single_preds == real_y).mean()

    # --- TTA inference ---
    N_TTA = 20
    tta_probs, tta_preds = tta_predict(model, real_X, n_aug=N_TTA, device=DEVICE)
    tta_acc = (tta_preds == real_y).mean()
    tta_metrics = compute_metrics(real_y, tta_preds, class_names=MECHANISMS)

    print(f"\n{'='*60}")
    print(
        f"Real Data Accuracy (single pass): {single_acc:.2%}  "
        f"({(single_preds == real_y).sum()}/{len(real_y)})"
    )
    print(
        f"Real Data Accuracy (TTA, n={N_TTA}):   {tta_acc:.2%}  "
        f"({(tta_preds == real_y).sum()}/{len(real_y)})"
    )
    print(f"{'='*60}")

    print(
        f"\n{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}"
    )
    print("-" * 48)
    for entry in tta_metrics["per_class"]:
        print(
            f"{entry['name']:<8} {entry['precision']:>10.4f} {entry['recall']:>10.4f} "
            f"{entry['f1']:>10.4f} {entry['support']:>8d}"
        )

    print(
        f"\nSim-to-real gap: {cnn_test_acc:.2%} (sim) -> {tta_acc:.2%} (real) = "
        f"delta {cnn_test_acc - tta_acc:+.2%}"
    )

    # Per-sample breakdown
    print(
        f"\n{'#':<4} {'True':<8} {'Pred(1x)':<10} {'Pred(TTA)':<10} "
        f"{'Conf(TTA)':>10}  Source"
    )
    print("-" * 72)
    for i in range(len(real_y)):
        true_name = MECHANISMS[real_y[i]]
        pred_1x = MECHANISMS[single_preds[i]]
        pred_tta = MECHANISMS[tta_preds[i]]
        conf = tta_probs[i].max()
        marker = "" if single_preds[i] == tta_preds[i] else " << FLIPPED"
        source = real_sources[i] if real_sources is not None else ""
        print(
            f"{i:<4} {true_name:<8} {pred_1x:<10} {pred_tta:<10} "
            f"{conf:>10.4f}  {source}{marker}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, history, cnn_test_acc = run_training()
    run_real_eval(model, cnn_test_acc)

    # Final summary
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    final_train_acc = history["train_accuracy"][-1]
    best_val_acc = max(history["val_accuracy"])
    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    print(f"Final train accuracy:  {final_train_acc:.4f}")
    print(f"Best val accuracy:     {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"Simulated test acc:    {cnn_test_acc:.4f}")
    print(f"Total epochs trained:  {len(history['train_loss'])}")

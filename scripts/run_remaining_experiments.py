"""Run remaining experiments for final report: ADDA, few-shot sweep, t-SNE domain.

Generates figures:
  - Report/figures/adda_confusion.png
  - Report/figures/few_shot_sweep.png
  - Report/figures/tsne_domain_before.png
  - Report/figures/tsne_domain_after.png
  - Report/figures/dann_confusion.png
  - Report/figures/method_comparison.png

Usage:
    python -u scripts/run_remaining_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.cv_dataset import CVDataset, create_cv_dataloaders
from src.data.tta import tta_predict
from src.models.registry import get_model
from src.models.dann import DANN, schedule_lambda
from src.models.adda import ADDA, adda_adapt
from src.models.few_shot import few_shot_sweep

import src.models.resnet1d  # noqa: F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = str(REPO_ROOT / "data" / "simulated")
REAL_DATA_PATH = str(REPO_ROOT / "data" / "real" / "processed" / "real_cvs.npz")
CKPT_DIR = str(REPO_ROOT / "checkpoints")
FIG_DIR = str(REPO_ROOT / "Report" / "figures")
PRETRAINED_CKPT = os.path.join(CKPT_DIR, "best_resnet1d.pt")
DANN_CKPT = os.path.join(CKPT_DIR, "best_dann.pt")

os.makedirs(FIG_DIR, exist_ok=True)

# ── Mechanisms ───────────────────────────────────────────────────────────────
_sim_data = np.load(os.path.join(DATA_DIR, "simulated_cvs.npz"))
_labels_str = _sim_data["labels"]
MECHANISMS = sorted(set(_labels_str.tolist()))
MECHANISM_TO_IDX = {m: i for i, m in enumerate(MECHANISMS)}
N_CLASSES = len(MECHANISMS)
print(f"Mechanisms ({N_CLASSES}): {MECHANISMS}")

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


def load_real_data_tensor():
    """Load and preprocess real data."""
    real_data = np.load(REAL_DATA_PATH, allow_pickle=True)
    real_signals = real_data["signals"]
    real_labels_str = real_data["labels"]
    real_scan_rates = real_data["scan_rates"]
    real_sources = real_data["sources"] if "sources" in real_data else None

    mask = np.array([lbl in MECHANISMS for lbl in real_labels_str])
    r_signals = real_signals[mask]
    r_labels_str = real_labels_str[mask]
    r_scan_rates = real_scan_rates[mask]
    if real_sources is not None:
        real_sources = real_sources[mask]

    print(f"Real data: {len(r_labels_str)} samples — {dict(Counter(r_labels_str.tolist()))}")

    n, n_sr, length = r_signals.shape
    mid = length // 2
    real_tensor = np.empty((n, n_sr * 3, mid), dtype=np.float32)
    for sr_i in range(n_sr):
        real_tensor[:, sr_i * 3, :] = r_signals[:, sr_i, :mid]
        real_tensor[:, sr_i * 3 + 1, :] = r_signals[:, sr_i, mid:]
        real_tensor[:, sr_i * 3 + 2, :] = r_scan_rates[:, sr_i : sr_i + 1]

    norm_ds = CVDataset(os.path.join(str(REPO_ROOT), "data", "simulated"), split="train")
    ch_mean = norm_ds.channel_mean[np.newaxis, :, np.newaxis]
    ch_std = norm_ds.channel_std[np.newaxis, :, np.newaxis]
    real_tensor = (real_tensor - ch_mean) / ch_std

    real_y = np.array([MECHANISM_TO_IDX[lbl] for lbl in r_labels_str])
    real_X = torch.from_numpy(real_tensor)
    return real_X, real_y, real_sources, r_labels_str


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save a confusion matrix figure."""
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def extract_features(model, data_tensor, device):
    """Extract 512-dim features from a backbone model."""
    model.eval()
    with torch.no_grad():
        feats = model.get_features(data_tensor.to(device))
    return feats.cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load pretrained backbone ─────────────────────────────────────────
    assert os.path.exists(PRETRAINED_CKPT), f"Missing: {PRETRAINED_CKPT}"
    backbone = get_model("resnet1d_18", in_channels=18, num_classes=N_CLASSES, zero_init_residual=True)
    ckpt = torch.load(PRETRAINED_CKPT, map_location=DEVICE, weights_only=True)
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.to(DEVICE)
    print(f"Loaded pretrained backbone (epoch {ckpt['epoch']})")

    real_X, real_y, real_sources, real_labels_str = load_real_data_tensor()
    train_dl, val_dl, _ = create_cv_dataloaders(CNN_CONFIG)

    # ── 1. BASELINE confusion matrix ────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. BASELINE CONFUSION MATRIX")
    print("=" * 60)
    backbone.eval()
    with torch.no_grad():
        base_logits = backbone(real_X.to(DEVICE))
        base_preds = base_logits.argmax(1).cpu().numpy()
    base_acc = accuracy_score(real_y, base_preds)
    print(f"  Baseline real accuracy: {base_acc:.2%}")
    plot_confusion_matrix(real_y, base_preds, MECHANISMS,
                          f"Baseline ResNet1D (Acc={base_acc:.1%})",
                          os.path.join(FIG_DIR, "baseline_confusion.png"))

    # ── 2. DANN confusion matrix (reload best checkpoint) ───────────────
    print("\n" + "=" * 60)
    print("2. DANN CONFUSION MATRIX")
    print("=" * 60)
    if os.path.exists(DANN_CKPT):
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        dann_ckpt = torch.load(DANN_CKPT, map_location=DEVICE, weights_only=True)

        dann_backbone = get_model("resnet1d_18", in_channels=18, num_classes=N_CLASSES, zero_init_residual=True)
        dann_model = DANN(dann_backbone, feature_dim=512, num_classes=N_CLASSES).to(DEVICE)
        dann_model.load_state_dict(dann_ckpt["dann_state_dict"])
        dann_eval_backbone = dann_model.backbone
        dann_eval_backbone.eval()

        with torch.no_grad():
            dann_logits = dann_eval_backbone(real_X.to(DEVICE))
            dann_preds = dann_logits.argmax(1).cpu().numpy()
        dann_acc = accuracy_score(real_y, dann_preds)
        print(f"  DANN real accuracy: {dann_acc:.2%}")
        plot_confusion_matrix(real_y, dann_preds, MECHANISMS,
                              f"DANN-Adapted (Acc={dann_acc:.1%})",
                              os.path.join(FIG_DIR, "dann_confusion.png"))
    else:
        print("  WARNING: No DANN checkpoint found, skipping.")
        dann_eval_backbone = None
        dann_acc = None

    # ── 3. ADDA training + eval ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. ADDA TRAINING + EVAL")
    print("=" * 60)
    adda_backbone = get_model("resnet1d_18", in_channels=18, num_classes=N_CLASSES, zero_init_residual=True)
    adda_backbone.load_state_dict(ckpt["model_state_dict"])

    adda = ADDA(adda_backbone, feature_dim=512, num_classes=N_CLASSES)
    adda_history = adda_adapt(
        adda, train_dl, real_X,
        epochs=50, lr=1e-4, device=DEVICE, verbose=True,
    )

    # Evaluate ADDA: use target encoder + source classifier
    adda.target_encoder.eval()
    adda.classifier.eval()
    with torch.no_grad():
        tgt_feats = adda.target_encoder.get_features(real_X.to(DEVICE))
        adda_logits = adda.classifier(tgt_feats)
        adda_preds = adda_logits.argmax(1).cpu().numpy()
    adda_acc = accuracy_score(real_y, adda_preds)
    print(f"  ADDA real accuracy: {adda_acc:.2%}")

    plot_confusion_matrix(real_y, adda_preds, MECHANISMS,
                          f"ADDA-Adapted (Acc={adda_acc:.1%})",
                          os.path.join(FIG_DIR, "adda_confusion.png"))

    # Save ADDA checkpoint
    torch.save({
        "target_encoder_state_dict": adda.target_encoder.state_dict(),
        "discriminator_state_dict": adda.discriminator.state_dict(),
        "real_accuracy": float(adda_acc),
    }, os.path.join(CKPT_DIR, "best_adda.pt"))

    # ── 4. FEW-SHOT SWEEP ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. FEW-SHOT FINE-TUNING SWEEP")
    print("=" * 60)
    fs_backbone = get_model("resnet1d_18", in_channels=18, num_classes=N_CLASSES, zero_init_residual=True)
    fs_backbone.load_state_dict(ckpt["model_state_dict"])

    fs_results = few_shot_sweep(
        fs_backbone, real_X, real_y,
        n_values=[1, 2, 3, 5],
        class_names=MECHANISMS,
        num_classes=N_CLASSES,
        feature_dim=512,
        epochs=100,
        lr=1e-3,
        n_trials=5,
        device=DEVICE,
    )

    # Plot few-shot sweep
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [r["n_per_class"] for r in fs_results if r["mean_test_acc"] is not None]
    means = [r["mean_test_acc"] for r in fs_results if r["mean_test_acc"] is not None]
    stds = [r["std_test_acc"] for r in fs_results if r["mean_test_acc"] is not None]

    ax.errorbar(ns, means, yerr=stds, marker="o", capsize=5, linewidth=2,
                markersize=8, color="#2196F3", label="Few-shot (FC head only)")

    # Add horizontal reference lines
    ax.axhline(base_acc, color="#F44336", linestyle="--", linewidth=1.5, label=f"Baseline ({base_acc:.1%})")
    if dann_acc is not None:
        ax.axhline(dann_acc, color="#4CAF50", linestyle="--", linewidth=1.5, label=f"DANN ({dann_acc:.1%})")
    ax.axhline(adda_acc, color="#FF9800", linestyle="--", linewidth=1.5, label=f"ADDA ({adda_acc:.1%})")

    ax.set_xlabel("Labeled examples per class", fontsize=11)
    ax.set_ylabel("Test accuracy on real CVs", fontsize=11)
    ax.set_title("Few-Shot Fine-Tuning vs. Domain Adaptation", fontsize=12, fontweight="bold")
    ax.set_xticks(ns)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "few_shot_sweep.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "few_shot_sweep.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(FIG_DIR, 'few_shot_sweep.png')}")

    # Save few-shot results
    with open(os.path.join(CKPT_DIR, "few_shot_results.json"), "w") as f:
        json.dump(fs_results, f, indent=2)

    # ── 5. t-SNE DOMAIN VISUALIZATION (before/after DANN) ───────────────
    print("\n" + "=" * 60)
    print("5. t-SNE DOMAIN VISUALIZATION")
    print("=" * 60)

    # Sample simulated data for t-SNE (use val set for clean representation)
    sim_X_list, sim_y_list = [], []
    for vx, vy in val_dl:
        sim_X_list.append(vx)
        sim_y_list.append(vy)
    sim_X = torch.cat(sim_X_list)
    sim_y = torch.cat(sim_y_list).numpy()

    # Subsample sim data to ~200 for readable t-SNE
    rng = np.random.default_rng(42)
    n_sim_sample = min(200, len(sim_y))
    sim_idx = rng.choice(len(sim_y), n_sim_sample, replace=False)
    sim_X_sub = sim_X[sim_idx]
    sim_y_sub = sim_y[sim_idx]

    # --- Before DANN (original backbone) ---
    print("  Extracting features (before DANN)...")
    feats_sim_before = extract_features(backbone, sim_X_sub, DEVICE)
    feats_real_before = extract_features(backbone, real_X, DEVICE)

    all_feats_before = np.concatenate([feats_sim_before, feats_real_before])
    all_labels_before = np.concatenate([sim_y_sub, real_y])
    domain_labels = np.array(["Simulated"] * len(sim_y_sub) + ["Real"] * len(real_y))

    print("  Running t-SNE (before DANN)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_feats_before) - 1))
    emb_before = tsne.fit_transform(all_feats_before)

    # Plot: colored by domain (before)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: by domain
    ax = axes[0]
    sim_mask = domain_labels == "Simulated"
    real_mask = domain_labels == "Real"
    ax.scatter(emb_before[sim_mask, 0], emb_before[sim_mask, 1],
               c="#2196F3", alpha=0.4, s=20, label="Simulated")
    ax.scatter(emb_before[real_mask, 0], emb_before[real_mask, 1],
               c="#F44336", alpha=0.9, s=60, marker="*", label="Real")
    ax.set_title("Before DANN — by Domain", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Right: by mechanism class
    ax = axes[1]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for ci, mech in enumerate(MECHANISMS):
        mask_class = all_labels_before == ci
        # sim points
        mask_sim_c = mask_class & sim_mask
        ax.scatter(emb_before[mask_sim_c, 0], emb_before[mask_sim_c, 1],
                   c=colors[ci], alpha=0.3, s=20)
        # real points (starred)
        mask_real_c = mask_class & real_mask
        ax.scatter(emb_before[mask_real_c, 0], emb_before[mask_real_c, 1],
                   c=colors[ci], alpha=0.9, s=80, marker="*", label=f"{mech} (real)")
    ax.set_title("Before DANN — by Mechanism", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("t-SNE of 512-dim Features (Pre-trained Backbone)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tsne_domain_before.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "tsne_domain_before.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: tsne_domain_before.png")

    # --- After DANN ---
    if dann_eval_backbone is not None:
        print("  Extracting features (after DANN)...")
        feats_sim_after = extract_features(dann_eval_backbone, sim_X_sub, DEVICE)
        feats_real_after = extract_features(dann_eval_backbone, real_X, DEVICE)

        all_feats_after = np.concatenate([feats_sim_after, feats_real_after])

        print("  Running t-SNE (after DANN)...")
        tsne2 = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_feats_after) - 1))
        emb_after = tsne2.fit_transform(all_feats_after)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: by domain
        ax = axes[0]
        ax.scatter(emb_after[sim_mask, 0], emb_after[sim_mask, 1],
                   c="#2196F3", alpha=0.4, s=20, label="Simulated")
        ax.scatter(emb_after[real_mask, 0], emb_after[real_mask, 1],
                   c="#F44336", alpha=0.9, s=60, marker="*", label="Real")
        ax.set_title("After DANN — by Domain", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Right: by mechanism class
        ax = axes[1]
        for ci, mech in enumerate(MECHANISMS):
            mask_class = all_labels_before == ci
            mask_sim_c = mask_class & sim_mask
            ax.scatter(emb_after[mask_sim_c, 0], emb_after[mask_sim_c, 1],
                       c=colors[ci], alpha=0.3, s=20)
            mask_real_c = mask_class & real_mask
            ax.scatter(emb_after[mask_real_c, 0], emb_after[mask_real_c, 1],
                       c=colors[ci], alpha=0.9, s=80, marker="*", label=f"{mech} (real)")
        ax.set_title("After DANN — by Mechanism", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks([])
        ax.set_yticks([])

        fig.suptitle("t-SNE of 512-dim Features (DANN-Adapted Backbone)", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "tsne_domain_after.png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(FIG_DIR, "tsne_domain_after.svg"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: tsne_domain_after.png")

    # ── 6. METHOD COMPARISON BAR CHART ──────────────────────────────────
    print("\n" + "=" * 60)
    print("6. METHOD COMPARISON SUMMARY")
    print("=" * 60)

    # Also get TTA for each method
    _, base_tta_preds = tta_predict(backbone, real_X.to(DEVICE), n_aug=20, device=DEVICE)
    base_tta_acc = accuracy_score(real_y, base_tta_preds)

    if dann_eval_backbone is not None:
        _, dann_tta_preds = tta_predict(dann_eval_backbone, real_X.to(DEVICE), n_aug=20, device=DEVICE)
        dann_tta_acc = accuracy_score(real_y, dann_tta_preds)
    else:
        dann_tta_acc = 0

    # ADDA TTA: use target encoder forwarded through classifier
    # Need a wrapper for tta_predict that uses target_encoder + classifier
    class ADDAWrapper(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
        def forward(self, x):
            return self.classifier(self.encoder.get_features(x))

    adda_wrapper = ADDAWrapper(adda.target_encoder, adda.classifier).to(DEVICE)
    _, adda_tta_preds = tta_predict(adda_wrapper, real_X.to(DEVICE), n_aug=20, device=DEVICE)
    adda_tta_acc = accuracy_score(real_y, adda_tta_preds)

    # Best few-shot result
    best_fs = max([r for r in fs_results if r["mean_test_acc"] is not None],
                  key=lambda r: r["mean_test_acc"], default=None)

    methods = ["Baseline", "DANN", "ADDA"]
    single_accs = [base_acc, dann_acc or 0, adda_acc]
    tta_accs = [base_tta_acc, dann_tta_acc, adda_tta_acc]

    if best_fs and best_fs["mean_test_acc"] is not None:
        methods.append(f"Few-shot\n(n={best_fs['n_per_class']})")
        single_accs.append(best_fs["mean_test_acc"])
        tta_accs.append(best_fs["mean_test_acc"])  # no TTA for few-shot

    x_pos = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x_pos - width / 2, single_accs, width, label="Single-pass",
                   color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, tta_accs, width, label="TTA (20-aug)",
                   color="#4CAF50", alpha=0.8)

    ax.set_ylabel("Accuracy on Real CVs", fontsize=11)
    ax.set_title("Method Comparison: Sim-to-Real Transfer", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "method_comparison.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "method_comparison.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: method_comparison.png")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<25} {'Single':>10} {'TTA':>10}")
    print("-" * 47)
    print(f"{'Baseline':<25} {base_acc:>9.2%} {base_tta_acc:>9.2%}")
    if dann_acc is not None:
        print(f"{'DANN':<25} {dann_acc:>9.2%} {dann_tta_acc:>9.2%}")
    print(f"{'ADDA':<25} {adda_acc:>9.2%} {adda_tta_acc:>9.2%}")
    if best_fs and best_fs["mean_test_acc"] is not None:
        print(f"{'Few-shot (n=' + str(best_fs['n_per_class']) + ')':<25} {best_fs['mean_test_acc']:>9.2%} {'—':>10}")
    print("=" * 60)
    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Done!")

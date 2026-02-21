"""
evaluate.py — Comprehensive performance visualization for the Deepfake Detector.

Generates a full evaluation report with:
  1. Training History       — Loss / Accuracy / AUC curves (if history exists)
  2. Confusion Matrix       — TP, FP, TN, FN breakdown
  3. ROC Curve              — AUC-ROC with operating point
  4. Precision-Recall Curve — AUC-PR (better metric for imbalanced data)
  5. Threshold Analysis     — Precision/Recall/F1 vs decision threshold
  6. Per-Class Metrics      — Printed table: Accuracy, AUC, F1, Precision, Recall

Run:
    python src/evaluate.py

Output:
    models/evaluation_report.png   — All plots in a single figure
    Console                        — Metric table + best threshold
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_DIR, MODELS_DIR, MODEL_PATH, MODEL_PATH_H5,
    IMG_SIZE, BATCH_SIZE, HISTORY_PLOT,
)
from src.model import build_model, CUSTOM_OBJECTS

import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, f1_score,
)

REPORT_PATH = MODELS_DIR / "evaluation_report.png"


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    custom_objects = {**CUSTOM_OBJECTS, "tf": tf}
    for target in (MODEL_PATH, MODEL_PATH_H5):
        if target.exists():
            try:
                model = tf.keras.models.load_model(
                    str(target), custom_objects=custom_objects, compile=False
                )
                print(f"  ✓ Loaded model from: {target.name}")
                return model
            except Exception:
                try:
                    model = build_model(trainable_base=False)
                    model.load_weights(str(target))
                    print(f"  ✓ Loaded weights from: {target.name}")
                    return model
                except Exception:
                    continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data Generator (validation set only — no augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def get_validation_data():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )
    gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    return gen


# ─────────────────────────────────────────────────────────────────────────────
# Plot Helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "fake":    "#FF416C",
    "real":    "#11998E",
    "neutral": "#6C63FF",
    "bg":      "#1A1A2E",
    "card":    "#16213E",
    "text":    "#E0E0E0",
    "grid":    "#2A2A4A",
}

def _style_ax(ax, title=""):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    ax.spines[["top","right","left","bottom"]].set_color(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, linestyle="--")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def plot_confusion_matrix(ax, y_true, y_pred_binary):
    cm = confusion_matrix(y_true, y_pred_binary)
    im = ax.imshow(cm, cmap="RdYlGn", aspect="auto")

    labels = ["Real (0)", "Fake (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color=PALETTE["text"])
    ax.set_yticklabels(labels, color=PALETTE["text"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] < cm.max() / 2 else "black")

    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(f"Predicted\nTN={tn}  FP={fp}  FN={fn}  TP={tp}",
                  color=PALETTE["text"])
    ax.set_ylabel("Actual", color=PALETTE["text"])
    _style_ax(ax, "Confusion Matrix")


def plot_roc(ax, y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=PALETTE["neutral"], linewidth=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["grid"], linestyle="--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["neutral"])

    # Mark the operating point closest to top-left corner
    dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_idx = np.argmin(dist)
    ax.scatter(fpr[best_idx], tpr[best_idx], s=120,
               color=PALETTE["fake"], zorder=5,
               label=f"Best threshold ≈ {thresholds[best_idx]:.2f}")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", facecolor=PALETTE["card"],
              labelcolor=PALETTE["text"], framealpha=0.9)
    _style_ax(ax, "ROC Curve")
    return roc_auc


def plot_pr_curve(ax, y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_prec = average_precision_score(y_true, y_scores)

    ax.plot(recall, precision, color=PALETTE["real"], linewidth=2.5,
            label=f"AP = {avg_prec:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color=PALETTE["real"])

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(baseline, linestyle="--", color=PALETTE["grid"],
               linewidth=1, label=f"Baseline = {baseline:.2f}")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="upper right", facecolor=PALETTE["card"],
              labelcolor=PALETTE["text"], framealpha=0.9)
    _style_ax(ax, "Precision-Recall Curve")
    return avg_prec


def plot_threshold_analysis(ax, y_true, y_scores):
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        precisions.append(p); recalls.append(r); f1s.append(f1)

    best_idx = np.argmax(f1s)
    best_t   = thresholds[best_idx]

    ax.plot(thresholds, precisions, color=PALETTE["neutral"],
            linewidth=2, label="Precision")
    ax.plot(thresholds, recalls, color=PALETTE["real"],
            linewidth=2, label="Recall")
    ax.plot(thresholds, f1s, color=PALETTE["fake"],
            linewidth=2.5, label="F1 Score")
    ax.axvline(best_t, linestyle="--", color="white", linewidth=1,
               label=f"Best threshold = {best_t:.2f}  (F1={f1s[best_idx]:.3f})")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.legend(loc="center left", facecolor=PALETTE["card"],
              labelcolor=PALETTE["text"], framealpha=0.9)
    _style_ax(ax, "Threshold Analysis (F1 / Precision / Recall)")
    return best_t, f1s[best_idx]


def plot_score_distribution(ax, y_true, y_scores):
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]

    bins = np.linspace(0, 1, 40)
    ax.hist(real_scores, bins=bins, color=PALETTE["real"],
            alpha=0.7, label=f"Real  (n={len(real_scores)})", edgecolor="none")
    ax.hist(fake_scores, bins=bins, color=PALETTE["fake"],
            alpha=0.7, label=f"Fake  (n={len(fake_scores)})", edgecolor="none")

    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.5,
               label="Default threshold (0.5)")
    ax.set_xlabel("Model Output Score (0=Real, 1=Fake)")
    ax.set_ylabel("Frame Count")
    ax.legend(facecolor=PALETTE["card"],
              labelcolor=PALETTE["text"], framealpha=0.9)
    _style_ax(ax, "Score Distribution (Real vs Fake)")


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():
    print("=" * 60)
    print("  Deepfake Detector — Evaluation Report")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    model = load_model()
    if model is None:
        print("[ERROR] No trained model found. Run train.py first.")
        return

    # ── Validation data ───────────────────────────────────────────────────
    print("[2/4] Loading validation data...")
    val_gen = get_validation_data()
    if val_gen.samples == 0:
        print("[ERROR] No processed data found in data/processed. Run preprocess.py first.")
        return

    print(f"  Samples  : {val_gen.samples}")
    print(f"  Classes  : {val_gen.class_indices}")

    # ── Run predictions ───────────────────────────────────────────────────
    print("[3/4] Running predictions on validation set...")
    y_scores = model.predict(val_gen, verbose=1).flatten()
    y_true   = val_gen.classes
    y_pred   = (y_scores >= 0.5).astype(int)

    # ── Console metrics table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=["Real", "Fake"],
        digits=4,
    ))

    # ── Build figure ──────────────────────────────────────────────────────
    print("[4/4] Generating visualisation...")
    fig = plt.figure(figsize=(22, 14), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Deepfake Detector — Performance Report",
        fontsize=18, fontweight="bold", color=PALETTE["text"], y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])

    # Stats text panel (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(PALETTE["card"])
    ax6.axis("off")

    plot_confusion_matrix(ax1, y_true, y_pred)
    roc_auc   = plot_roc(ax2, y_true, y_scores)
    avg_prec  = plot_pr_curve(ax3, y_true, y_scores)
    best_t, best_f1 = plot_threshold_analysis(ax4, y_true, y_scores)
    plot_score_distribution(ax5, y_true, y_scores)

    # ── Summary stats panel ───────────────────────────────────────────────
    acc    = np.mean(y_pred == y_true)
    f1     = f1_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn + 1e-9)
    fnr_val = fn / (fn + tp + 1e-9)

    stats = [
        ("Accuracy",         f"{acc*100:.2f}%"),
        ("AUC-ROC",          f"{roc_auc:.4f}"),
        ("Avg Precision",    f"{avg_prec:.4f}"),
        ("F1  @ 0.5",        f"{f1:.4f}"),
        ("Best F1",          f"{best_f1:.4f}  (t={best_t:.2f})"),
        ("True Positives",   str(int(tp))),
        ("True Negatives",   str(int(tn))),
        ("False Positives",  str(int(fp))),
        ("False Negatives",  str(int(fn))),
        ("False Pos Rate",   f"{fpr_val*100:.1f}%"),
        ("False Neg Rate",   f"{fnr_val*100:.1f}%"),
    ]

    ax6.text(0.05, 0.97, "📊  Summary Statistics",
             transform=ax6.transAxes,
             fontsize=12, fontweight="bold",
             color=PALETTE["text"], va="top")

    for i, (label, value) in enumerate(stats):
        y_pos = 0.85 - i * 0.075
        ax6.text(0.05, y_pos, label + ":", transform=ax6.transAxes,
                 fontsize=10, color="#AAAAAA", va="top")
        color = PALETTE["neutral"]
        if "False" in label:
            color = PALETTE["fake"]
        elif "True" in label or "Accuracy" in label or "AUC" in label:
            color = PALETTE["real"]
        ax6.text(0.65, y_pos, value, transform=ax6.transAxes,
                 fontsize=10, fontweight="bold", color=color, va="top")

    plt.savefig(str(REPORT_PATH), dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()

    print(f"\n✅ Report saved to: {REPORT_PATH}")
    print(f"\n  AUC-ROC       : {roc_auc:.4f}")
    print(f"  Avg Precision : {avg_prec:.4f}")
    print(f"  Best F1       : {best_f1:.4f}  (threshold = {best_t:.2f})")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()

"""
evaluate.py -- Comprehensive evaluation for the DeepGuard AI model.

Generates:
  1. Confusion Matrix
  2. ROC Curve with AUC
  3. Precision-Recall Curve
  4. Threshold Analysis (F1/Precision/Recall vs threshold)
  5. Score Distribution histogram
  6. Summary statistics panel

Uses the TEST split (completely held-out data) for unbiased evaluation.

Run:
    python src/evaluate.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_DIR, MODELS_DIR, MODEL_PATH, MODEL_PATH_H5,
    IMG_SIZE, BATCH_SIZE, HISTORY_PLOT,
    TEST_DIR, VALID_DIR,
)
from src.model import build_model, CUSTOM_OBJECTS

import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, f1_score,
)

REPORT_PATH = MODELS_DIR / "evaluation_report.png"


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

def load_model():
    custom_objects = {**CUSTOM_OBJECTS, "tf": tf}
    # Check all possible model file locations
    candidates = [MODEL_PATH, MODELS_DIR / "deepfake_detector.h5", MODEL_PATH_H5]
    for target in candidates:
        if target.exists():
            try:
                model = tf.keras.models.load_model(
                    str(target), custom_objects=custom_objects, compile=False
                )
                print(f"  [OK] Loaded model from: {target.name}")
                return model
            except Exception:
                try:
                    model = build_model(trainable_base=False)
                    model.load_weights(str(target))
                    print(f"  [OK] Loaded weights from: {target.name}")
                    return model
                except Exception:
                    continue
    return None


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def get_evaluation_data():
    """Loads the test set (or validation set as fallback)."""
    # Prefer test set for unbiased evaluation
    for eval_dir in [TEST_DIR, VALID_DIR]:
        if eval_dir.exists() and any(eval_dir.iterdir()):
            print(f"  Using evaluation data from: {eval_dir}")
            ds = tf.keras.utils.image_dataset_from_directory(
                eval_dir,
                seed=42,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode="binary",
                shuffle=False,
            )
            return ds, ds.file_paths, ds.class_names

    # Fallback: use validation split from DATA_DIR
    print(f"  Fallback: using validation split from {DATA_DIR}")
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    return ds, ds.file_paths, ds.class_names


# -----------------------------------------------------------------------------
# Plot Helpers
# -----------------------------------------------------------------------------

PALETTE = {
    "fake":    "#FF416C",
    "real":    "#11998E",
    "neutral": "#6C63FF",
    "bg":      "#0A0A1A",
    "card":    "#12122A",
    "text":    "#E0E0E0",
    "grid":    "#2A2A4A",
}


def _style_ax(ax, title=""):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    ax.spines[["top", "right", "left", "bottom"]].set_color(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, linestyle="--")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def plot_confusion_matrix(ax, y_true, y_pred_binary, class_names):
    cm = confusion_matrix(y_true, y_pred_binary)
    im = ax.imshow(cm, cmap="RdYlGn", aspect="auto")

    labels = [f"{name} (0)" if i == 0 else f"{name} (1)" for i, name in enumerate(class_names)]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color=PALETTE["text"])
    ax.set_yticklabels(labels, color=PALETTE["text"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] < cm.max() / 2 else "black")

    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(f"Predicted\nTN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}",
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

    dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_idx = np.argmin(dist)
    ax.scatter(fpr[best_idx], tpr[best_idx], s=120,
               color=PALETTE["fake"], zorder=5,
               label=f"Optimal threshold ≈ {thresholds[best_idx]:.3f}")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", facecolor=PALETTE["card"],
              labelcolor=PALETTE["text"], framealpha=0.9)
    _style_ax(ax, "ROC Curve")
    return roc_auc


def plot_pr_curve(ax, y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_prec = average_precision_score(y_true, y_scores)

    ax.plot(recall, precision, color=PALETTE["real"], linewidth=2.5,
            label=f"AP = {avg_prec:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color=PALETTE["real"])

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

    ax.plot(thresholds, precisions, color=PALETTE["neutral"], linewidth=2, label="Precision")
    ax.plot(thresholds, recalls, color=PALETTE["real"], linewidth=2, label="Recall")
    ax.plot(thresholds, f1s, color=PALETTE["fake"], linewidth=2.5, label="F1 Score")
    ax.axvline(best_t, linestyle="--", color="white", linewidth=1,
               label=f"Best t={best_t:.3f} (F1={f1s[best_idx]:.4f})")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.legend(loc="center left", facecolor=PALETTE["card"],
              labelcolor=PALETTE["text"], framealpha=0.9, fontsize=9)
    _style_ax(ax, "Threshold Analysis")
    return best_t, f1s[best_idx]


def plot_score_distribution(ax, y_true, y_scores, class_names):
    bins = np.linspace(0, 1, 50)
    ax.hist(y_scores[y_true == 0], bins=bins, color=PALETTE["real"],
            alpha=0.7, label=f"{class_names[0]} (n={np.sum(y_true==0):,})", edgecolor="none")
    ax.hist(y_scores[y_true == 1], bins=bins, color=PALETTE["fake"],
            alpha=0.7, label=f"{class_names[1]} (n={np.sum(y_true==1):,})", edgecolor="none")

    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.5,
               label="Default threshold (0.5)")
    ax.set_xlabel("Model Output Score")
    ax.set_ylabel("Count")
    ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"], framealpha=0.9)
    _style_ax(ax, "Score Distribution")


# -----------------------------------------------------------------------------
# Main Evaluation
# -----------------------------------------------------------------------------

def evaluate():
    print("=" * 70)
    print("  DeepGuard AI -- Evaluation Report")
    print("=" * 70)

    # -- Load model --------------------------------------------------------
    print("\n[1/4] Loading model...")
    model = load_model()
    if model is None:
        print("[ERROR] No trained model found. Run train.py first.")
        return

    # -- Test data ---------------------------------------------------------
    print("\n[2/4] Loading evaluation data...")
    eval_ds, file_paths, class_names = get_evaluation_data()

    print(f"  Samples     : {len(file_paths):,}")
    print(f"  Classes     : {class_names}")

    y_true = np.concatenate([y.numpy() for _, y in eval_ds], axis=0).flatten()

    # -- Run predictions ---------------------------------------------------
    print("\n[3/4] Running predictions...")
    y_scores = model.predict(eval_ds, verbose=1).flatten()
    y_pred   = (y_scores >= 0.5).astype(int)

    # -- Console report ----------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  CLASSIFICATION REPORT (threshold = 0.5)")
    print(f"{'=' * 70}")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    ))

    # -- Build figure ------------------------------------------------------
    print("[4/4] Generating visualisation...")
    fig = plt.figure(figsize=(24, 14), facecolor=PALETTE["bg"])
    fig.suptitle(
        "DeepGuard AI -- Performance Report",
        fontsize=20, fontweight="bold", color=PALETTE["text"], y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(PALETTE["card"])
    ax6.axis("off")

    plot_confusion_matrix(ax1, y_true, y_pred, class_names)
    roc_auc       = plot_roc(ax2, y_true, y_scores)
    avg_prec      = plot_pr_curve(ax3, y_true, y_scores)
    best_t, best_f1 = plot_threshold_analysis(ax4, y_true, y_scores)
    plot_score_distribution(ax5, y_true, y_scores, class_names)

    # -- Summary stats panel -----------------------------------------------
    acc     = np.mean(y_pred == y_true)
    f1      = f1_score(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn + 1e-9)
    fnr_val = fn / (fn + tp + 1e-9)

    stats = [
        ("Accuracy",       f"{acc*100:.2f}%"),
        ("AUC-ROC",        f"{roc_auc:.4f}"),
        ("Avg Precision",  f"{avg_prec:.4f}"),
        ("F1  @ 0.5",      f"{f1:.4f}"),
        ("Best F1",        f"{best_f1:.4f}  (t={best_t:.3f})"),
        ("True Positives", f"{int(tp):,}"),
        ("True Negatives", f"{int(tn):,}"),
        ("False Positives", f"{int(fp):,}"),
        ("False Negatives", f"{int(fn):,}"),
        ("False Pos Rate", f"{fpr_val*100:.2f}%"),
        ("False Neg Rate", f"{fnr_val*100:.2f}%"),
    ]

    ax6.text(0.05, 0.97, "Summary Statistics",
             transform=ax6.transAxes,
             fontsize=13, fontweight="bold",
             color=PALETTE["text"], va="top")

    for i, (label, value) in enumerate(stats):
        y_pos = 0.87 - i * 0.072
        ax6.text(0.05, y_pos, label + ":", transform=ax6.transAxes,
                 fontsize=10.5, color="#AAAAAA", va="top")
        color = PALETTE["neutral"]
        if "False" in label:
            color = PALETTE["fake"]
        elif "True" in label or "Accuracy" in label or "AUC" in label:
            color = PALETTE["real"]
        ax6.text(0.60, y_pos, value, transform=ax6.transAxes,
                 fontsize=10.5, fontweight="bold", color=color, va="top")

    plt.savefig(str(REPORT_PATH), dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"  FINAL METRICS")
    print(f"{'=' * 70}")
    print(f"  Accuracy       : {acc*100:.2f}%")
    print(f"  AUC-ROC        : {roc_auc:.4f}")
    print(f"  Avg Precision  : {avg_prec:.4f}")
    print(f"  F1 @ 0.5       : {f1:.4f}")
    print(f"  Best F1        : {best_f1:.4f}  (threshold = {best_t:.3f})")
    print(f"\n  Report saved to: {REPORT_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    evaluate()

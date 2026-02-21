"""
train.py — Optimized 2-phase training pipeline for Deepfake Detection.

Phase 1: Train custom head only (base frozen) — 20 epochs, LR=1e-4
Phase 2: Fine-tune top 30 Xception layers — 20 epochs, LR=1e-5

Key improvements over original:
  - Correct Xception preprocess_input() normalization (not /255.0)
  - 2-phase training with proper fine-tuning
  - EarlyStopping, ReduceLROnPlateau, ModelCheckpoint callbacks
  - AUC, Precision, Recall metrics
  - Automatic class-weight balancing
  - Training history plot saved to models/
  - Model saved in modern .keras format
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)

# ── Path setup so we can import from src/ ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    IMG_SIZE, BATCH_SIZE,
    EPOCHS_HEAD, EPOCHS_FINETUNE,
    LR_HEAD, LR_FINETUNE, FINETUNE_LAYERS,
    DATA_DIR, MODELS_DIR, MODEL_PATH, HISTORY_PLOT,
)
from src.model import build_model, unfreeze_top_layers
from tensorflow.keras.applications.xception import preprocess_input

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def cutout(img: np.ndarray) -> np.ndarray:
    """
    Cutout augmentation: randomly masks a square region with the channel mean.
    Applied BEFORE preprocess_input, so img values are in [0, 255].
    """
    h, w, _ = img.shape
    mask_size = max(16, h // 4)  # Adaptive mask: 25% of image height

    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)

    y1 = np.clip(cy - mask_size // 2, 0, h)
    y2 = np.clip(cy + mask_size // 2, 0, h)
    x1 = np.clip(cx - mask_size // 2, 0, w)
    x2 = np.clip(cx + mask_size // 2, 0, w)

    img[y1:y2, x1:x2, :] = np.mean(img)  # Fill with image mean (neutral)
    return img


def combined_preprocessing(img: np.ndarray) -> np.ndarray:
    """Applies Cutout then Xception's preprocess_input in one step."""
    img = cutout(img)
    img = preprocess_input(img)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────────────────────────────────────

def make_generators():
    """Creates train and validation ImageDataGenerators."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=combined_preprocessing,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
    )

    # Validation: only normalization, no augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(train_gen):
    """Computes balanced class weights to handle imbalanced datasets."""
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    cw = dict(zip(classes, weights))
    print(f"\n  Class weights: {cw}  "
          f"(fake={cw.get(0,'?'):.3f}, real={cw.get(1,'?'):.3f})")
    return cw


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def make_callbacks(model_path: Path, monitor: str = "val_auc", phase: int = 1):
    """Returns standard callback set for a training phase."""
    return [
        EarlyStopping(
            monitor=monitor,
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor,
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# History Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(histories: list, save_path: Path):
    """
    Plots and saves training curves (loss, accuracy, AUC) across all phases.

    Args:
        histories: List of Keras History objects (one per training phase).
        save_path: Path to save the PNG plot.
    """
    # Merge histories across phases
    merged = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(values)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Deepfake Detector — Training History", fontsize=14, fontweight="bold")

    metrics = [
        ("loss", "val_loss", "Loss", axes[0]),
        ("accuracy", "val_accuracy", "Accuracy", axes[1]),
        ("auc", "val_auc", "AUC", axes[2]),
    ]

    for train_key, val_key, title, ax in metrics:
        if train_key in merged:
            ax.plot(merged[train_key], label="Train", linewidth=2)
        if val_key in merged:
            ax.plot(merged[val_key], label="Val", linewidth=2, linestyle="--")

        # Draw vertical line at phase boundary
        phase1_len = len(histories[0].history.get(train_key, []))
        if phase1_len > 0 and len(histories) > 1:
            ax.axvline(x=phase1_len - 1, color="gray", linestyle=":", label="Fine-tune starts")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Training history saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def train():
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        print("  Run `python src/preprocess.py` first.")
        return

    print("=" * 60)
    print("  Deepfake Detector — Training Pipeline")
    print("=" * 60)
    print(f"  IMG_SIZE        : {IMG_SIZE}")
    print(f"  BATCH_SIZE      : {BATCH_SIZE}")
    print(f"  Phase 1 epochs  : {EPOCHS_HEAD}  (LR={LR_HEAD})")
    print(f"  Phase 2 epochs  : {EPOCHS_FINETUNE}  (LR={LR_FINETUNE})")
    print(f"  Unfreeze layers : top {FINETUNE_LAYERS} Xception layers")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Setting up data generators...")
    train_gen, val_gen = make_generators()

    if train_gen.samples == 0:
        print("[ERROR] No training data found in data/processed. "
              "Run preprocess.py first.")
        return

    print(f"  Train samples : {train_gen.samples}")
    print(f"  Val samples   : {val_gen.samples}")
    print(f"  Classes       : {train_gen.class_indices}")

    class_weights = get_class_weights(train_gen)

    # ── Build Model ───────────────────────────────────────────────────────
    print("\n[2/5] Building model (base frozen)...")
    model = build_model(trainable_base=False)
    model.summary(line_length=100)

    # ── Phase 1: Train Head ───────────────────────────────────────────────
    print(f"\n[3/5] Phase 1 — Training head ({EPOCHS_HEAD} epochs, LR={LR_HEAD})...")
    callbacks_p1 = make_callbacks(MODEL_PATH, phase=1)

    history_p1 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1,
    )

    print("\n  ✓ Phase 1 complete.")

    # ── Phase 2: Fine-tune Top Layers ─────────────────────────────────────
    print(f"\n[4/5] Phase 2 — Fine-tuning top {FINETUNE_LAYERS} layers "
          f"({EPOCHS_FINETUNE} epochs, LR={LR_FINETUNE})...")

    model = unfreeze_top_layers(model, n_layers=FINETUNE_LAYERS, new_lr=LR_FINETUNE)
    callbacks_p2 = make_callbacks(MODEL_PATH, phase=2)

    history_p2 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1,
    )

    print("\n  ✓ Phase 2 complete.")

    # ── Save & Plot ────────────────────────────────────────────────────────
    print("\n[5/5] Saving model and training history...")
    model.save(str(MODEL_PATH))
    print(f"  ✓ Model saved to: {MODEL_PATH}")

    plot_history([history_p1, history_p2], HISTORY_PLOT)

    # Final metrics summary
    best_val_auc = max(
        max(history_p1.history.get("val_auc", [0])),
        max(history_p2.history.get("val_auc", [0])),
    )
    print(f"\n🏆 Best Val AUC   : {best_val_auc:.4f}")
    print("=" * 60)
    print("  Training complete! Run `streamlit run app.py` to test.")
    print("=" * 60)


if __name__ == "__main__":
    train()

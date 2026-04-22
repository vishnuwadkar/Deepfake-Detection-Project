"""
train.py -- Optimized training pipeline for DeepGuard AI.

Target: 95%+ accuracy on AI-generated face detection.

Key optimizations:
  - 140K Real-and-Fake-Faces dataset (StyleGAN-generated, 256×256)
  - Two-phase training: frozen backbone → fine-tune top 50 layers
  - Mixup augmentation for robust generalization
  - Label smoothing (0.1) to prevent overconfidence
  - Cosine decay learning rate schedule
  - Comprehensive GPU-accelerated augmentation pipeline
  - Class weight balancing (auto-computed)
  - Best-model checkpointing on val_auc

Run:
    python src/train.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard,
)

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    IMG_SIZE, BATCH_SIZE,
    EPOCHS_HEAD, EPOCHS_FINETUNE,
    LR_HEAD, LR_FINETUNE, FINETUNE_LAYERS,
    MODELS_DIR, MODEL_PATH, HISTORY_PLOT,
    TRAIN_DIR, VALID_DIR, DATA_DIR,
    MIXUP_ALPHA, MAX_TRAIN_SAMPLES,
)
from src.model import build_model, unfreeze_top_layers

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Augmentation Pipeline (GPU accelerated)
# -----------------------------------------------------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
], name="augmentation")


# -----------------------------------------------------------------------------
# Mixup Augmentation
# -----------------------------------------------------------------------------

def mixup(images, labels, alpha=MIXUP_ALPHA):
    """
    Applies Mixup augmentation: blends pairs of images and labels.
    This is one of the most effective regularization techniques for CNNs.
    """
    if alpha <= 0:
        return images, labels

    batch_size = tf.shape(images)[0]
    # Sample lambda from Beta distribution
    lam = tf.random.uniform([], 0, alpha)

    # Shuffle indices
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    # Blend
    images = lam * images + (1 - lam) * shuffled_images
    labels = lam * labels + (1 - lam) * shuffled_labels

    return images, labels


# -----------------------------------------------------------------------------
# tf.data Pipeline
# -----------------------------------------------------------------------------

def make_datasets():
    """
    Creates train and validation tf.data.Datasets.
    
    The 140K dataset has pre-built splits:
      train/ (real + fake)  → 100K images
      valid/ (real + fake)  → 20K images
      test/  (real + fake)  → 20K images
      
    If the dataset uses the pre-split structure, we use those.
    Otherwise, we fall back to automatic splitting from DATA_DIR.
    """
    # Check if pre-split directories exist
    if TRAIN_DIR.exists() and VALID_DIR.exists():
        print(f"  Using pre-split dataset:")
        print(f"    Train: {TRAIN_DIR}")
        print(f"    Valid: {VALID_DIR}")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=True,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            VALID_DIR,
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False,
        )
    else:
        # Fallback: split from single directory
        print(f"  Using auto-split from: {DATA_DIR}")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
        )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE

    # Subsample training data for CPU speed
    if MAX_TRAIN_SAMPLES is not None:
        steps_per_epoch = MAX_TRAIN_SAMPLES // BATCH_SIZE
        train_ds = train_ds.take(steps_per_epoch)
        print(f"  Subsampled to {MAX_TRAIN_SAMPLES:,} images ({steps_per_epoch} steps/epoch)")

    # Apply augmentation + mixup to training data
    def augment_and_mixup(images, labels):
        images = data_augmentation(images, training=True)
        images, labels = mixup(images, labels)
        return images, labels

    train_ds = (
        train_ds
        .cache()  # Cache decoded images — avoids re-reading JPEGs every epoch
        .map(augment_and_mixup, num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


# -----------------------------------------------------------------------------
# Class Weights (auto-computed)
# -----------------------------------------------------------------------------

def get_class_weights():
    """Computes balanced class weights from directory counts."""
    # Check both possible structures
    for base_dir in [TRAIN_DIR, DATA_DIR]:
        if not base_dir.exists():
            continue

        counts = {}
        for class_dir in sorted(base_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.*")))
                counts[class_dir.name] = count

        if counts:
            total = sum(counts.values())
            # Alphabetical order = class index
            sorted_names = sorted(counts.keys())
            weights = {}
            for i, name in enumerate(sorted_names):
                weights[i] = total / (len(counts) * counts[name]) if counts[name] > 0 else 1.0

            print(f"\n  Class counts: {counts}")
            print(f"  Class weights: {weights}")
            return weights

    print("  [WARN] Could not compute class weights, using uniform")
    return {0: 1.0, 1: 1.0}


# -----------------------------------------------------------------------------
# Cosine Decay Learning Rate Schedule
# -----------------------------------------------------------------------------

def cosine_decay_schedule(epoch, lr, total_epochs, base_lr, min_lr=1e-7):
    """Cosine annealing with warm restarts."""
    progress = epoch / max(total_epochs, 1)
    new_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    return float(new_lr)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

def make_callbacks(model_path: Path, phase: int = 1, total_epochs: int = 15, base_lr: float = 3e-4):
    """Returns callback set for a training phase."""
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=7 if phase == 2 else 5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path).replace('.keras', '.h5'),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
            save_format='h5',
        ),
        LearningRateScheduler(
            lambda epoch, lr: cosine_decay_schedule(epoch, lr, total_epochs, base_lr),
            verbose=0,
        ),
    ]
    return callbacks


# -----------------------------------------------------------------------------
# History Plotting
# -----------------------------------------------------------------------------

def plot_history(histories: list, save_path: Path):
    """Plots training curves across all phases."""
    merged = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(values)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("DeepGuard AI -- Training History", fontsize=14, fontweight="bold")

    metrics = [
        ("loss", "val_loss", "Loss", axes[0]),
        ("accuracy", "val_accuracy", "Accuracy", axes[1]),
        ("auc", "val_auc", "AUC-ROC", axes[2]),
        ("precision", "val_precision", "Precision", axes[3]),
    ]

    for train_key, val_key, title, ax in metrics:
        if train_key in merged:
            ax.plot(merged[train_key], label="Train", linewidth=2)
        if val_key in merged:
            ax.plot(merged[val_key], label="Val", linewidth=2, linestyle="--")

        phase1_len = len(histories[0].history.get(train_key, []))
        if phase1_len > 0 and len(histories) > 1:
            ax.axvline(x=phase1_len - 1, color="gray", linestyle=":", label="Fine-tune start")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [INFO] Training history saved to: {save_path}")


# -----------------------------------------------------------------------------
# Main Training Entry Point
# -----------------------------------------------------------------------------

def train():
    # Verify data exists
    data_found = False
    for d in [TRAIN_DIR, DATA_DIR]:
        if d.exists() and any(d.iterdir()):
            data_found = True
            break

    if not data_found:
        print(f"[ERROR] No data found. Run download_data.py first.")
        print(f"  Checked: {TRAIN_DIR}")
        print(f"  Checked: {DATA_DIR}")
        return

    print("=" * 70)
    print("  DeepGuard AI -- Optimized Training Pipeline")
    print("=" * 70)
    print(f"  Target          : 95%+ accuracy on AI-generated face detection")
    print(f"  IMG_SIZE        : {IMG_SIZE}")
    print(f"  BATCH_SIZE      : {BATCH_SIZE}")
    print(f"  Phase 1 epochs  : {EPOCHS_HEAD}  (LR={LR_HEAD})")
    print(f"  Phase 2 epochs  : {EPOCHS_FINETUNE}  (LR={LR_FINETUNE})")
    print(f"  Unfreeze layers : top {FINETUNE_LAYERS}")
    print(f"  Label smoothing : {MIXUP_ALPHA}")
    print(f"  Mixup alpha     : {MIXUP_ALPHA}")
    print("=" * 70)

    # -- Data -------------------------------------------------------------
    print("\n[1/5] Setting up tf.data pipeline...")
    train_ds, val_ds, class_names = make_datasets()
    print(f"  Classes: {class_names}")

    class_weights = get_class_weights()

    # -- Build Model -------------------------------------------------------
    print("\n[2/5] Building EfficientNetV2B0 + CBAM model (base frozen)...")
    model = build_model(trainable_base=False)
    model.summary(line_length=110, print_fn=lambda x: print(f"  {x}"))

    # -- Phase 1: Train Head -----------------------------------------------
    print(f"\n[3/5] Phase 1 -- Training classification head ({EPOCHS_HEAD} epochs)...")
    callbacks_p1 = make_callbacks(MODEL_PATH, phase=1, total_epochs=EPOCHS_HEAD, base_lr=LR_HEAD)

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1,
    )

    best_p1_auc = max(history_p1.history.get("val_auc", [0]))
    best_p1_acc = max(history_p1.history.get("val_accuracy", [0]))
    print(f"\n  [OK] Phase 1 complete -- Best val_auc: {best_p1_auc:.4f}, val_acc: {best_p1_acc:.4f}")

    # Save Phase 1 backup (in case Phase 2 crashes / runtime disconnects)
    phase1_backup = MODELS_DIR / "deepfake_detector_phase1.h5"
    model.save(str(phase1_backup))
    print(f"  [OK] Phase 1 backup saved to: {phase1_backup}")

    # -- Phase 2: Fine-tune Top Layers -------------------------------------
    print(f"\n[4/5] Phase 2 -- Fine-tuning top {FINETUNE_LAYERS} layers ({EPOCHS_FINETUNE} epochs)...")

    model = unfreeze_top_layers(model, n_layers=FINETUNE_LAYERS, new_lr=LR_FINETUNE)
    callbacks_p2 = make_callbacks(MODEL_PATH, phase=2, total_epochs=EPOCHS_FINETUNE, base_lr=LR_FINETUNE)

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1,
    )

    best_p2_auc = max(history_p2.history.get("val_auc", [0]))
    best_p2_acc = max(history_p2.history.get("val_accuracy", [0]))
    print(f"\n  [OK] Phase 2 complete -- Best val_auc: {best_p2_auc:.4f}, val_acc: {best_p2_acc:.4f}")

    # -- Save & Plot --------------------------------------------------------
    print("\n[5/5] Saving model and training history...")
    save_path = str(MODEL_PATH).replace('.keras', '.h5')
    model.save(save_path)
    print(f"  [OK] Model saved to: {save_path}")

    plot_history([history_p1, history_p2], HISTORY_PLOT)

    # Final metrics summary
    best_auc = max(best_p1_auc, best_p2_auc)
    best_acc = max(best_p1_acc, best_p2_acc)

    print(f"\n{'=' * 70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Best Val Accuracy : {best_acc*100:.2f}%")
    print(f"  Best Val AUC-ROC  : {best_auc:.4f}")
    print(f"  Model saved to    : {MODEL_PATH}")
    print(f"  History plot      : {HISTORY_PLOT}")
    print(f"{'=' * 70}")
    print(f"  Next steps:")
    print(f"    1. Run evaluation : python src/evaluate.py")
    print(f"    2. Convert to TF.js : python scripts/convert_model.py")
    print(f"    3. Load extension in Chrome")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    train()

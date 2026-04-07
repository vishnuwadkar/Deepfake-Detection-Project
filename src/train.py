import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
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

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation layers (GPU accelerated)
# ─────────────────────────────────────────────────────────────────────────────

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2),
])

# ─────────────────────────────────────────────────────────────────────────────
# tf.data Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def make_datasets():
    """Creates train and validation tf.data.Dataset."""
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    
    # Store class names before prefetching
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    # Apply augmentation to training data
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # Cache and prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights():
    """Computes balanced class weights directly from directory counts."""
    from src.config import REAL_PROCESSED, FAKE_PROCESSED
    
    real_count = len(list(REAL_PROCESSED.glob('*.png')))
    fake_count = len(list(FAKE_PROCESSED.glob('*.png')))
    total = real_count + fake_count
    
    # Class 0: fake, Class 1: real (Keras standard alphanumeric sorting)
    # weights = total / (2.0 * count)
    weight_0 = total / (2.0 * fake_count) if fake_count > 0 else 1.0
    weight_1 = total / (2.0 * real_count) if real_count > 0 else 1.0
    
    cw = {0: weight_0, 1: weight_1}
    print(f"\n  Class weights: {cw}  "
          f"(fake={weight_0:.3f}, real={weight_1:.3f})")
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
    print(f"\n  [INFO] Training history saved to: {save_path}")


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
    print("\n[1/5] Setting up tf.data pipeline...")
    train_ds, val_ds, class_names = make_datasets()

    print(f"  Classes       : {class_names}")

    class_weights = get_class_weights()

    # ── Build Model ───────────────────────────────────────────────────────
    print("\n[2/5] Building model (base frozen)...")
    model = build_model(trainable_base=False)
    model.summary(line_length=100)

    # ── Phase 1: Train Head ───────────────────────────────────────────────
    print(f"\n[3/5] Phase 1 — Training head ({EPOCHS_HEAD} epochs, LR={LR_HEAD})...")
    callbacks_p1 = make_callbacks(MODEL_PATH, phase=1)

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        verbose=1,
    )

    print("\n  [OK] Phase 1 complete.")

    # ── Phase 2: Fine-tune Top Layers ─────────────────────────────────────
    print(f"\n[4/5] Phase 2 — Fine-tuning top {FINETUNE_LAYERS} layers "
          f"({EPOCHS_FINETUNE} epochs, LR={LR_FINETUNE})...")

    model = unfreeze_top_layers(model, n_layers=FINETUNE_LAYERS, new_lr=LR_FINETUNE)
    callbacks_p2 = make_callbacks(MODEL_PATH, phase=2)

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=callbacks_p2,
        verbose=1,
    )

    print("\n  [OK] Phase 2 complete.")

    # ── Save & Plot ────────────────────────────────────────────────────────
    print("\n[5/5] Saving model and training history...")
    model.save(str(MODEL_PATH))
    print(f"  [OK] Model saved to: {MODEL_PATH}")

    plot_history([history_p1, history_p2], HISTORY_PLOT)

    # Final metrics summary
    best_val_auc = max(
        max(history_p1.history.get("val_auc", [0])),
        max(history_p2.history.get("val_auc", [0])),
    )
    print(f"\n[INFO] Best Val AUC   : {best_val_auc:.4f}")
    print("=" * 60)
    print("  Training complete! Run `streamlit run app.py` to test.")
    print("=" * 60)


if __name__ == "__main__":
    train()

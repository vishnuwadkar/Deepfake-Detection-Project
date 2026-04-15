"""
config.py — Centralized configuration for the DeepGuard AI project.
All scripts import from here; change once, applies everywhere.

Optimized for 140K Real-and-Fake-Faces dataset (StyleGAN-generated faces).
"""

from pathlib import Path

# ─── Image / Model ───────────────────────────────────────────────
IMG_SIZE = (224, 224)          # EfficientNetV2B0 optimal input
BATCH_SIZE = 64                # Larger batch -> fewer steps per epoch
NUM_CLASSES = 1                # Binary: real vs fake

# ─── Training Phases ─────────────────────────────────────────────
EPOCHS_HEAD = 8                # Phase 1: train head only (frozen base)
EPOCHS_FINETUNE = 12           # Phase 2: fine-tune top layers
LR_HEAD = 3e-4                 # Phase 1 learning rate (higher for head-only)
LR_FINETUNE = 1e-5             # Phase 2 (fine-tuning) learning rate
FINETUNE_LAYERS = 50           # Number of top backbone layers to unfreeze
MAX_TRAIN_SAMPLES = 20000      # Subsample train set for CPU training speed
                               # Set to None to use all 100K (GPU recommended)

# ─── Regularization ──────────────────────────────────────────────
LABEL_SMOOTHING = 0.1          # Prevents overconfidence, improves generalization
DROPOUT_HEAD = 0.5             # Dropout after first dense
DROPOUT_TAIL = 0.3             # Dropout after second dense
MIXUP_ALPHA = 0.2              # Mixup augmentation strength (0 = disabled)

# ─── CBAM Attention ──────────────────────────────────────────────
CBAM_RATIO = 8

# ─── Paths ───────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data" / "processed"
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
REAL_RAW       = RAW_DIR / "real"
FAKE_RAW       = RAW_DIR / "fake"
REAL_PROCESSED = DATA_DIR / "real"
FAKE_PROCESSED = DATA_DIR / "fake"
MODELS_DIR     = PROJECT_ROOT / "models"

MODEL_PATH     = MODELS_DIR / "deepfake_detector.keras"   # primary (modern format)
MODEL_PATH_H5  = MODELS_DIR / "deepfake_detector_cbam.h5" # legacy fallback
HISTORY_PLOT   = MODELS_DIR / "training_history.png"

# ─── Chrome Extension ────────────────────────────────────────────
EXTENSION_DIR  = PROJECT_ROOT / "extension"
TFJS_MODEL_DIR = EXTENSION_DIR / "model"

# ─── Preprocessing ───────────────────────────────────────────────
FACE_PADDING   = 0.30     # 30% padding around face bounding box
FRAME_STEP     = 10       # Sample every Nth frame (≈3 fps at 30 fps)

# ─── Dataset ─────────────────────────────────────────────────────
DATASET_SLUG      = "xhlulu/140k-real-and-fake-faces"
DATASET_NAME      = "140K Real and Fake Faces"
# After download, the dataset provides train/valid/test splits:
#   train/ (real: 50K, fake: 50K)
#   valid/ (real: 10K, fake: 10K)
#   test/  (real: 10K, fake: 10K)
TRAIN_DIR         = DATA_DIR / "train"
VALID_DIR         = DATA_DIR / "valid"
TEST_DIR          = DATA_DIR / "test"

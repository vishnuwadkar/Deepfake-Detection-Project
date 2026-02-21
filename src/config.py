"""
config.py — Centralized configuration for the Deepfake Detection project.
All scripts import from here; change once, applies everywhere.
"""

from pathlib import Path

# ─── Image / Model ───────────────────────────────────────────────
IMG_SIZE = (224, 224)          # Xception optimal input (up from 128×128)
BATCH_SIZE = 32

# ─── Training Phases ─────────────────────────────────────────────
EPOCHS_HEAD = 20               # Phase 1: train head only (frozen base)
EPOCHS_FINETUNE = 20           # Phase 2: fine-tune top layers
LR_HEAD = 1e-4                 # Phase 1 learning rate
LR_FINETUNE = 1e-5             # Phase 2 (fine-tuning) learning rate
FINETUNE_LAYERS = 30           # Number of top Xception layers to unfreeze

# ─── CBAM Attention ──────────────────────────────────────────────
CBAM_RATIO = 8

# ─── Paths ───────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
REAL_RAW      = RAW_DIR / "real"
FAKE_RAW      = RAW_DIR / "fake"
REAL_PROCESSED = DATA_DIR / "real"
FAKE_PROCESSED = DATA_DIR / "fake"
MODELS_DIR    = PROJECT_ROOT / "models"

MODEL_PATH    = MODELS_DIR / "deepfake_detector.keras"   # primary (modern format)
MODEL_PATH_H5 = MODELS_DIR / "deepfake_detector_cbam.h5" # legacy fallback
HISTORY_PLOT  = MODELS_DIR / "training_history.png"

# ─── Preprocessing ───────────────────────────────────────────────
FACE_PADDING  = 0.30     # 30% padding around MTCNN face bounding box
FRAME_STEP    = 10       # Sample every Nth frame (≈3 fps at 30 fps)
JPEG_QUALITY  = 95       # cv2 JPEG save quality (0-100)

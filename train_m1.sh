#!/bin/bash
# =============================================================================
#  DeepGuard AI — M1 Mac Training Script (M1 Base, 8GB RAM Optimized)
#  Run this file ONCE. It does everything automatically:
#    1. Creates Python virtual environment
#    2. Installs all dependencies (TF Metal for GPU acceleration)
#    3. Downloads the 140K dataset from Kaggle
#    4. Trains the model (optimized for M1 base, ~3-4 hours)
#    5. Evaluates and generates accuracy report
#    6. Converts model to TF.js for the extension
# =============================================================================

set -e  # Stop immediately if any command fails

# --- Colors for pretty output ------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo ""
echo -e "${BOLD}${BLUE}================================================================${NC}"
echo -e "${BOLD}${BLUE}   DeepGuard AI — M1 Mac Local Training Pipeline${NC}"
echo -e "${BOLD}${BLUE}   Optimized for: Apple M1 Base (8GB Unified Memory)${NC}"
echo -e "${BOLD}${BLUE}================================================================${NC}"
echo ""

# Get script directory (works regardless of where you run it from)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo -e "${GREEN}[INFO] Working directory: $SCRIPT_DIR${NC}"

# =============================================================================
# STEP 1: Check Python version
# =============================================================================
echo ""
echo -e "${BOLD}[1/7] Checking Python...${NC}"

if command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
elif command -v python3.10 &>/dev/null; then
    PYTHON=python3.10
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo -e "${RED}[ERROR] Python 3 not found. Install via: brew install python@3.11${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "${GREEN}[OK] Using: $PYTHON_VERSION${NC}"

# =============================================================================
# STEP 2: Create virtual environment
# =============================================================================
echo ""
echo -e "${BOLD}[2/7] Setting up virtual environment...${NC}"

if [ ! -d "venv_mac" ]; then
    echo "  Creating fresh venv_mac..."
    $PYTHON -m venv venv_mac
    echo -e "${GREEN}  [OK] Virtual environment created${NC}"
else
    echo -e "${YELLOW}  [SKIP] venv_mac already exists${NC}"
fi

# Activate it
source venv_mac/bin/activate
echo -e "${GREEN}  [OK] Virtual environment activated${NC}"

# =============================================================================
# STEP 3: Install dependencies (Metal-accelerated TensorFlow)
# =============================================================================
echo ""
echo -e "${BOLD}[3/7] Installing dependencies (this may take 3-5 minutes)...${NC}"

pip install --upgrade pip --quiet

# Check if TF Metal already installed
if python -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" 2>/dev/null | grep -q "GPU"; then
    echo -e "${YELLOW}  [SKIP] TensorFlow Metal already installed${NC}"
else
    echo "  Installing tensorflow-macos + tensorflow-metal..."
    pip install tensorflow-macos==2.13.0 tensorflow-metal==1.0.1 --quiet
    echo -e "${GREEN}  [OK] TensorFlow Metal installed${NC}"
fi

# Install other requirements
pip install -r requirements_mac.txt --quiet
echo -e "${GREEN}  [OK] All dependencies installed${NC}"

# =============================================================================
# STEP 4: Verify Metal GPU is active
# =============================================================================
echo ""
echo -e "${BOLD}[4/7] Verifying Apple Metal GPU acceleration...${NC}"

python -c "
import tensorflow as tf
import os

# Suppress TF info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  [OK] Metal GPU detected: {len(gpus)} device(s)')
    for g in gpus:
        print(f'       {g}')
else:
    print('  [WARN] No Metal GPU. Training will use CPU (slower).')
    print('  Tip: Make sure tensorflow-metal is installed correctly.')
print(f'  TensorFlow version: {tf.__version__}')
"

# =============================================================================
# STEP 5: Download dataset (skip if already downloaded)
# =============================================================================
echo ""
echo -e "${BOLD}[5/7] Dataset setup...${NC}"

if [ -d "data/processed/train/real" ] && [ "$(ls -A data/processed/train/real 2>/dev/null | wc -l)" -gt "1000" ]; then
    TRAIN_COUNT=$(ls data/processed/train/real | wc -l | tr -d ' ')
    echo -e "${YELLOW}  [SKIP] Dataset already exists (train/real: $TRAIN_COUNT files)${NC}"
else
    echo "  Dataset not found. Starting download (~3.8 GB)..."
    echo ""

    # Check for kaggle.json
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo -e "${RED}  [ERROR] kaggle.json not found at ~/.kaggle/kaggle.json${NC}"
        echo ""
        echo "  To fix this:"
        echo "  1. Visit: https://www.kaggle.com/settings"
        echo "  2. API section -> Create New Token"
        echo "  3. Move the downloaded kaggle.json:"
        echo "     mkdir -p ~/.kaggle"
        echo "     mv ~/Downloads/kaggle.json ~/.kaggle/"
        echo "     chmod 600 ~/.kaggle/kaggle.json"
        echo "  4. Re-run this script"
        exit 1
    fi

    python download_data.py
fi

# =============================================================================
# STEP 6: Train the model (M1 optimized settings)
# =============================================================================
echo ""
echo -e "${BOLD}[6/7] Starting training (M1 Base optimized)...${NC}"
echo ""
echo -e "${YELLOW}  Settings for M1 Base 8GB:${NC}"
echo "    Batch size     : 32  (memory-safe for 8GB)"
echo "    Training data  : 40,000 images  (balanced for speed vs accuracy)"
echo "    Phase 1        : 10 epochs  (head training, ~50 min)"
echo "    Phase 2        : 12 epochs  (fine-tuning, ~80 min)"
echo "    Expected total : ~2.5 - 3.5 hours"
echo "    Target accuracy: 90-95%"
echo ""

# Override config settings specifically for M1 base via environment variables
export DEEPGUARD_BATCH_SIZE=32
export DEEPGUARD_MAX_SAMPLES=40000
export DEEPGUARD_EPOCHS_HEAD=10
export DEEPGUARD_EPOCHS_FINETUNE=12

# Suppress Metal warnings that clutter the output
export TF_CPP_MIN_LOG_LEVEL=2
export METAL_DEVICE_WRAPPING_ENABLED=1

python -c "
import os, sys
sys.path.insert(0, '.')
import src.config as cfg

# Apply M1 optimizations
cfg.BATCH_SIZE          = int(os.environ.get('DEEPGUARD_BATCH_SIZE', 32))
cfg.MAX_TRAIN_SAMPLES   = int(os.environ.get('DEEPGUARD_MAX_SAMPLES', 40000))
cfg.EPOCHS_HEAD         = int(os.environ.get('DEEPGUARD_EPOCHS_HEAD', 10))
cfg.EPOCHS_FINETUNE     = int(os.environ.get('DEEPGUARD_EPOCHS_FINETUNE', 12))

print(f'  Config applied: batch={cfg.BATCH_SIZE}, samples={cfg.MAX_TRAIN_SAMPLES}, '
      f'epochs={cfg.EPOCHS_HEAD}+{cfg.EPOCHS_FINETUNE}')

from src.train import train
train()
"

echo ""
echo -e "${GREEN}[OK] Training complete!${NC}"

# =============================================================================
# STEP 7: Evaluate + Convert to TF.js
# =============================================================================
echo ""
echo -e "${BOLD}[7/7] Evaluating model & converting to TF.js...${NC}"

# Evaluate
python src/evaluate.py
echo -e "${GREEN}  [OK] Evaluation report saved to models/evaluation_report.png${NC}"

# Install TF.js converter
pip install tensorflowjs --quiet

# Convert
python scripts/convert_model.py
echo -e "${GREEN}  [OK] TF.js model saved to extension/model/${NC}"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}================================================================${NC}"
echo -e "${BOLD}${GREEN}   ALL DONE! DeepGuard AI model is ready.${NC}"
echo -e "${BOLD}${GREEN}================================================================${NC}"
echo ""
echo "  Files created:"
echo "    models/deepfake_detector.keras        <- trained model"
echo "    models/deepfake_detector_phase1.keras  <- phase 1 backup"
echo "    models/evaluation_report.png           <- accuracy report"
echo "    models/training_history.png            <- loss/accuracy graphs"
echo "    extension/model/model.json             <- TF.js model (for Chrome)"
echo "    extension/model/*.bin                  <- TF.js weight shards"
echo ""
echo "  Next steps on your Windows PC:"
echo "    1. Copy  models/deepfake_detector.keras  ->  models/"
echo "    2. Copy  extension/model/*               ->  extension/model/"
echo "    3. Open Chrome -> chrome://extensions"
echo "    4. Load unpacked -> select extension/ folder"
echo ""
echo -e "${YELLOW}  Open evaluation_report.png to see your accuracy breakdown!${NC}"
echo ""

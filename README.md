# DeepGuard AI — Deepfake Detection Project

A deep learning deepfake detection system built with EfficientNetV2B0 + CBAM Attention, available as both a **Streamlit web app** and a **Chrome Extension** for real-time AI-generated content detection on any web page.

## 🛡️ Chrome Extension (DeepGuard AI)

### Features
- **Real-time page scanning** — Automatically detects AI-generated images and videos on any web page
- **100% local inference** — All analysis runs on-device using TensorFlow.js; no data ever leaves your machine
- **Smart detection** — MutationObserver-based scanning catches dynamically loaded content (infinite scroll, SPAs)
- **Video frame analysis** — Extracts and analyzes video frames at configurable intervals
- **Premium UI** — Glassmorphism popup dashboard with real-time stats, detection history, and configurable settings
- **In-page overlays** — Non-intrusive badges on scanned images showing Real/AI-Generated status with confidence scores
- **User-configurable** — Toggle auto-scan, adjust sensitivity, show/hide overlays, enable/disable video scanning

### Extension Setup

1. **Convert the trained model** to TensorFlow.js format:
   ```bash
   pip install tensorflowjs
   python scripts/convert_model.py
   ```
   This produces `extension/model/model.json` + weight `.bin` files.

2. **Load the extension** in Chrome:
   - Open `chrome://extensions/`
   - Enable **Developer mode** (top-right toggle)
   - Click **Load unpacked**
   - Select the `extension/` folder

3. **Usage**:
   - Click the 🛡️ DeepGuard AI icon in the toolbar
   - Click **"Scan This Page"** to analyze all images/videos
   - Toggle **Auto-Scan** in settings to scan pages automatically
   - Hover over badges on images to see detailed confidence scores

### Extension Architecture

```
extension/
├── manifest.json          # Manifest V3 configuration
├── background.js          # Service worker (message routing, state management)
├── content/
│   ├── content.js         # DOM scanner, MutationObserver, overlay injection
│   └── overlay.css        # In-page glassmorphism badges
├── offscreen/
│   ├── offscreen.html     # Hidden document for TF.js inference
│   └── offscreen.js       # Model loading, preprocessing, prediction
├── popup/
│   ├── popup.html         # Dashboard UI
│   ├── popup.css          # Premium glassmorphism styles
│   └── popup.js           # Settings, stats, scan control
├── lib/
│   └── tf.min.js          # TensorFlow.js library (~1.4 MB)
├── model/
│   ├── model.json         # Converted model topology
│   └── *.bin              # Weight shards (~10-15 MB)
└── assets/
    └── icons/             # Extension icons (16, 48, 128px)
```

---

## 🔬 ML Training Pipeline (Python)

### Folder Structure
- `data/raw`: Original dataset images/videos (managed by `download_data.py`).
- `data/processed`: Cropped faces and preprocessed data.
- `models/`: Saved model weights (`.h5`, `.keras`).
- `src/`: Source code for preprocessing, training, and inference.
- `notebooks/`: Jupyter notebooks for experimentation.
- `scripts/`: Model conversion utilities.
- `download_data.py`: Script to download and set up the dataset.

### Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup
1. **Kaggle API**:
   - Go to [Kaggle Account](https://www.kaggle.com/account).
   - Click "Create New API Token" to download `kaggle.json`.
   - Place `kaggle.json` in `C:\Users\<YourUser>\.kaggle\` OR in this project folder.
2. **Download Data**:
   ```bash
   python download_data.py
   ```
   This will download the `ciplab/real-and-fake-face-detection` dataset and organize it into `data/raw/real` and `data/raw/fake`.

### Data Preprocessing
Run the preprocessing script to extract faces from videos/images:
```bash
python src/preprocess.py
```
This will populate `data/processed/real` and `data/processed/fake`.

### Model Training
Train the EfficientNetV2B0 + CBAM model with two-phase training:
```bash
python src/train.py
```
This saves the model to `models/deepfake_detector.keras`.

### Evaluation
Generate a comprehensive evaluation report:
```bash
python src/evaluate.py
```

### Streamlit App
Launch the web interface for manual video analysis:
```bash
streamlit run app.py
```

---

## Model Architecture
- **Base Model**: EfficientNetV2B0 (pre-trained on ImageNet)
- **Attention**: CBAM (Convolutional Block Attention Module)
- **Custom Head**: GAP → Dense(512) → BN → Dropout → Dense(256) → BN → Dropout → Sigmoid
- **Training**: Two-phase (frozen backbone → fine-tuning top 30 layers)
- **Loss**: Binary Focal Crossentropy
- **Metrics**: Accuracy, AUC, Precision, Recall

## Requirements
- Python 3.10+
- TensorFlow 2.12+
- Chrome 120+ (for extension)
- See `requirements.txt` for full dependency list

# Deepfake Detection Project

A deep learning project to detect deepfake images and videos using an Xception-based CNN classifier.

## Folder Structure
- `data/raw`: Original dataset images/videos (managed by `download_data.py`).
- `data/processed`: Cropped faces and preprocessed data.
- `models/`: Saved model weights (`.h5`, `.keras`).
- `src/`: Source code for preprocessing, training, and inference.
- `notebooks/`: Jupyter notebooks for experimentation.
- `download_data.py`: Script to download and set up the dataset.

## Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install tensorflow opencv-python mtcnn streamlit matplotlib kaggle tqdm
   ```

### 3. Dataset Setup
1. **Kaggle API**:
   - Go to [Kaggle Account](https://www.kaggle.com/account).
   - Click "Create New API Token" to download `kaggle.json`.
   - Place `kaggle.json` in `C:\Users\<YourUser>\.kaggle\` OR in this project folder.
2. **Download Data**:
   ```bash
   python download_data.py
   ```
   This will download the `ciplab/real-and-fake-face-detection` dataset and organize it into `data/raw/real` and `data/raw/fake`.

### 4. Data Preprocessing
Run the preprocessing script to extract faces from videos/images and resize them to 128x128:
```bash
python src/preprocess.py
```
This will populate `data/processed/real` and `data/processed/fake`.

### 5. Running the App
Once the model is trained, launch the Streamlit app:
```bash
streamlit run app.py
```
- **Upload**: Select an MP4 video.
- **Analyze**: Click the button to extract frames and get a prediction.
- **Result**: The app will show "REAL" or "FAKE" with a confidence score.

### 6. Advanced Model Training (Optional)
To train the upgraded model with **Cutout Augmentation** and **CBAM Attention**:
```bash
python src/train.py
```
This will save the model to `models/deepfake_detector_cbam.h5`.
Note: The script currently defaults to this advanced configuration.

## Implementation Plan: Xception-based CNN Classifier

### 1. Data Preprocessing
- **Face Extraction**: Use MTCNN (Multi-task Cascaded Convolutional Networks) to detect and crop faces from video frames or images.
- **Normalization**: Resize faces to the required input size for Xception (usually 299x299) and normalize pixel values to [-1, 1].
- **Augmentation**: Apply random rotations, flips, and brightness adjustments to prevent overfitting.

### 2. Model Architecture
- **Base Model**: Xception (pre-trained on ImageNet) used as a feature extractor.
- **Custom Head**:
  - Global Average Pooling layer.
  - Dense layer (e.g., 512 units) with ReLU activation and Dropout.
  - Output Dense layer (1 unit) with Sigmoid activation for binary classification (Real vs. Fake).

### 3. Training Strategy
- **Transfer Learning**: Freeze the base Xception layers initially and train only the custom head.
- **Fine-tuning**: Unfreeze the top layers of Xception and retrain with a lower learning rate.
- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam.

### 4. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Confusion Matrix to analyze false positives/negatives.

### 5. Deployment
- **Streamlit App**: A simple web interface where users can upload an image or video to get a real/fake prediction with a confidence score.

"""
app.py — Optimized Streamlit Deepfake Detection App.

Key improvements over original:
  - MTCNN face detection during inference (matches training pipeline)
  - Correct Xception preprocess_input() normalization (was /255.0 — broken)
  - IMG_SIZE updated to 224×224 to match trained model
  - Batched model.predict(batch_size=32) — avoids OOM on long videos
  - Fixed progress bar (shows actual verdict confidence)
  - Face detection count displayed in UI
  - Temp file cleanup in finally block (no leaks on errors)
  - Graceful fallback if no face detected in a frame
"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st
import cv2
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import IMG_SIZE, BATCH_SIZE, MODEL_PATH, MODEL_PATH_H5, FACE_PADDING, FRAME_STEP
from src.model import build_model, CUSTOM_OBJECTS
from tensorflow.keras.applications.xception import preprocess_input

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.1rem;
    }

    .verdict-fake {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 32px rgba(255, 65, 108, 0.4);
        animation: pulse 1.5s infinite;
    }
    .verdict-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.4);
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    .info-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 0.8rem;
    }
    .metric-label { color: #aaa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #fff; font-size: 1.4rem; font-weight: 700; }

    .warning-box {
        background: rgba(255, 200, 0, 0.1);
        border-left: 4px solid #ffc800;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        color: #ffc800;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    div[data-testid="stProgress"] > div > div {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached Resources
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading face detector...")
def load_face_detector():
    """Loads MTCNN once and caches across sessions."""
    from mtcnn import MTCNN
    return MTCNN()


@st.cache_resource(show_spinner="Loading deepfake detection model...")
def load_model():
    """
    Loads a trained model from disk.

    Strategy:
      1. Try tf.keras.models.load_model() — loads architecture + weights
         together from .keras or .h5. This is robust to architecture changes
         and is the correct approach for fully-saved models.
      2. If that fails (e.g. custom Lambda layer deserialization issues),
         fall back to build_model() + load_weights() which requires the
         current architecture to exactly match the saved weights.
      3. Return None if no model file exists at all.
    """
    import tensorflow as tf

    # Custom objects needed to deserialize CBAM's serializable layers
    custom_objects = {**CUSTOM_OBJECTS, "tf": tf}

    # Prefer new .keras format, fall back to legacy .h5
    candidates = [p for p in (MODEL_PATH, MODEL_PATH_H5) if p.exists()]
    if not candidates:
        return None

    for target in candidates:
        # ── Attempt 1: Full model load (architecture + weights) ──────────
        try:
            model = tf.keras.models.load_model(
                str(target),
                custom_objects=custom_objects,
                compile=False,
            )
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            return model
        except Exception as e1:
            # ── Attempt 2: Build fresh architecture + load weights ────────
            try:
                model = build_model(trainable_base=False)
                model.load_weights(str(target))
                return model
            except Exception as e2:
                # This model file is incompatible — try next candidate
                continue

    return None  # All candidates failed — user must retrain


# ─────────────────────────────────────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_face(frame_bgr: np.ndarray, box: tuple) -> np.ndarray | None:
    """Crops a face with 30% padding, returns None if degenerate."""
    x, y, w, h = box
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_bgr.shape[1], x + w + pad_x)
    y2 = min(frame_bgr.shape[0], y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def extract_faces_from_video(video_path: str, detector, max_frames: int = 100):
    """
    Extracts face crops from a video for inference.

    Samples every FRAME_STEP frames. Uses MTCNN to crop faces with padding.
    Falls back to full-frame crop if no face is detected (flagged with a warning).

    Args:
        video_path: Path to the video file.
        detector:   Cached MTCNN instance.
        max_frames: Maximum number of face crops to collect.

    Returns:
        Tuple of:
          - np.ndarray of preprocessed face crops, shape (N, H, W, 3)
          - int: number of frames where no face was detected (fallback used)
          - int: total frames sampled
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.array([]), 0, 0

    faces_collected = []
    fallback_count = 0
    frames_sampled = 0
    current_frame = 0

    while len(faces_collected) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % FRAME_STEP == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = detector.detect_faces(rgb_frame)
            frames_sampled += 1

            if detected:
                # Use the highest-confidence face detection
                best = max(detected, key=lambda d: d["confidence"])
                cropped = _crop_face(frame, best["box"])
                if cropped is not None:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    resized    = cv2.resize(cropped_rgb, IMG_SIZE)
                    faces_collected.append(resized)
                else:
                    fallback_count += 1
            else:
                # No face found — use full frame as fallback
                rgb_resized = cv2.resize(rgb_frame, IMG_SIZE)
                faces_collected.append(rgb_resized)
                fallback_count += 1

        current_frame += 1

    cap.release()

    if not faces_collected:
        return np.array([]), fallback_count, frames_sampled

    # Stack and apply Xception preprocess_input (scale to [-1, 1])
    faces_array = np.array(faces_collected, dtype=np.float32)  # [0, 255]
    faces_preprocessed = preprocess_input(faces_array)         # [-1, 1]

    return faces_preprocessed, fallback_count, frames_sampled


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🕵️ Deepfake Detector</h1>
        <p>Upload a video — AI analyzes each face frame-by-frame to detect manipulations.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Resources ──────────────────────────────────────────────────────
    model    = load_model()
    detector = load_face_detector()

    if model is None:
        st.error(
            f"No trained model found. Checked:\n"
            f"- `{MODEL_PATH}`\n"
            f"- `{MODEL_PATH_H5}`"
        )
        st.info("Run `python src/preprocess.py` then `python src/train.py` to train the model.")
        return

    # ── File Upload ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported: MP4, MOV, AVI, MKV",
    )

    if uploaded_file is None:
        st.markdown(
            "<br><div style='text-align:center; color:#555; font-size:1.1rem;'>"
            "⬆️ Upload a video above to begin analysis</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Write to temp file ──────────────────────────────────────────────────
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = tfile.name
    try:
        tfile.write(uploaded_file.read())
        tfile.close()

        col_video, col_results = st.columns([1, 1], gap="large")

        with col_video:
            st.markdown("#### 🎬 Uploaded Video")
            st.video(video_path)

        with col_results:
            st.markdown("#### 🔬 Analysis")
            analyze_btn = st.button(
                "🔍 Analyze Video",
                type="primary",
                use_container_width=True,
            )

            if analyze_btn:
                with st.spinner("Detecting faces and analyzing frames..."):
                    faces, fallback_count, frames_sampled = extract_faces_from_video(
                        video_path, detector, max_frames=100
                    )

                if len(faces) == 0:
                    st.error("❌ Could not extract any frames from the video.")
                else:
                    # ── Run Inference ─────────────────────────────────────
                    predictions = model.predict(faces, batch_size=BATCH_SIZE, verbose=0)
                    preds_flat  = predictions.flatten()
                    avg_score   = float(np.mean(preds_flat))
                    is_fake     = avg_score > 0.5

                    # Confidence = how certain the model is in ITS verdict
                    confidence = avg_score if is_fake else (1.0 - avg_score)

                    st.divider()

                    # ── Verdict Banner ─────────────────────────────────────
                    if is_fake:
                        st.markdown(
                            '<div class="verdict-fake">🚨 DEEPFAKE DETECTED</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="verdict-real">✅ AUTHENTIC VIDEO</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Metrics Row ────────────────────────────────────────
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""
                        <div class="info-card">
                          <div class="metric-label">Confidence</div>
                          <div class="metric-value">{confidence*100:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="info-card">
                          <div class="metric-label">Faces Analyzed</div>
                          <div class="metric-value">{len(faces)}</div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        fake_frame_pct = float(np.mean(preds_flat > 0.5)) * 100
                        st.markdown(f"""
                        <div class="info-card">
                          <div class="metric-label">Fake Frames</div>
                          <div class="metric-value">{fake_frame_pct:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                    # ── Confidence Bar ─────────────────────────────────────
                    st.markdown(f"**Fakeness Score** (0 = Real · 1 = Fake): `{avg_score:.3f}`")
                    st.progress(float(avg_score))

                    # ── Fallback Warning ───────────────────────────────────
                    if fallback_count > 0:
                        pct = fallback_count / max(frames_sampled, 1) * 100
                        st.markdown(
                            f'<div class="warning-box">⚠️ No face detected in '
                            f'{fallback_count} of {frames_sampled} sampled frames '
                            f'({pct:.0f}%) — full frame used as fallback. '
                            f'Results may be less accurate.</div>',
                            unsafe_allow_html=True,
                        )

                    # ── Per-Frame Chart ────────────────────────────────────
                    st.markdown("##### Per-Frame Fakeness Score")
                    import pandas as pd
                    chart_data = pd.DataFrame({
                        "Frame Score (0=Real, 1=Fake)": preds_flat,
                        "Threshold (0.5)": [0.5] * len(preds_flat),
                    })
                    st.line_chart(chart_data, use_container_width=True)

    finally:
        # Always clean up the temp file, even if an exception occurred
        try:
            os.remove(video_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()

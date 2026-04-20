"""
DeepGuard AI — Premium Web Interface
Standalone web app for image & video deepfake detection.
Uses the same trained model as the Chrome extension.

Run:  streamlit run app.py
"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st
import cv2
from pathlib import Path
from PIL import Image
import io

# ── Path setup ──────────────────────────────────────────────────────────────
from src.config import IMG_SIZE, BATCH_SIZE, MODEL_PATH, MODEL_PATH_H5, FACE_PADDING, FRAME_STEP, MODELS_DIR
from src.model import build_model, CUSTOM_OBJECTS

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard AI — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark UI with Glassmorphism
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Global Background ─────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #101030 40%, #0d1117 100%);
    }

    /* ── Hero Header ───────────────────────────── */
    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        position: relative;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50px;
        left: 50%;
        transform: translateX(-50%);
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(72,202,228,0.2));
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 100px;
        padding: 0.35rem 1.2rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #8b8bff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 50%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    .hero p {
        color: #7a7a9e;
        font-size: 1.05rem;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Glass Card ────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .glass-card-sm {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* ── Verdict Banners ───────────────────────── */
    .verdict-fake {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        box-shadow: 0 8px 40px rgba(255, 65, 108, 0.35);
        animation: pulse-red 2s infinite;
    }
    .verdict-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        box-shadow: 0 8px 40px rgba(17, 153, 142, 0.35);
    }
    @keyframes pulse-red {
        0%, 100% { transform: scale(1); box-shadow: 0 8px 40px rgba(255, 65, 108, 0.35); }
        50% { transform: scale(1.01); box-shadow: 0 12px 50px rgba(255, 65, 108, 0.5); }
    }

    /* ── Metric Cards ──────────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label { color: #7a7a9e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    .metric-value { color: #e0e0ff; font-size: 1.8rem; font-weight: 800; margin-top: 0.3rem; }
    .metric-sub { color: #5a5a7e; font-size: 0.75rem; margin-top: 0.2rem; }

    /* ── Score Bar ──────────────────────────────── */
    .score-container {
        width: 100%;
        position: relative;
        margin: 1.5rem 0;
    }
    .score-bar-bg {
        width: 100%;
        height: 14px;
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        overflow: hidden;
    }
    .score-bar-fill-fake {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        transition: width 1s ease;
    }
    .score-bar-fill-real {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        transition: width 1s ease;
    }
    .score-label {
        display: flex;
        justify-content: space-between;
        color: #5a5a7e;
        font-size: 0.75rem;
        margin-top: 0.4rem;
    }

    /* ── Upload Zone ───────────────────────────── */
    .upload-zone {
        border: 2px dashed rgba(108,99,255,0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(108,99,255,0.03);
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(108,99,255,0.6);
        background: rgba(108,99,255,0.06);
    }
    .upload-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .upload-title { color: #b0b0d0; font-size: 1.1rem; font-weight: 600; }
    .upload-sub { color: #5a5a7e; font-size: 0.85rem; margin-top: 0.3rem; }

    /* ── Warning Box ───────────────────────────── */
    .warning-box {
        background: rgba(255, 200, 0, 0.08);
        border-left: 4px solid #ffc800;
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.2rem;
        color: #d4a800;
        font-size: 0.85rem;
        margin: 0.8rem 0;
    }

    /* ── Mode Toggle ───────────────────────────── */
    .mode-tab {
        display: inline-flex;
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .mode-btn {
        padding: 0.6rem 1.8rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
        background: transparent;
        color: #7a7a9e;
    }
    .mode-btn-active {
        background: linear-gradient(135deg, #6C63FF, #5a54e6);
        color: white;
        box-shadow: 0 4px 16px rgba(108,99,255,0.3);
    }

    /* ── Footer ────────────────────────────────── */
    .footer {
        text-align: center;
        color: #3a3a5e;
        font-size: 0.78rem;
        padding: 3rem 0 1rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 3rem;
    }

    /* ── Hide Streamlit Defaults ────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

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


@st.cache_resource(show_spinner="Loading DeepGuard AI model...")
def load_model():
    """
    Loads a trained model from disk.
    Tries multiple file formats: .keras, .h5 (from training), legacy .h5.
    """
    import tensorflow as tf

    custom_objects = {**CUSTOM_OBJECTS, "tf": tf}

    # All candidate model paths (in priority order)
    candidates = []
    for ext_path in [
        MODEL_PATH,                                           # .keras
        MODELS_DIR / "deepfake_detector.h5",                 # new .h5
        MODEL_PATH_H5,                                       # legacy .h5
        MODELS_DIR / "deepfake_detector_phase1.h5",          # phase 1 backup
    ]:
        if ext_path.exists():
            candidates.append(ext_path)

    if not candidates:
        return None

    for target in candidates:
        # Attempt 1: Full model load
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
        except Exception:
            # Attempt 2: Build + load weights
            try:
                model = build_model(trainable_base=False)
                model.load_weights(str(target))
                return model
            except Exception:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_face(frame_bgr: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """Crops a face with padding, returns None if degenerate."""
    x, y, w, h = bbox
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_bgr.shape[1], x + w + pad_x)
    y2 = min(frame_bgr.shape[0], y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def analyze_image(image_data, model, detector):
    """
    Analyzes a single image for deepfake content.
    Returns: (prediction_score, face_count, faces_array)
    """
    # Convert to numpy BGR
    if isinstance(image_data, Image.Image):
        img_rgb = np.array(image_data.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image_data

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect faces
    detections = detector.detect_faces(img_rgb)

    if detections:
        faces = []
        for det in detections:
            cropped = _crop_face(img_bgr, det["box"])
            if cropped is not None:
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(cropped_rgb, IMG_SIZE)
                faces.append(resized)

        if faces:
            faces_array = np.array(faces, dtype=np.float32)
            predictions = model.predict(faces_array, batch_size=BATCH_SIZE, verbose=0)
            avg_score = float(np.mean(predictions.flatten()))
            return avg_score, len(faces), faces_array
    
    # No face detected — use the full image
    resized = cv2.resize(img_rgb, IMG_SIZE)
    full_img = np.array([resized], dtype=np.float32)
    predictions = model.predict(full_img, batch_size=1, verbose=0)
    avg_score = float(predictions[0][0])
    return avg_score, 0, full_img


def analyze_video(video_path, model, detector, progress_callback=None, max_frames=100):
    """
    Analyzes a video frame-by-frame.
    Returns: (avg_score, per_frame_scores, face_count, missed_count, total_sampled)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, [], 0, 0, 0

    faces_collected = []
    per_frame_scores = []
    missed_count = 0
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
                best = max(detected, key=lambda d: d["confidence"])
                cropped = _crop_face(frame, best["box"])
                if cropped is not None:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(cropped_rgb, IMG_SIZE)
                    faces_collected.append(resized)
                else:
                    missed_count += 1
            else:
                missed_count += 1

            if progress_callback:
                progress_callback(frames_sampled / (total_frames / FRAME_STEP))

        current_frame += 1

    cap.release()

    if not faces_collected:
        return None, [], 0, missed_count, frames_sampled

    faces_array = np.array(faces_collected, dtype=np.float32)
    predictions = model.predict(faces_array, batch_size=BATCH_SIZE, verbose=0)
    per_frame_scores = predictions.flatten().tolist()
    avg_score = float(np.mean(per_frame_scores))

    return avg_score, per_frame_scores, len(faces_collected), missed_count, frames_sampled


def render_score_bar(score, is_fake):
    """Renders a custom glassmorphic score bar."""
    pct = int(score * 100)
    fill_class = "score-bar-fill-fake" if is_fake else "score-bar-fill-real"
    st.markdown(f"""
    <div class="score-container">
        <div class="score-bar-bg">
            <div class="{fill_class}" style="width: {pct}%;"></div>
        </div>
        <div class="score-label">
            <span>Authentic (Real)</span>
            <span>{score:.3f}</span>
            <span>AI Generated (Fake)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(is_fake, confidence, score):
    """Renders the verdict banner + metrics."""
    if is_fake:
        st.markdown('<div class="verdict-fake">AI-GENERATED CONTENT DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-real">AUTHENTIC CONTENT VERIFIED</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_score_bar(score, is_fake)


def render_metrics(confidence, face_count, extra_metrics=None):
    """Renders the metric cards row."""
    cols = st.columns(3 if extra_metrics else 2)

    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence*100:.1f}%</div>
            <div class="metric-sub">Model certainty in verdict</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        face_text = f"{face_count}" if face_count > 0 else "Full Image"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Faces Analyzed</div>
            <div class="metric-value">{face_text}</div>
            <div class="metric-sub">{"MTCNN face detection" if face_count > 0 else "No faces found, analyzed full image"}</div>
        </div>
        """, unsafe_allow_html=True)

    if extra_metrics:
        with cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{extra_metrics["label"]}</div>
                <div class="metric-value">{extra_metrics["value"]}</div>
                <div class="metric-sub">{extra_metrics["sub"]}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Hero Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">AI-Powered Detection Engine</div>
        <h1>DeepGuard AI</h1>
        <p>Upload images or videos to instantly detect AI-generated deepfakes. 
        Powered by EfficientNetV2 + CBAM attention — the same model used in the Chrome extension.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Model ───────────────────────────────────────────────────────────
    model = load_model()
    detector = load_face_detector()

    if model is None:
        st.error("No trained model found. Please train the model first.")
        st.markdown("""
        <div class="glass-card">
            <h4>How to train:</h4>
            <ol>
                <li>Download dataset: <code>python download_data.py</code></li>
                <li>Preprocess data: <code>python src/preprocess.py</code></li>
                <li>Train model: <code>python src/train.py</code></li>
                <li>Reload this page</li>
            </ol>
            <p>Or use the Colab notebook at <code>notebooks/DeepGuard_AI_Training.ipynb</code></p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Mode Selection ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode = st.radio(
            "Detection Mode",
            ["Image Analysis", "Video Analysis"],
            horizontal=True,
            label_visibility="collapsed",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE MODE
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "Image Analysis":
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown("""
            <div class="glass-card" style="text-align:center;">
                <div class="upload-icon">📸</div>
                <div class="upload-title">Upload Image for Analysis</div>
                <div class="upload-sub">Supports: JPG, PNG, WEBP (max 200MB)</div>
            </div>
            """, unsafe_allow_html=True)

            uploaded_image = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
                key="img_uploader",
            )

            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption=uploaded_image.name, use_container_width=True)

                st.markdown(f"""
                <div class="glass-card-sm">
                    <strong>File:</strong> {uploaded_image.name}<br>
                    <strong>Size:</strong> {uploaded_image.size / 1024:.1f} KB &nbsp;&nbsp;
                    <strong>Resolution:</strong> {image.size[0]} x {image.size[1]}
                </div>
                """, unsafe_allow_html=True)

        with col_result:
            if uploaded_image:
                analyze_btn = st.button(
                    "Analyze Image",
                    type="primary",
                    use_container_width=True,
                    key="analyze_img",
                )

                if analyze_btn:
                    with st.spinner("Running DeepGuard AI inference..."):
                        score, face_count, faces = analyze_image(image, model, detector)

                    is_fake = score > 0.5
                    confidence = score if is_fake else (1.0 - score)

                    render_verdict(is_fake, confidence, score)
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_metrics(confidence, face_count)

                    # Show detected faces
                    if face_count > 0 and faces is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("##### Detected Faces")
                        face_cols = st.columns(min(face_count, 4))
                        for i, face in enumerate(faces[:4]):
                            with face_cols[i]:
                                st.image(face.astype(np.uint8), caption=f"Face {i+1}", width=120)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding: 4rem 2rem;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🛡️</div>
                    <div style="color: #7a7a9e; font-size: 1.1rem;">
                        Upload an image to begin analysis
                    </div>
                    <div style="color: #4a4a6e; font-size: 0.85rem; margin-top: 0.5rem;">
                        The model detects AI-generated faces with 90-95% accuracy
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO MODE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown("""
            <div class="glass-card" style="text-align:center;">
                <div class="upload-icon">🎬</div>
                <div class="upload-title">Upload Video for Analysis</div>
                <div class="upload-sub">Supports: MP4, MOV, AVI, MKV (max 200MB)</div>
            </div>
            """, unsafe_allow_html=True)

            uploaded_video = st.file_uploader(
                "Choose a video",
                type=["mp4", "mov", "avi", "mkv"],
                label_visibility="collapsed",
                key="vid_uploader",
            )

            if uploaded_video:
                st.video(uploaded_video)

                st.markdown(f"""
                <div class="glass-card-sm">
                    <strong>File:</strong> {uploaded_video.name}<br>
                    <strong>Size:</strong> {uploaded_video.size / (1024*1024):.1f} MB
                </div>
                """, unsafe_allow_html=True)

        with col_result:
            if uploaded_video:
                analyze_btn = st.button(
                    "Analyze Video",
                    type="primary",
                    use_container_width=True,
                    key="analyze_vid",
                )

                if analyze_btn:
                    # Save to temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    video_path = tfile.name
                    tfile.write(uploaded_video.read())
                    tfile.close()

                    try:
                        progress_bar = st.progress(0, text="Detecting faces and analyzing frames...")

                        def update_progress(pct):
                            progress_bar.progress(min(pct, 1.0), text=f"Analyzing... {int(min(pct,1.0)*100)}%")

                        score, per_frame, face_count, missed, total_sampled = analyze_video(
                            video_path, model, detector,
                            progress_callback=update_progress,
                            max_frames=100,
                        )

                        progress_bar.empty()

                        if score is None:
                            st.error("Could not extract any faces from the video.")
                        else:
                            is_fake = score > 0.5
                            confidence = score if is_fake else (1.0 - score)

                            render_verdict(is_fake, confidence, score)
                            st.markdown("<br>", unsafe_allow_html=True)

                            fake_frame_pct = float(np.mean(np.array(per_frame) > 0.5)) * 100

                            render_metrics(confidence, face_count, extra_metrics={
                                "label": "Fake Frames",
                                "value": f"{fake_frame_pct:.0f}%",
                                "sub": f"{face_count} frames analyzed",
                            })

                            # Missed faces warning
                            if missed > 0:
                                pct = missed / max(total_sampled, 1) * 100
                                st.markdown(
                                    f'<div class="warning-box">No face detected in '
                                    f'{missed} of {total_sampled} sampled frames '
                                    f'({pct:.0f}%). These frames were safely skipped.</div>',
                                    unsafe_allow_html=True,
                                )

                            # Per-frame chart
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("##### Per-Frame Analysis")
                            import pandas as pd
                            chart_data = pd.DataFrame({
                                "Fakeness Score": per_frame,
                                "Threshold": [0.5] * len(per_frame),
                            })
                            st.line_chart(chart_data, use_container_width=True)

                    finally:
                        try:
                            os.remove(video_path)
                        except OSError:
                            pass
            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding: 4rem 2rem;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🛡️</div>
                    <div style="color: #7a7a9e; font-size: 1.1rem;">
                        Upload a video to begin analysis
                    </div>
                    <div style="color: #4a4a6e; font-size: 0.85rem; margin-top: 0.5rem;">
                        DeepGuard AI analyzes each frame for AI-generated content
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <strong>DeepGuard AI</strong> &mdash; AI-Powered Deepfake Detection<br>
        EfficientNetV2-B0 + CBAM Attention &bull; Trained on 140K Real &amp; Fake Faces<br>
        Privacy-First: All analysis runs locally on your machine. No data is uploaded.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

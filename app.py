import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input,
    GlobalMaxPooling2D, Reshape, Add, Activation, Multiply,
    Conv2D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Configuration
MODEL_PATH = "models/deepfake_detector_cbam.h5"
IMG_SIZE = (128, 128)

st.set_page_config(page_title="Deepfake Detector", layout="wide")

def cbam_block(input_tensor, ratio=8):
    """
    Convolutional Block Attention Module (CBAM)
    """
    # --- Channel Attention ---
    channel = input_tensor.shape[-1]
    
    # Shared MLP layers
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    # Average Pooling path
    avg_pool = GlobalAveragePooling2D()(input_tensor)    
    avg_out = shared_layer_two(shared_layer_one(avg_pool))
    
    # Max Pooling path
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_out = shared_layer_two(shared_layer_one(max_pool))
    
    # Combine
    channel_out = Add()([avg_out, max_out])
    channel_out = Activation('sigmoid')(channel_out)
    channel_out = Reshape((1, 1, channel))(channel_out)
    
    # Apply Channel Attention
    channel_refined = Multiply()([input_tensor, channel_out])
    
    # --- Spatial Attention ---
    # Average Pooling along channel axis
    avg_pool_spatial = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(channel_refined)
    # Max Pooling along channel axis
    max_pool_spatial = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(channel_refined)
    
    # Concatenate
    spatial = Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])
    
    # Conv layer
    spatial = Conv2D(1, (7, 7), strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(spatial)
    
    # Apply Spatial Attention
    output_tensor = Multiply()([channel_refined, spatial])
    
    return output_tensor

def build_model():
    # Base model: Xception (frozen)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False

    x = base_model.output
    
    # Insert CBAM Block
    x = cbam_block(x)
    
    # Custom head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    
    # Build the architecture first
    model = build_model()
    # Load weights
    model.load_weights(MODEL_PATH)
    return model

def extract_random_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return []

    # Select random frame indices
    frame_indices = sorted(np.random.choice(total_frames, min(num_frames, total_frames), replace=False))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and Preprocess
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def main():
    st.title("🕵️‍♂️ Deepfake Detection Demo")
    st.markdown("Upload a video to verify if it's **Real** or **Fake**.")

    model = load_model()

    if model is None:
        st.error(f"Model not found at `{MODEL_PATH}`. Please train the model first.")
        st.info("Run `python src/train.py` to train the model.")
        return

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
        tfile.write(uploaded_file.read())
        tfile.close() # Close file so it can be opened by other processes (opencv)
        video_path = tfile.name

        col1, col2 = st.columns(2)

        with col1:
            st.video(video_path)

        with col2:
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Extracting frames and analyzing..."):
                    # Extract frames
                    frames = extract_random_frames(video_path)
                    
                    if len(frames) == 0:
                        st.error("Could not extract frames from video.")
                    else:
                        # Normalize
                        frames_norm = frames / 255.0
                        
                        # Predict
                        predictions = model.predict(frames_norm)
                        avg_score = np.mean(predictions)
                        
                        # Display Results
                        st.divider()
                        if avg_score > 0.5:
                            st.error(f"## 🚨 FAKE DETECTED")
                            confidence = avg_score
                        else:
                            st.success(f"## ✅ REAL VIDEO")
                            confidence = 1 - avg_score
                            
                        st.write(f"Confidence: **{confidence*100:.2f}%**")
                        st.progress(float(avg_score)) # 0 = Real, 1 = Fake
                        
                        st.write("Frame Predictions (0=Real, 1=Fake):")
                        st.bar_chart(predictions)

        # Cleanup
        try:
            os.remove(video_path)
        except:
            pass

if __name__ == "__main__":
    main()

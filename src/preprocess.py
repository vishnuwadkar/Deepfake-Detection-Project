import os
import cv2
from mtcnn import MTCNN
import sys
from pathlib import Path
from tqdm import tqdm

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REAL_RAW = RAW_DIR / "real"
FAKE_RAW = RAW_DIR / "fake"
REAL_PROCESSED = PROCESSED_DIR / "real"
FAKE_PROCESSED = PROCESSED_DIR / "fake"

# Ensure processed directories exist
REAL_PROCESSED.mkdir(parents=True, exist_ok=True)
FAKE_PROCESSED.mkdir(parents=True, exist_ok=True)

# Initialize MTCNN detector
detector = MTCNN()

def process_frame(frame, output_dir, filename_prefix, frame_idx):
    """Detects faces in a frame, crops, resizes, and saves them."""
    try:
        # MTCNN expects RGB but OpenCV reads BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            # Add padding if needed, but for now simple crop
            # Ensure coordinates are within frame bounds
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

            # Crop face
            face_img = frame[y:y+h, x:x+w]
            
            # Resize to 128x128
            try:
                face_resized = cv2.resize(face_img, (128, 128))
            except cv2.error:
                continue # Skip if resize fails (e.g. empty image)

            # Save
            output_filename = f"{filename_prefix}_frame{frame_idx}_face{i}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), face_resized)
            
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")

def process_video(video_path, output_dir, label):
    """Extracts frames from video (1 per second) and processes them."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 # Fallback
        
        frame_interval = int(fps) # 1 frame per second
        current_frame = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_interval == 0:
                process_frame(frame, output_dir, video_path.stem, current_frame)
                saved_count += 1

            current_frame += 1

        cap.release()
        print(f"Processed video: {video_path.name} ({saved_count} frames checked)")
        
    except Exception as e:
        print(f"Failed to process video {video_path}: {e}")

def process_image(image_path, output_dir, label):
    """Processes a single image file."""
    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Treat image as a single frame (frame_idx=0)
        process_frame(frame, output_dir, image_path.stem, 0)
        
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")

def main():
    print("Starting preprocessing...")
    print(f"Raw Data: {RAW_DIR}")
    print(f"Processed Data: {PROCESSED_DIR}")

    # Limit for testing/demonstration if needed, but running full for now
    # Check Real folder
    print(f"Processing REAL data from {REAL_RAW}...")
    if REAL_RAW.exists():
        files = list(REAL_RAW.iterdir())
        print(f"Found {len(files)} files in real folder.")
        for file_path in tqdm(files, desc="Processing Real"):
            if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                process_video(file_path, REAL_PROCESSED, "real")
            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                process_image(file_path, REAL_PROCESSED, "real")
    else:
        print(f"Directory not found: {REAL_RAW}")

    # Check Fake folder
    print(f"Processing FAKE data from {FAKE_RAW}...")
    if FAKE_RAW.exists():
        files = list(FAKE_RAW.iterdir())
        print(f"Found {len(files)} files in fake folder.")
        for file_path in tqdm(files, desc="Processing Fake"):
            if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                process_video(file_path, FAKE_PROCESSED, "fake")
            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                process_image(file_path, FAKE_PROCESSED, "fake")
    else:
        print(f"Directory not found: {FAKE_RAW}")

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()

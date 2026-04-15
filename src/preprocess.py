"""
preprocess.py -- Data preprocessing for DeepGuard AI.

For the 140K Real-and-Fake-Faces dataset:
  The dataset already contains clean face images at 256×256.
  This script handles:
    1. Resizing images to IMG_SIZE (224×224) for EfficientNetV2
    2. Verifying image integrity (remove corrupted files)
    3. Computing dataset statistics

For custom datasets with videos/raw images:
  Use the --raw flag to run the original MTCNN-based face extraction pipeline.

Run:
    python src/preprocess.py            # For 140K dataset (resize only)
    python src/preprocess.py --raw      # For raw video/image datasets
"""

import os
import sys
import argparse
import hashlib
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -- Import shared config ----------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    REAL_RAW, FAKE_RAW,
    REAL_PROCESSED, FAKE_PROCESSED,
    IMG_SIZE, FACE_PADDING, FRAME_STEP,
    DATA_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR,
)


# -----------------------------------------------------------------------------
# Mode 1: Verify & Resize (for 140K dataset -- images already extracted)
# -----------------------------------------------------------------------------

def verify_and_resize():
    """
    Verifies image integrity and optionally resizes images.
    For the 140K dataset, images are already 256×256 face crops.
    TensorFlow handles resizing to 224×224 during data loading,
    so this step mainly validates dataset health.
    """
    print("=" * 60)
    print("  DeepGuard AI -- Dataset Verification")
    print("=" * 60)
    print(f"  IMG_SIZE: {IMG_SIZE}")
    print(f"  Checking: {DATA_DIR}")
    print("=" * 60)

    total_files = 0
    corrupted = 0
    stats = {}

    # Check all splits and classes
    splits_to_check = []
    if TRAIN_DIR.exists():
        splits_to_check = [("train", TRAIN_DIR), ("valid", VALID_DIR), ("test", TEST_DIR)]
    elif DATA_DIR.exists():
        splits_to_check = [("all", DATA_DIR)]

    for split_name, split_dir in splits_to_check:
        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            files = list(class_dir.glob("*.*"))
            valid_count = 0
            
            for img_path in tqdm(files, desc=f"Verifying {split_name}/{class_name}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"  [CORRUPT] {img_path.name}")
                        corrupted += 1
                        img_path.unlink()  # Remove corrupted file
                        continue
                    valid_count += 1
                    total_files += 1
                except Exception as e:
                    print(f"  [ERROR] {img_path.name}: {e}")
                    corrupted += 1

            key = f"{split_name}/{class_name}"
            stats[key] = valid_count

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    for key, count in stats.items():
        print(f"  {key:20s}: {count:>6,} valid images")
    print(f"  {'TOTAL':20s}: {total_files:>6,}")
    if corrupted > 0:
        print(f"  {'REMOVED CORRUPT':20s}: {corrupted:>6,}")
    print(f"{'=' * 60}")

    if total_files > 0:
        # Check class balance
        class_counts = {}
        for key, count in stats.items():
            cls = key.split("/")[-1]
            class_counts[cls] = class_counts.get(cls, 0) + count

        if len(class_counts) >= 2:
            counts = list(class_counts.values())
            ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            print(f"\n  Class balance: {class_counts}")
            if ratio > 1.5:
                print(f"  [WARN] Imbalance ratio: {ratio:.1f}x -- class weights will be applied")
            else:
                print(f"  [OK] Classes are well-balanced (ratio: {ratio:.2f}x)")


# -----------------------------------------------------------------------------
# Mode 2: Raw Face Extraction (for custom video/image datasets)
# -----------------------------------------------------------------------------

def get_detector():
    """Lazy-loads MTCNN face detector."""
    global _detector
    if "_detector" not in globals() or _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector

_detector = None


def _frame_hash(frame: np.ndarray) -> str:
    """Returns a compact MD5 hash for duplicate detection."""
    small = cv2.resize(frame, (16, 16))
    return hashlib.md5(small.tobytes()).hexdigest()


def _crop_face_with_padding(frame, bbox, padding=FACE_PADDING):
    """Crops a face region with proportional padding."""
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def process_frame(frame, output_dir, prefix, frame_idx):
    """Detects faces and saves crops."""
    detector = get_detector()
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_faces(rgb_frame)

        if not detected_faces:
            return 0

        saved = 0
        for i, face_data in enumerate(detected_faces):
            if face_data['confidence'] < 0.90:
                continue
            cropped = _crop_face_with_padding(frame, face_data['box'])
            if cropped is None:
                continue
            try:
                resized = cv2.resize(cropped, IMG_SIZE)
            except cv2.error:
                continue

            filename = f"{prefix}_f{frame_idx}_face{i}.png"
            cv2.imwrite(str(output_dir / filename), resized)
            saved += 1
            if saved >= 2:
                break
        return saved
    except Exception as e:
        print(f"  [WARN] Frame {frame_idx}: {e}")
        return 0


def process_video(video_path, output_dir):
    """Extracts faces from a video."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        seen = set()
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % FRAME_STEP == 0:
                fhash = _frame_hash(frame)
                if fhash not in seen:
                    seen.add(fhash)
                    saved += process_frame(frame, output_dir, video_path.stem, idx)
            idx += 1

        cap.release()
        print(f"  [OK] {video_path.name}: {saved} faces from {total_frames} frames")
    except Exception as e:
        print(f"  [ERROR] {video_path.name}: {e}")


def process_raw():
    """Process raw videos/images from data/raw into data/processed."""
    print("=" * 60)
    print("  DeepGuard AI -- Raw Data Preprocessing (MTCNN)")
    print("=" * 60)

    REAL_PROCESSED.mkdir(parents=True, exist_ok=True)
    FAKE_PROCESSED.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for label, src_dir, out_dir in [
        ("real", REAL_RAW, REAL_PROCESSED),
        ("fake", FAKE_RAW, FAKE_PROCESSED),
    ]:
        if not src_dir.exists():
            print(f"  [WARN] {src_dir} not found, skipping")
            continue

        files = sorted(src_dir.iterdir())
        print(f"\n[{label.upper()}] {len(files)} files in {src_dir}")

        for f in tqdm(files, desc=f"Processing {label}"):
            ext = f.suffix.lower()
            if ext in video_exts:
                process_video(f, out_dir)
            elif ext in image_exts:
                frame = cv2.imread(str(f))
                if frame is not None:
                    process_frame(frame, out_dir, f.stem, 0)

    # Print summary
    real_count = len(list(REAL_PROCESSED.glob("*.png")))
    fake_count = len(list(FAKE_PROCESSED.glob("*.png")))
    print(f"\n[OK] Preprocessing complete!")
    print(f"  Real faces : {real_count:,}")
    print(f"  Fake faces : {fake_count:,}")
    print(f"  Total      : {real_count + fake_count:,}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepGuard AI Preprocessor")
    parser.add_argument("--raw", action="store_true",
                        help="Run MTCNN face extraction on raw videos/images")
    args = parser.parse_args()

    if args.raw:
        process_raw()
    else:
        verify_and_resize()


if __name__ == "__main__":
    main()

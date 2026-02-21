"""
preprocess.py — Optimized face extraction pipeline.

Improvements over original:
  - 30% face padding for more context around face edges
  - Frame sampling every FRAME_STEP frames (~3 fps at 30 fps)
  - High-quality JPEG save (JPEG_QUALITY=95) to reduce compression artifacts
  - Lazy MTCNN initialization (avoids re-init on import)
  - Perceptual hash-based near-duplicate frame skipping
  - Resize output to IMG_SIZE (224x224) to match training config
"""

import os
import sys
import hashlib
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Import shared config ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    REAL_RAW, FAKE_RAW,
    REAL_PROCESSED, FAKE_PROCESSED,
    IMG_SIZE, FACE_PADDING, FRAME_STEP, JPEG_QUALITY,
)

# Ensure output directories exist
REAL_PROCESSED.mkdir(parents=True, exist_ok=True)
FAKE_PROCESSED.mkdir(parents=True, exist_ok=True)


def get_mtcnn():
    """Lazy-loads MTCNN on first call so module import stays fast."""
    global _detector
    if "_detector" not in globals() or _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector


_detector = None  # Module-level sentinel


def _frame_hash(frame: np.ndarray) -> str:
    """Returns a compact MD5 hash of a downsampled frame for duplicate detection."""
    small = cv2.resize(frame, (16, 16))
    return hashlib.md5(small.tobytes()).hexdigest()


def _crop_face_with_padding(frame: np.ndarray, box: tuple, padding: float = FACE_PADDING):
    """
    Crops a face region from `frame` with proportional padding.

    Args:
        frame:   BGR image (H, W, 3).
        box:     MTCNN bounding box (x, y, w, h).
        padding: Fractional padding around each side (default 30%).

    Returns:
        Cropped BGR face image, or None if the crop is degenerate.
    """
    x, y, w, h = box
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def process_frame(frame: np.ndarray, output_dir: Path, prefix: str, frame_idx: int):
    """
    Detects faces in a frame, crops with padding, resizes, and saves.

    Args:
        frame:      BGR image from OpenCV.
        output_dir: Directory to save cropped face images.
        prefix:     Filename prefix (usually video stem).
        frame_idx:  Current frame number (for unique filenames).
    """
    detector = get_mtcnn()
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        if not faces:
            return  # No face detected — skip frame silently

        for i, face in enumerate(faces):
            cropped = _crop_face_with_padding(frame, face["box"])
            if cropped is None:
                continue

            try:
                face_resized = cv2.resize(cropped, IMG_SIZE)
            except cv2.error:
                continue

            filename = f"{prefix}_f{frame_idx}_face{i}.jpg"
            cv2.imwrite(
                str(output_dir / filename),
                face_resized,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

    except Exception as exc:
        print(f"  [WARN] Error processing frame {frame_idx}: {exc}")


def process_video(video_path: Path, output_dir: Path):
    """
    Extracts faces from a video, sampling every FRAME_STEP frames.
    Near-duplicate frames (same perceptual hash) are skipped.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open video: {video_path.name}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_count = 0
        seen_hashes: set = set()
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % FRAME_STEP == 0:
                fhash = _frame_hash(frame)
                if fhash not in seen_hashes:
                    seen_hashes.add(fhash)
                    process_frame(frame, output_dir, video_path.stem, current_frame)
                    saved_count += 1

            current_frame += 1

        cap.release()
        print(f"  ✓ {video_path.name}: {saved_count} unique frames processed "
              f"(of {total_frames} total)")

    except Exception as exc:
        print(f"  [ERROR] Failed to process video {video_path.name}: {exc}")


def process_image(image_path: Path, output_dir: Path):
    """Processes a single image file (treated as a single frame)."""
    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  [WARN] Could not read image: {image_path.name}")
            return
        process_frame(frame, output_dir, image_path.stem, 0)
    except Exception as exc:
        print(f"  [ERROR] Failed to process image {image_path.name}: {exc}")


def process_directory(source_dir: Path, output_dir: Path, label: str):
    """Processes all videos and images in a directory."""
    if not source_dir.exists():
        print(f"[WARN] Directory not found: {source_dir}")
        return

    files = sorted(source_dir.iterdir())
    print(f"\n[{label.upper()}] {len(files)} files found in {source_dir}")

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for file_path in tqdm(files, desc=f"Processing {label}"):
        ext = file_path.suffix.lower()
        if ext in video_exts:
            process_video(file_path, output_dir)
        elif ext in image_exts:
            process_image(file_path, output_dir)


def main():
    print("=" * 60)
    print("  Deepfake Detection — Preprocessing Pipeline")
    print("=" * 60)
    print(f"  IMG_SIZE     : {IMG_SIZE}")
    print(f"  FACE_PADDING : {FACE_PADDING * 100:.0f}%")
    print(f"  FRAME_STEP   : every {FRAME_STEP} frames")
    print(f"  JPEG_QUALITY : {JPEG_QUALITY}")
    print("=" * 60)

    process_directory(REAL_RAW, REAL_PROCESSED, "real")
    process_directory(FAKE_RAW, FAKE_PROCESSED, "fake")

    real_count = len(list(REAL_PROCESSED.glob("*.jpg")))
    fake_count = len(list(FAKE_PROCESSED.glob("*.jpg")))
    print(f"\n✅ Preprocessing complete!")
    print(f"   Real faces : {real_count}")
    print(f"   Fake faces : {fake_count}")
    print(f"   Total      : {real_count + fake_count}")

    if real_count > 0 and fake_count > 0:
        ratio = max(real_count, fake_count) / min(real_count, fake_count)
        if ratio > 1.5:
            print(f"\n⚠️  Class imbalance ratio: {ratio:.1f}x — "
                  "train.py will apply class weights automatically.")


if __name__ == "__main__":
    main()

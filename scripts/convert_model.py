"""
convert_model.py -- Convert trained Keras model to TensorFlow.js format.

This script:
  1. Loads the trained EfficientNetV2B0 + CBAM model
  2. Exports as a TensorFlow SavedModel
  3. Converts to TF.js Graph Model format
  4. Applies float16 quantization to reduce size (~50%)

Usage:
    pip install tensorflowjs
    python scripts/convert_model.py

Output:
    extension/model/model.json + weight shard .bin files
"""

import os
import sys
import shutil
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

def convert():
    import tensorflow as tf
    from src.config import MODEL_PATH, MODEL_PATH_H5, IMG_SIZE
    from src.model import CUSTOM_OBJECTS

    print("=" * 60)
    print("  DeepGuard AI -- Model Conversion Pipeline")
    print("=" * 60)

    # -- Step 1: Load Model ------------------------------------------------
    print("\n[1/4] Loading trained model...")
    
    from src.config import MODELS_DIR
    from src.model import build_model as _build_model

    custom_objects = {**CUSTOM_OBJECTS, "tf": tf}
    model = None

    candidates = [MODEL_PATH, MODELS_DIR / "deepfake_detector.h5", MODEL_PATH_H5]
    
    for target in candidates:
        if not target.exists():
            continue
        
        # Attempt 1: Full model load
        try:
            model = tf.keras.models.load_model(
                str(target), custom_objects=custom_objects, compile=False
            )
            print(f"  [OK] Loaded full model from: {target.name}")
            break
        except Exception as e1:
            print(f"  [INFO] Full load failed for {target.name}, trying weights-only...")
        
        # Attempt 2: Build architecture fresh + load weights only
        # This handles cross-TF-version .h5 files (e.g. TF 2.13 Mac -> TF 2.16 Windows)
        try:
            model = _build_model(trainable_base=True)
            model.load_weights(str(target))
            print(f"  [OK] Loaded weights from: {target.name} (architecture rebuilt locally)")
            break
        except Exception as e2:
            print(f"  [WARN] Weights load also failed for {target.name}: {e2}")
            model = None
            continue

    if model is None:
        print("[ERROR] No trained model found. Run train.py first.")
        return

    # -- Step 2: Export as SavedModel --------------------------------------
    print("\n[2/4] Exporting as TensorFlow SavedModel...")
    
    saved_model_dir = PROJECT_ROOT / "models" / "saved_model_temp"
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))

    model.export(str(saved_model_dir))
    print(f"  [OK] SavedModel exported to: {saved_model_dir}")

    # -- Step 3: Convert to TF.js -----------------------------------------
    print("\n[3/4] Converting to TensorFlow.js format (with float16 quantization)...")
    
    output_dir = PROJECT_ROOT / "extension" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing model files
    for f in output_dir.glob("*"):
        f.unlink()

    try:
        import tensorflowjs as tfjs
        
        tfjs.converters.convert_tf_saved_model(
            str(saved_model_dir),
            str(output_dir),
            quantization_dtype_map={
                tf.float32: tf.float16  # Quantize all float32 → float16
            },
        )
        print(f"  [OK] TF.js model saved to: {output_dir}")
    except ImportError:
        print("  [ERROR] tensorflowjs not installed. Install with:")
        print("    pip install tensorflowjs")
        print("\n  Attempting command-line conversion instead...")
        
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--quantize_float16=*",
            str(saved_model_dir),
            str(output_dir),
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  [ERROR] Conversion failed:\n{result.stderr}")
            return
        else:
            print(f"  [OK] TF.js model saved to: {output_dir}")

    # -- Step 4: Validate & Report -----------------------------------------
    print("\n[4/4] Validation report...")
    
    model_json = output_dir / "model.json"
    if not model_json.exists():
        print("  [ERROR] model.json not found -- conversion may have failed.")
        return
    
    # Calculate total size
    total_bytes = sum(f.stat().st_size for f in output_dir.iterdir())
    total_mb = total_bytes / (1024 * 1024)
    
    original_bytes = 0
    for target in (MODEL_PATH, MODEL_PATH_H5):
        if target.exists():
            original_bytes = target.stat().st_size
            break
    original_mb = original_bytes / (1024 * 1024)
    
    shard_files = list(output_dir.glob("*.bin"))
    
    print(f"  Original model size : {original_mb:.1f} MB")
    print(f"  TF.js model size    : {total_mb:.1f} MB")
    print(f"  Compression ratio   : {original_mb/total_mb:.1f}x")
    print(f"  Weight shards       : {len(shard_files)}")
    print(f"  Files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size:.2f} MB)")

    # -- Cleanup -----------------------------------------------------------
    print("\n  Cleaning up temporary SavedModel...")
    shutil.rmtree(str(saved_model_dir), ignore_errors=True)

    print("\n" + "=" * 60)
    print("  [OK] Conversion complete!")
    print(f"  Model ready at: {output_dir}")
    print("=" * 60)

    # -- Optional: Verify numerical accuracy ------------------------------
    print("\n[OPTIONAL] Running numerical verification...")
    try:
        import json
        # Create test input
        test_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32) * 255.0
        
        # Python prediction
        python_pred = model.predict(test_input, verbose=0)[0][0]
        print(f"  Python model prediction: {python_pred:.6f}")
        print("  [INFO]  Compare this with TF.js output in the browser to verify accuracy.")
        
        # Save test input for browser verification
        test_path = output_dir / "test_input.json"
        test_data = {
            "shape": list(test_input.shape),
            "python_prediction": float(python_pred),
        }
        with open(str(test_path), 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"  Test data saved to: {test_path}")
        
    except Exception as e:
        print(f"  [WARN] Verification skipped: {e}")


if __name__ == "__main__":
    convert()

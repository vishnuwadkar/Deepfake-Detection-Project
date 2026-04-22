"""
Minimal TF.js converter — monkey-patches ALL broken dependencies
before importing tensorflowjs. Works around TF 2.20 + Python 3.12
compatibility issues on Windows.
"""
import sys
import types
import os

# =====================================================================
# STEP 1: Stub out ALL problematic modules BEFORE any tensorflowjs import
# =====================================================================

# Stub tensorflow_decision_forests
for mod_name in [
    'tensorflow_decision_forests',
    'tensorflow_decision_forests.keras',
    'tensorflow_decision_forests.keras.core',
]:
    sys.modules[mod_name] = types.ModuleType(mod_name)

# Stub tensorflow_hub (broken with TF 2.20 — missing tf.compat.v1.estimator)
hub_stub = types.ModuleType('tensorflow_hub')
hub_stub.KerasLayer = type('KerasLayer', (), {})  # dummy class
hub_stub.load = lambda *a, **k: None
hub_stub.resolve = lambda *a, **k: ""
for sub in [
    'tensorflow_hub.estimator',
    'tensorflow_hub.feature_column',
    'tensorflow_hub.keras_layer',
    'tensorflow_hub.module',
]:
    sys.modules[sub] = types.ModuleType(sub)
sys.modules['tensorflow_hub'] = hub_stub

# =====================================================================
# STEP 2: Now import tensorflowjs (all problematic imports are stubbed)
# =====================================================================

import tensorflow as tf
from tensorflowjs.converters import tf_saved_model_conversion_v2 as conv_v2

# =====================================================================
# STEP 3: Convert SavedModel → TF.js
# =====================================================================

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJ, 'models', 'saved_model_export')
OUTPUT_DIR = os.path.join(PROJ, 'extension', 'model')

if not os.path.exists(INPUT_DIR):
    print(f"[ERROR] SavedModel not found at: {INPUT_DIR}")
    print("Run the main convert_model.py first to export the SavedModel.")
    sys.exit(1)

# Clear output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
for f in os.listdir(OUTPUT_DIR):
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fp):
        os.remove(fp)

print("=" * 60)
print("  DeepGuard AI — TF.js Conversion (Minimal)")
print("=" * 60)
print(f"\n  Input:  {INPUT_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Quantization: float16\n")

conv_v2.convert_tf_saved_model(
    INPUT_DIR,
    OUTPUT_DIR,
)

print("\n" + "=" * 60)
print("  CONVERSION COMPLETE!")
print("=" * 60)
total_kb = 0
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    total_kb += size_kb
    print(f"  {f:40s} {size_kb:8.1f} KB")
print(f"  {'TOTAL':40s} {total_kb:8.1f} KB ({total_kb/1024:.1f} MB)")
print("\nReload the extension in chrome://extensions to use the new model.")

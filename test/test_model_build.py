"""
test_model_build.py — Quick smoke test for model architecture.
Verifies build_model() produces correct input/output shapes.
Run: python test/test_model_build.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model
from src.config import IMG_SIZE

def test_model_build():
    print("Building model (trainable_base=False)...")
    model = build_model(trainable_base=False)

    input_shape  = tuple(model.input.shape[1:])
    output_shape = tuple(model.output.shape[1:])

    expected_input  = IMG_SIZE + (3,)
    expected_output = (1,)

    print(f"  Input shape  : {input_shape}  (expected {expected_input})")
    print(f"  Output shape : {output_shape}  (expected {expected_output})")

    assert input_shape == expected_input, (
        f"Input shape mismatch: {input_shape} != {expected_input}"
    )
    assert output_shape == expected_output, (
        f"Output shape mismatch: {output_shape} != {expected_output}"
    )

    total_params     = model.count_params()
    trainable_params = sum(
        p.numpy().size for p in model.trainable_weights
    )
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    print("\n✅ Model build test PASSED.")


if __name__ == "__main__":
    test_model_build()

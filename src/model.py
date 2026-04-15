"""
model.py — Shared model architecture for training and inference.

EfficientNetV2B0 + CBAM Attention + optimized classification head.
Uses serializable custom Keras layers for clean save/load.
Optimized for AI-generated face detection (95%+ accuracy target).
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dropout, Reshape, Add, Activation, Multiply, Conv2D, Concatenate,
    BatchNormalization, Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

from src.config import (
    IMG_SIZE, CBAM_RATIO, LR_HEAD,
    LABEL_SMOOTHING, DROPOUT_HEAD, DROPOUT_TAIL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Serializable Custom Layers (for CBAM — safe to save/load)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAvgPool(Layer):
    """Reduces mean across the channel axis (axis=-1), keepdims=True."""
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


class ChannelMaxPool(Layer):
    """Reduces max across the channel axis (axis=-1), keepdims=True."""
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


# Registry of custom objects needed for model saving/loading
CUSTOM_OBJECTS = {
    "ChannelAvgPool": ChannelAvgPool,
    "ChannelMaxPool": ChannelMaxPool,
}


# ─────────────────────────────────────────────────────────────────────────────
# CBAM Block
# ─────────────────────────────────────────────────────────────────────────────

def cbam_block(input_tensor, ratio: int = CBAM_RATIO):
    """
    Convolutional Block Attention Module (CBAM).
    Uses serializable custom layers instead of Lambda — safe to save/load.
    """
    channel = input_tensor.shape[-1]

    # ── Channel Attention ────────────────────────────────────────────────
    shared_dense_1 = Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_dense_2 = Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )

    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_out  = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_out  = shared_dense_2(shared_dense_1(max_pool))

    channel_out = Activation("sigmoid")(Add()([avg_out, max_out]))
    channel_out = Reshape((1, 1, channel))(channel_out)
    channel_refined = Multiply()([input_tensor, channel_out])

    # ── Spatial Attention ────────────────────────────────────────────────
    avg_spatial = ChannelAvgPool()(channel_refined)
    max_spatial = ChannelMaxPool()(channel_refined)

    spatial = Concatenate(axis=-1)([avg_spatial, max_spatial])
    spatial = Conv2D(
        1, (7, 7),
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(spatial)

    return Multiply()([channel_refined, spatial])


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(trainable_base: bool = False) -> Model:
    """
    Builds EfficientNetV2B0 + CBAM + optimized classification head.

    Architecture optimizations for 95%+ accuracy:
      - EfficientNetV2B0 backbone with internal preprocessing
      - CBAM attention for better feature focus
      - Deeper head: 512 → 256 with BN + Dropout
      - Label smoothing via BinaryFocalCrossentropy
      - He-normal initialization throughout

    Args:
        trainable_base: If True the entire backbone is trainable.
                        Keep False for Phase-1, set True for fine-tuning.
    Returns:
        Compiled Keras Model.
    """
    base = EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        include_preprocessing=True  # Internal preprocessing (handles [0,255] → normalized)
    )
    base.trainable = trainable_base

    x = base.output
    x = cbam_block(x)

    x = GlobalAveragePooling2D()(x)

    # Dense block 1
    x = Dense(512, kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_HEAD)(x)

    # Dense block 2
    x = Dense(256, kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_TAIL)(x)

    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LR_HEAD),
        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_layers(model: Model, n_layers: int, new_lr: float) -> Model:
    """
    Unfreezes the top `n_layers` of the backbone for fine-tuning
    and recompiles with a lower learning rate + cosine decay.
    """
    base = model.layers[1]   # Backbone sub-model is always index 1
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    # Count trainable params for logging
    trainable_count = sum(1 for layer in base.layers if layer.trainable)
    print(f"  Unfroze {trainable_count} layers (top {n_layers} requested)")

    model.compile(
        optimizer=Adam(learning_rate=new_lr),
        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model

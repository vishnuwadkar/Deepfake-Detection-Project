"""
model.py — Shared model architecture for training and inference.

Replaces Lambda layers (which can't be serialized) with proper
custom Keras Layer subclasses that save/load cleanly.
"""

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input  # noqa: F401
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dropout, Reshape, Add, Activation, Multiply, Conv2D, Concatenate,
    BatchNormalization, Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

from src.config import IMG_SIZE, CBAM_RATIO, LR_HEAD


# ─────────────────────────────────────────────────────────────────────────────
# Serializable custom layers to replace Lambda (which cannot be saved/loaded)
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
    avg_spatial = ChannelAvgPool()(channel_refined)   # replaces Lambda
    max_spatial = ChannelMaxPool()(channel_refined)   # replaces Lambda

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
    Builds Xception + CBAM + custom classification head.

    Args:
        trainable_base: If True the entire Xception backbone is trainable.
                        Keep False for Phase-1, set True for fine-tuning.
    Returns:
        Compiled Keras Model.
    """
    base = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,),
    )
    base.trainable = trainable_base

    x = base.output
    x = cbam_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=LR_HEAD),
        loss="binary_crossentropy",
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
    Unfreezes the top `n_layers` of the Xception backbone for fine-tuning
    and recompiles with a lower learning rate.
    """
    base = model.layers[1]   # Xception sub-model is always index 1
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=new_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model

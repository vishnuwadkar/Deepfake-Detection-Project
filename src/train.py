import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input,
    GlobalMaxPooling2D, Reshape, Add, Activation, Multiply,
    Conv2D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/deepfake_detector_cbam.h5")

# Ensure models directory exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def cutout(img):
    """
    Applies Cutout augmentation: Randomly masks a 32x32 pixel square with gray.
    Input img is expected to be 0-255 (ImageDataGenerator default before rescale).
    """
    h, w, c = img.shape
    mask_size = 32
    
    # Center of the mask
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    
    # Bounding box
    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)
    
    # Fill with gray (127.5 for 0-255 range)
    img[y1:y2, x1:x2, :] = 127.5
    return img

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
    print("Building Xception model with CBAM...")
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

def train():
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return

    print("Setting up data generators with Cutout...")
    # Data Augmentation for training with Cutout
    train_datagen = ImageDataGenerator(
        preprocessing_function=cutout, # Apply Cutout
        rescale=1./255, # Apply rescale after Cutout
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 
    )

    # Train Generator
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    # Validation Generator (No augmentations/cutout, just rescale)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    if train_generator.samples == 0:
        print("No training data found. Check data/processed.")
        return

    # Build and Train
    model = build_model()
    model.summary()

    print(f"Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save Model
    print(f"Saving upgraded model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Training complete!")

if __name__ == "__main__":
    train()

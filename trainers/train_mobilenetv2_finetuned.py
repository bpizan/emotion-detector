import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# === Configuration ===
BASE_DIR             = 'fer2013'
TRAIN_DIR            = os.path.join(BASE_DIR, 'train')
VAL_DIR              = os.path.join(BASE_DIR, 'val')
IMG_SIZE             = 224
BATCH_SIZE           = 32
INITIAL_EPOCHS       = 10
FINE_TUNE_EPOCHS     = 5
FINE_TUNE_LR         = 1e-5
BASE_MODEL_PATH      = 'mobilenetv2_emotion_model.h5'
FINE_TUNE_MODEL_PATH = 'mobilenetv2_emotion_finetuned.h5'
PLOT_DIR             = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load & Prep Model for Fine-Tuning ===
model = load_model(BASE_MODEL_PATH)

# Freeze all layers, then unfreeze the last 20
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Data Generators ===
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === Fine-Tuning ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS,
    initial_epoch=0
)

# === Save Fine-Tuned Model ===
model.save(FINE_TUNE_MODEL_PATH)

# === Plot & Save Fine-Tuning Accuracy Curve ===
plt.figure()
epochs_range = range(1, FINE_TUNE_EPOCHS + 1)
plt.plot(epochs_range, history.history['accuracy'],    label='Fine-tune Train Acc')
plt.plot(epochs_range, history.history['val_accuracy'],label='Fine-tune Val Acc')
plt.title('MobileNetV2 Fine-Tuning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, 'mobilenetv2_fine_tune_accuracy.png')
plt.savefig(plot_path)
plt.close()

print(f"✅ Saved fine-tuned model to '{FINE_TUNE_MODEL_PATH}' and plot to '{plot_path}'.")

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# === Configuration ===
BASE_DIR       = 'fer2013'
TRAIN_DIR      = os.path.join(BASE_DIR, 'train')
VAL_DIR        = os.path.join(BASE_DIR, 'val')
IMG_SIZE       = 224   # for MobileNetV2
BATCH_SIZE     = 32
EPOCHS         = 10
MODEL_PATH     = 'mobilenetv2_emotion_model.h5'
PLOT_DIR       = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

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

# === Model Setup ===
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_layer')
)
base_model.trainable = False  # freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D(name='gap')(x)
x = Dropout(0.5, name='dropout')(x)
outputs = Dense(train_data.num_classes, activation='softmax', name='output')(x)

model = Model(inputs=base_model.input, outputs=outputs, name='MobileNetV2_CNN')
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Training ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# === Save Model ===
model.save(MODEL_PATH)

# === Plot & Save Accuracy Curve ===
plt.figure()
plt.plot(history.history['accuracy'],    label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Val Accuracy')
plt.title('MobileNetV2 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, 'mobilenetv2_accuracy.png')
plt.savefig(plot_path)
plt.close()

print(f"✅ Saved model to '{MODEL_PATH}' and plot to '{plot_path}'.")

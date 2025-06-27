import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === 1. Configuration ===
BASE_DIR      = 'fer2013'
TRAIN_DIR     = os.path.join(BASE_DIR, 'train')
VAL_DIR       = os.path.join(BASE_DIR, 'val')
IMG_HEIGHT    = 48
IMG_WIDTH     = 48
BATCH_SIZE    = 32
EPOCHS        = 15
MODEL_PATH    = 'cnn_emotion_model.h5'
PLOT_DIR      = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# === 2. Data Generators ===
train_gen = ImageDataGenerator(rescale=1./255)
val_gen   = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === 3. Model Definition ===
model = Sequential(name='EmotionCNN')
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='conv1'))
model.add(MaxPooling2D((2, 2), name='pool1'))
model.add(Dropout(0.25, name='drop1'))

model.add(Conv2D(64, (3, 3), activation='relu', name='conv2'))
model.add(MaxPooling2D((2, 2), name='pool2'))
model.add(Dropout(0.25, name='drop2'))

model.add(Conv2D(128, (3, 3), activation='relu', name='conv3'))
model.add(MaxPooling2D((2, 2), name='pool3'))
model.add(Dropout(0.25, name='drop3'))

model.add(Flatten(name='flatten'))
model.add(Dense(128, activation='relu', name='dense1'))
model.add(Dropout(0.5, name='drop4'))
model.add(Dense(train_data.num_classes, activation='softmax', name='output'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === 4. Train ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# === 5. Save Model ===
model.save(MODEL_PATH)

# === 6. Plot & Save Accuracy Curve ===
plt.figure()
plt.plot(history.history['accuracy'],    label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Custom CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, 'cnn_accuracy.png')
plt.savefig(plot_path)
plt.close()
print(f"✅ Model saved to '{MODEL_PATH}' and accuracy plot saved to '{plot_path}'.")

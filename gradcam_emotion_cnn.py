import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

# === Configuration ===
BASE_DIR         = 'fer2013'
VAL_DIR          = os.path.join(BASE_DIR, 'val')
IMG_HEIGHT       = 48
IMG_WIDTH        = 48
BATCH_SIZE       = 1
WEIGHTS_PATH     = 'cnn_emotion_model.h5'
OUTPUT_DIR       = 'gradcam_outputs_cnn'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Rebuild and load the custom CNN (Functional API) ===
inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='input_layer')
x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inp)
x = MaxPooling2D((2, 2), name='pool1')(x)
x = Dropout(0.25, name='drop1')(x)
x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
x = MaxPooling2D((2, 2), name='pool2')(x)
x = Dropout(0.25, name='drop2')(x)
x = Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
x = MaxPooling2D((2, 2), name='pool3')(x)
x = Dropout(0.25, name='drop3')(x)
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='dense1')(x)
x = Dropout(0.5, name='drop4')(x)
out = Dense(7, activation='softmax', name='output_layer')(x)

model = Model(inputs=inp, outputs=out, name='EmotionCNN')
model.load_weights(WEIGHTS_PATH)

# === 2. Prepare validation data ===
val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True
)
labels = list(val_data.class_indices.keys())

# === 3. Find the last Conv2D layer ===
last_conv = next(
    layer.name for layer in reversed(model.layers)
    if isinstance(layer, Conv2D)
)

# === 4. Grad-CAM function ===
def get_gradcam_heatmap(model, img_tensor, layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_outs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outs = conv_outs[0]
    cam = conv_outs @ weights[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return cam.numpy()

# === 5. Generate & save Grad-CAM images ===
for i in range(5):
    img, label = next(val_data)
    true_c = np.argmax(label[0])
    pred = model.predict(img)
    pred_c = np.argmax(pred[0])

    # Prepare overlay
    img_rgb = np.repeat(img[0], 3, axis=-1)
    heatmap = get_gradcam_heatmap(model, img, last_conv)
    heatmap = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(img_rgb*255), 0.6, heatmap_color, 0.4, 0)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"True: {labels[true_c]} | Pred: {labels[pred_c]}")
    axs[0].imshow(img[0].squeeze(), cmap='gray'); axs[0].set_title("Original"); axs[0].axis('off')
    axs[1].imshow(heatmap, cmap='jet');             axs[1].set_title("Grad-CAM");    axs[1].axis('off')
    axs[2].imshow(overlay);                         axs[2].set_title("Overlay");     axs[2].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{i+1}_true_{labels[true_c]}_pred_{labels[pred_c]}.png"
    )
    fig.savefig(out_path)
    plt.close(fig)

print(f"✅ CNN Grad-CAM images saved to '{OUTPUT_DIR}/'")

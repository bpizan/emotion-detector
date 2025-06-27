import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# === Configuration ===
BASE_DIR         = 'fer2013'
VAL_DIR          = os.path.join(BASE_DIR, 'val')
IMG_SIZE         = 224
BATCH_SIZE       = 1
MODEL_PATH       = 'mobilenetv2_emotion_finetuned.h5'
OUTPUT_DIR       = 'gradcam_outputs_mobilenet_finetuned'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1) Load model & find last conv layer ===
model = load_model(MODEL_PATH)
last_conv = next(
    layer.name for layer in reversed(model.layers)
    if isinstance(layer, tf.keras.layers.Conv2D)
)

# === 2) Prepare validation data ===
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
labels = list(val_data.class_indices.keys())

# === 3) Grad-CAM function ===
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
    grads   = tape.gradient(class_channel, conv_outs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outs = conv_outs[0]
    cam = conv_outs @ weights[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return cam.numpy()

# === 4) Generate & save visualizations ===
for i in range(5):
    img, label = next(val_data)
    true_c = np.argmax(label[0])
    pred   = model.predict(img)
    pred_c = np.argmax(pred[0])

    heatmap = get_gradcam_heatmap(model, img, last_conv)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    img_display = ((img[0] + 1) * 127.5).astype('uint8')
    overlay     = cv2.addWeighted(img_display, 0.6, heatmap_color, 0.4, 0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"True: {labels[true_c]} | Pred: {labels[pred_c]}")
    axs[0].imshow(img_display);        axs[0].set_title("Original");   axs[0].axis('off')
    axs[1].imshow(heatmap, cmap='jet');axs[1].set_title("Grad-CAM");    axs[1].axis('off')
    axs[2].imshow(overlay);            axs[2].set_title("Overlay");     axs[2].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{i+1}_true_{labels[true_c]}_pred_{labels[pred_c]}.png"
    )
    fig.savefig(out_path)
    plt.close(fig)

print(f"✅ Fine-tuned MobileNetV2 Grad-CAM images saved to '{OUTPUT_DIR}/'")

# Emotion Detection from Facial Images

This repository implements a facial emotion recognition pipeline using the **FER2013** dataset and TensorFlow/Keras. It contains:

- **Models** (saved weights)  
- **Trainer scripts** to train a custom CNN, a MobileNetV2 transfer-learned model, and its fine-tuned variant  
- **Grad-CAM scripts** to visualize where each model “looks” when predicting  
- **Output folders** with sample predictions and heatmaps  
- **Plots** of training and validation accuracy  
- An **analysis & conclusion** write-up

---

## 📂 Repository Structure

```text
.
├── models/
│   ├── cnn_emotion_model.h5
│   ├── mobilenetv2_emotion_model.h5
│   └── mobilenetv2_emotion_finetuned.h5
│
├── trainers/
│   ├── train_cnn.py
│   ├── train_mobilenetv2_emotion.py
│   └── train_mobilenetv2_fine_tuned.py
│
├── gradcam_scripts/
│   ├── gradcam_emotion_cnn.py
│   ├── gradcam_emotion_mobilenet.py
│   └── gradcam_emotion_mobilenet_finetuned.py
│
├── gradcam_outputs_cnn/                     # 5 sample PNGs from custom CNN
├── gradcam_outputs_mobilenet/               # 5 sample PNGs from vanilla MobileNetV2
├── gradcam_outputs_mobilenet_finetuned/     # 5 sample PNGs from fine-tuned MobileNetV2
│
├── plots/
│   ├── cnn_accuracy.png
│   ├── mobilenetv2_accuracy.png
│   └── mobilenetv2_fine_tune_accuracy.png
│
├── create_val_split.py      # Utility to split off a validation set from train/
├── analysis_and_conclusion.md
└── README.md

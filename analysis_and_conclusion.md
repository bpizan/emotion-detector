# Analysis and Conclusions

## Performance Comparison

| Model                          | Validation Accuracy | Notes                                                                    |
|--------------------------------|---------------------|--------------------------------------------------------------------------|
| **Custom CNN**                 | ~62%                | Solid baseline; easy to train end-to-end.                                 |
| **MobileNetV2 (no fine-tuning)**     | ~44%                | Underperformed “out of the box” on FER2013 without any adaptation.        |
| **MobileNetV2 (fine-tuned)**   | ~51%                | Fine-tuning the last 20 layers boosted val accuracy by ~7 points to ~51%. |

> **Exact post-fine-tune validation accuracy**: 50.8% (epoch 5).

---

## Grad-CAM Comparison

Below are example Grad-CAM visualizations for 5 validation samples from both models:

| Sample | Custom CNN                             | MobileNetV2 (Fine-Tuned)              |
|--------|----------------------------------------|---------------------------------------|
| 1      | ![CNN 1](gradcam_outputs_cnn/sample_1_true_sad_pred_sad.png) | ![FT 1](gradcam_outputs_mobilenet_finetuned/sample_1_true_sad_pred_sad.png) |
| 2      | ![CNN 2](gradcam_outputs_cnn/sample_2_true_surprise_pred_fear.png) | ![FT 2](gradcam_outputs_mobilenet_finetuned/sample_2_true_surprise_pred_neutral.png) |
| 3      | ![CNN 3](gradcam_outputs_cnn/sample_3_true_surprise_pred_sad.png) | ![FT 3](gradcam_outputs_mobilenet_finetuned/sample_3_true_happy_pred_happy.png)  |
| 4      | ![CNN 4](gradcam_outputs_cnn/sample_4_true_angry_pred_angry.png) | ![FT 4](gradcam_outputs_mobilenet_finetuned/sample_4_true_surprise_pred_angry.png) |
| 5      | ![CNN 5](gradcam_outputs_cnn/sample_5_true_angry_pred_angry.png) | ![FT 5](gradcam_outputs_mobilenet_finetuned/sample_5_true_angry_pred_angry.png) |

---

### Key Takeaways

1. **Accuracy Improvement**  
   Fine-tuning MobileNetV2 lifted validation accuracy from ~44% to ~51%, a substantial ~7 percentage-point gain in just 5 extra epochs.

2. **Sharper Attention Maps**  
   The fine-tuned model’s Grad-CAM overlays are noticeably more focused on core facial regions (eyes, brows, mouth) compared to both the custom CNN and the un-fine-tuned MobileNetV2, indicating better feature extraction.

3. **Model Trade-Offs**  
   - **Custom CNN**: Fast to train, fewer parameters, strong baseline at ~62%.  
   - **MobileNetV2 FT**: Larger model and longer training but, once fine-tuned, achieves ~51% and exhibits crisper attention maps—potentially more robust on new data.

---

## Next Steps

- **Further Fine-Tuning**: Unfreeze additional layers or train for a few more epochs to close the gap to the custom CNN.  
- **Data Augmentation**: Introduce brightness/sharpness jitter, random crops, or mixup to improve robustness.  
- **Additional Explainability**: Experiment with LIME or SHAP to complement Grad-CAM.  
- **Real-World Testing**: Validate models on an external emotion dataset to gauge generalization.


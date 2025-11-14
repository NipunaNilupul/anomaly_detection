# Real-Time Anomaly Scoring for Industrial Pass/Fail Control
*Masters Project - In-Progress (NipunaNilupul)*

This project aims to develop a high-performance, real-time anomaly detection system. The current goal is to find a model that achieves high scores for both image-level (Pass/Fail) and pixel-level (Defect Localization) tasks.

---

## 📈 Current Status & Experimental Results

We are currently testing state-of-the-art (SOTA) feature-level detection models on the MVTec AD dataset.

### Experiment 1: Pixel-Loss Autoencoder (Baseline)
* **Model:** Convolutional Autoencoder (CAE)
* **Method:** Pixel-level L1 reconstruction loss.
* **Result:** **Total Failure.** The model failed to distinguish defects from normal textures.
    * **Average Image AUROC:** 0.6447
    * **Average Pixel AUPRO:** 0.5693

### Experiment 2: SOTA Global Feature Baseline (PaDiM-Style)
* **Model:** Pre-trained Wide-ResNet-50 (`wide_resnet50_2`)
* **Method:** Global feature vector (1x1792) extracted from each image. Anomaly score is the L2 distance to the nearest "normal" feature in a FAISS memory bank.
* **Result (on 'bottle' category):** **Partial Success.**

| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Image AUROC** | **0.9913** | **Excellent.** This model is near-perfect at the Pass/Fail (control) task. |
| **Pixel AUPRO** | 0.3770 | **Expected Low Score.** The model uses one global vector, so it has no pixel-level localization data. |

### Analysis & Next Steps

The global feature model (Exp 2) has **successfully solved the Pass/Fail detection problem (Objective O3)**.

The next and final step is to **solve the pixel-level localization problem (AUPRO)**. To do this, we will modify the model to use **patch-based features** instead of global features. This will provide the necessary spatial information to create a detailed anomaly map and will complete the project's primary objectives.# anomaly_detection
Real-time unsupervised anomaly detection for industrial QA using CAE/VAE on MVTec AD. Trained on defect-free images, evaluated via Image AUROC &amp; Pixel AUPRO, with smoothed L1 error for robust localization. Targets &lt;100ms inference and PLC-integrable Pass/Fail signals. 

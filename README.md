# Real-Time Anomaly Scoring for Industrial Pass/Fail Control
*Masters Project - In-Progress (NipunaNilupul)*

This project aims to develop a high-performance, real-time anomaly detection system. The current goal is to find a model that achieves high scores for both image-level (Pass/Fail) and pixel-level (Defect Localization) tasks.

---

## 📈 Experimental History & Research Progress

# Real-Time Anomaly Scoring for Industrial Pass/Fail Control
*Masters Project - In-Progress (NipunaNilupul)*

This project develops a high-performance, real-time anomaly detection system for industrial inspection. It documents the architectural evolution from Autoencoders to Patch-Based Feature Extraction to meet accuracy and speed targets.

---

## 📈 Experimental History & Research Progress

We systematically tested architectures to solve the difficult 'bottle' category (textured transparent surface) under a <100ms latency constraint.

### Experiment 1: Pixel-Loss Autoencoder (Baseline)
* **Model:** Convolutional Autoencoder (CAE)
* **Method:** Pixel-level L1 reconstruction loss.
* **Result:** **Accuracy Failure.** The model learned "Identity Mapping," reconstructing defects instead of flagging them.
    * [cite_start]**Pixel AUPRO:** 0.2056 (Random Guessing) [cite: 1177, 1191]

### Experiment 2: Global Feature Baseline (PaDiM-Style)
* **Model:** Wide-ResNet-50 (Global Pooling)
* **Method:** Single feature vector per image compared to a memory bank.
* **Result:** **Localization Failure.** Solved Pass/Fail but could not locate defects.
    * **Image AUROC:** 0.9913
    * **Pixel AUPRO:** 0.3770

### Experiment 3: Patch-Based ResNet-18 (Optimization I)
* **Model:** ResNet-18 (Patch Extraction)
* **Method:** 16x16 Feature Grid scoring using FAISS.
* **Result:** **Latency Failure.** Achieved perfect accuracy but failed the real-time constraint.
    * **Image AUROC:** 1.0000 (Perfect)
    * **Pixel AUPRO:** 0.9455 (Excellent Localization)
    * **Inference Latency:** ~276ms (Target: <100ms)

---

## 🚀 Current Status: Final Speed Optimization

We have solved the accuracy problem (Exp 3). We are now replacing the **ResNet-18** backbone with **MobileNetV3-Large** to reduce inference time from 276ms to <100ms.

### Objectives Tracking
| Objective | Status | Notes |
| :--- | :--- | :--- |
| **O1: Train Models** | ✅ Done | CAE, VAE, ResNet-18 trained/built. |
| **O2: Compare Models** | ✅ Done | Feature-Based >> Pixel-Based. |
| **O3: Thresholding** | ✅ Done | Optimal $\tau$ found via F1-max. |
| **O4: Real-Time (<100ms)** | ⚠️ In Progress | Current: 276ms. Goal: MobileNet optimization. |
| **O5: Control Bridge** | ⏳ Pending | Awaiting final model. |

---

## 🛠️ Usage

**1. Train (Build Memory Bank):**
```bash
python -m src.build_memory_bank

2. Evaluate (Generate Scores):
Bash

python -m src.evaluate

3. Profile Latency:
Bash

python -m src.profile_latency

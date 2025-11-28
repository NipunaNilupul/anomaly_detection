# Real-Time Anomaly Scoring for Industrial Pass/Fail Control
*Masters Project - Completed (NipunaNilupul)*

This project successfully developed a high-performance, real-time anomaly detection system for industrial inspection. It documents the architectural evolution from Autoencoders to Patch-Based Feature Extraction to meet accuracy (>0.90 AUPRO) and latency (<100ms) targets.

---

## 📈 Experimental History & Research Progress

We systematically tested four distinct architectures to solve the difficult 'bottle' category (textured transparent surface) under strict industrial constraints.

### Experiment 1: Pixel-Loss Autoencoder (Baseline)
* **Model:** Convolutional Autoencoder (CAE)
* **Method:** Pixel-level L1 reconstruction loss.
* **Result:** **Accuracy Failure.** The model learned "Identity Mapping," reconstructing defects instead of flagging them.
    * [cite_start]**Pixel AUPRO:** 0.2056 (Random Guessing) [cite: 1177, 1191]

### Experiment 2: Global Feature Baseline (PaDiM-Style)
* **Model:** Wide-ResNet-50 (Global Pooling)
* **Method:** Single feature vector per image compared to a memory bank.
* **Result:** **Localization Failure.** Solved Pass/Fail but could not locate defects.
    * [cite_start]**Image AUROC:** 0.9913 [cite: 1177]
    * [cite_start]**Pixel AUPRO:** 0.3770 [cite: 1177]

### Experiment 3: Patch-Based ResNet-18 (Optimization I)
* **Model:** ResNet-18 (Patch Extraction)
* **Method:** 16x16 Feature Grid scoring using FAISS.
* **Result:** **Latency Failure.** Achieved perfect accuracy but failed the real-time constraint.
    * **Image AUROC:** 1.0000
    * **Pixel AUPRO:** 0.9455
    * **Inference Latency:** ~276ms (Target: <100ms)

### Experiment 4: MobileNetV3 + 128px (Final Optimization)
* **Model:** MobileNetV3-Large (Layers 1, 2, 3)
* **Method:** Lightweight Patch Extraction on 128x128 input images.
* **Result:** **COMPLETE SUCCESS.**
    * **Image AUROC:** **0.9984**
    * **Pixel AUPRO:** **0.9333**
    * **Inference Latency:** **38.63ms** (25 FPS)

---

## 🚀 Final System Performance

We have successfully balanced the "Iron Triangle" of computer vision: Speed, Accuracy, and Localization.

### Objectives Tracking
| Objective | Status | Notes |
| :--- | :--- | :--- |
| **O1: Train Models** | ✅ Done | Tested 4 architectures (CAE, VAE, ResNet, MobileNet). |
| **O2: Compare Models** | ✅ Done | Feature-Based (MobileNet) >> Pixel-Based (CAE). |
| **O3: Thresholding** | ✅ Done | Optimal $\tau$ (`3288.27`) found via F1-max (F1=0.99). |
| **O4: Real-Time (<100ms)** | ✅ Done | Achieved **38.63ms** latency. |
| **O5: Control Bridge** | Pending | Implemented `control_bridge.py` for PLC simulation. |

---

## 🛠️ Usage

**1. Train (Build Memory Bank):**
Extracts normal patch features into a FAISS memory bank.
```bash
python -m src.build_memory_bank
---

2. Evaluate (Generate Scores): Calculates AUROC/AUPRO and saves visualization images to results/.
Bash

python -m src.evaluate

3. Optimize Threshold: Finds the best cut-off value (τ) for the control signal.
Bash

python -m src.optimize_threshold

4. Profile Latency: Checks if the model meets the <100ms requirement.
Bash

python -m src.profile_latency

5. Run Production Simulation (PLC Bridge): Simulates a live camera feed and outputs PASS/FAIL signals.
Bash

python -m src.control_bridge

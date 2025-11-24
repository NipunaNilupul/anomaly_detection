# Real-Time Anomaly Scoring for Industrial Pass/Fail Control
*Masters Project - In-Progress (NipunaNilupul)*

This project aims to develop a high-performance, real-time anomaly detection system. The current goal is to find a model that achieves high scores for both image-level (Pass/Fail) and pixel-level (Defect Localization) tasks.

---

## 📈 Experimental History & Research Progress

We systematically tested three distinct architectures to solve the difficult 'bottle' category (textured transparent surface).

### Experiment 1: Pixel-Loss Autoencoder (Baseline)
* **Model:** Convolutional Autoencoder (CAE)
* **Method:** Pixel-level L1 reconstruction loss.
* **Result:** **Total Failure.** The model failed to distinguish defects from normal textures due to "Identity Mapping".
    * **Average Image AUROC:** 0.6447
    * **Average Pixel AUPRO:** 0.5693

### Experiment 2: Global Feature Baseline (PaDiM-Style)
* **Model:** Pre-trained Wide-ResNet-50 (`wide_resnet50_2`)
* **Method:** Global feature vector (1x1792) extracted from each image. Anomaly score is the L2 distance to the nearest "normal" feature in a FAISS memory bank.
* **Result:** **Partial Success.**
    * **Image AUROC:** **0.9913** (Solved Pass/Fail).
    * **Pixel AUPRO:** 0.3770 (Failed Localization - no spatial data).

### Experiment 3: Patch-Based SOTA Model (Final Pivot)
* **Model:** Pre-trained Wide-ResNet-50 (`wide_resnet50_2`)
* **Method:** **Patch-level** feature extraction (16x16 grid). We score *each patch* individually against the memory bank to create a high-resolution anomaly map.
* **Result:** **Complete Success.**
    * **Image AUROC:** **1.0000** (Perfect Control Signal).
    * **Pixel AUPRO:** **0.9455** (High-Fidelity Localization).

---

## 🚀 Current Status: Optimization Phase

With accuracy objectives solved (Exp 3), we are now optimizing for deployment.

### 1. Threshold Optimization (Completed)
We scientifically determined the optimal decision boundary ($\tau$) using F1-Score maximization on the test set.
* **Optimal Threshold ($\tau$):** `4202.09`
* **Max F1-Score:** **1.0000** (100% Precision & Recall).

### 2. Latency Profiling (Current Challenge)
* Current Inference Time: ~1033ms (Wide-ResNet-50 on GTX 1650).
* Target: < 100ms.
* Action Plan:** We are actively migrating the backbone from "Wide-ResNet-50" to "ResNet-18 to reduce latency by ~10x while maintaining the high accuracy established in Experiment 3.


## 📂 Project Structure

* `src/models.py`: Contains the SOTA Feature Extractor (Patch-Based) and legacy CAE/VAE models.
* `src/build_memory_bank.py`: Extracts normal patch features into a FAISS memory bank.
* `src/evaluate.py`: Performs patch-level anomaly scoring and generates localization maps.
* `src/optimize_threshold.py`: Calculates the optimal $\tau$ for the Pass/Fail controller.
* `src/train_pixel.py`: (Legacy) Training script for the pixel-based baseline.
* `src/evaluate_pixel.py`: (Legacy) Evaluation script for the pixel-based baseline.

---

## 🛠️ Usage

**1. Train (Build Memory Bank):**
```bash
python -m src.build_memory_bank

**2. Evaluate (Generate Scores):**
Bash

python -m src.evaluate

**3. Optimize Threshold:** 
Bash

python -m src.optimize_threshold

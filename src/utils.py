import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score
from scipy.ndimage import gaussian_filter

# FULL LIST of all 15 MVTec AD categories
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 
    'transistor', 'wood', 'zipper'
]

def compute_pro(anomaly_maps, gt_masks, fpr_thresh=0.3):
    """Compute Per-Region Overlap (PRO) score."""
    # Flatten all maps and masks
    scores = np.concatenate([m.ravel() for m in anomaly_maps])
    labels = np.concatenate([m.ravel().astype(bool) for m in gt_masks])
    
    # Sort by score descending
    sorted_idx = np.argsort(scores)[::-1]
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    
    num_anom = labels.sum()
    num_norm = len(labels) - num_anom
    if num_anom == 0:
        return 1.0
    
    # Calculate TPR and FPR curves
    tpr = np.cumsum(labels) / num_anom
    fpr = np.cumsum(~labels) / num_norm
    
    # Find the TPR value at the specific FPR threshold
    valid = np.where(fpr <= fpr_thresh)[0]
    return tpr[valid[-1]] if len(valid) > 0 else tpr[0]

def compute_aupro(anomaly_maps, gt_masks):
    """Compute Area Under the PRO curve (AUPRO), integrated over FPR ∈ [0, 0.3]."""
    if not anomaly_maps: return 0.0
    fpr_vals = np.linspace(0, 0.3, 50)
    pro_vals = [compute_pro(anomaly_maps, gt_masks, fpr) for fpr in fpr_vals]
    return auc(fpr_vals, pro_vals) / 0.3

def compute_image_auroc(scores, labels):
    """Compute image-level AUROC."""
    return roc_auc_score(labels, scores)

# --- NEW VISUALIZATION FUNCTION ---
def save_anomaly_heatmap(image_path, scores, output_path, patch_grid_size=(8, 8)):
    """
    Generates and saves a heatmap overlay.
    
    Args:
        image_path (str): Path to the original input image.
        scores (np.array): Flat array of anomaly scores (distances from FAISS).
        output_path (str): Where to save the result.
        patch_grid_size (tuple): The spatial size of your features (e.g. 8x8 for 128px input).
    """
    try:
        # 1. Load Original Image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image at {image_path}")
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to match the resolution used in your MobileNet pipeline (128x128)
        img = cv2.resize(img, (128, 128)) 

        # 2. Reshape Scores to 2D Map
        # Ensure scores are numpy array
        if isinstance(scores, list):
            scores = np.array(scores)
            
        # Reshape flat scores (e.g., 64) into grid (e.g., 8x8)
        anomaly_map = scores.reshape(patch_grid_size)

        # 3. Upsample to Image Size
        # Resize 8x8 map to 128x128 using Cubic interpolation for smoothness
        anomaly_map = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        # 4. Gaussian Blur (Smoothing)
        # Smooth out blocky artifacts
        anomaly_map = cv2.GaussianBlur(anomaly_map, (11, 11), 0)

        # 5. Normalize (0 to 1) and Colorize
        # Normalize relative to this image's min/max to contrast the defect
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        norm_map = (norm_map * 255).astype(np.uint8)
        
        # Apply JET Colormap (Blue=Low, Red=High)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 6. Overlay
        alpha = 0.5 
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        # 7. Plot and Save
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image (128px)")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title("Anomaly Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title("Localization Overlay")
        axes[2].axis('off')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig) # Close to free memory
        print(f"Heatmap saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating heatmap for {image_path}: {e}")

import numpy as np
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
    scores = np.concatenate([m.ravel() for m in anomaly_maps])
    labels = np.concatenate([m.ravel().astype(bool) for m in gt_masks])
    
    sorted_idx = np.argsort(scores)[::-1]
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    
    num_anom = labels.sum()
    num_norm = len(labels) - num_anom
    if num_anom == 0:
        return 1.0
    
    tpr = np.cumsum(labels) / num_anom
    fpr = np.cumsum(~labels) / num_norm
    
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

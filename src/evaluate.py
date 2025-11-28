import os
import torch
import numpy as np
import faiss
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.utils import save_image
from scipy.ndimage import gaussian_filter
from src.models import get_feature_extractor
from src.utils import compute_aupro, compute_image_auroc

# Configuration 
# 128x128
RESOLUTION = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAUSSIAN_SIGMA = 4.0
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_memory_bank(category="bottle"):
    bank_path = f"models/{category}_patch_memory_bank.index"
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Memory bank not found: {bank_path}. Run build_memory_bank.py first.")
        
    print(f"Loading patch memory bank from {bank_path}")
    return faiss.read_index(bank_path)

def evaluate_category_features(category="bottle"):
    print(f"\n🚀 Starting PATCH-LEVEL evaluation for '{category}' (128x128)...")
    
    model = get_feature_extractor().to(DEVICE)
    index = load_memory_bank(category)
    raw_feature_dim = 176 

    test_dir = Path("data/mvtec_ad") / category / "test"
    gt_dir = Path("data/mvtec_ad") / category / "ground_truth"
    os.makedirs("results", exist_ok=True)
    
    test_transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),
        T.ToTensor(),
        IMAGENET_NORM
    ])
    
    anomaly_maps, gt_masks, image_scores, image_labels = [], [], [], []

    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue
        is_anomaly = defect_type.name != "good"
        
        for i, img_path in enumerate(tqdm(sorted(defect_type.glob("*.png")), desc=f"Evaluating {defect_type.name}")):
            
            image = Image.open(img_path).convert("RGB")
            x = test_transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                patch_features = model(x) 
                
                H, W = patch_features.shape[2:]
                patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(H * W, raw_feature_dim)
                patch_vectors_np = patch_vectors.cpu().numpy()

            D, I = index.search(patch_vectors_np, 1) 
            
            anomaly_map_hw = D.reshape(H, W)
            anomaly_map_full = T.Resize(
                (RESOLUTION, RESOLUTION), 
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True
            )(torch.tensor(anomaly_map_hw).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            
            error_map = gaussian_filter(anomaly_map_full, sigma=GAUSSIAN_SIGMA)
            
            if (defect_type.name == "good" and i == 0) or (defect_type.name == "broken_large" and i == 0):
                orig_img = Image.open(img_path).resize((RESOLUTION, RESOLUTION))
                orig_img.save(f"results/patch_input_{defect_type.name}.png")
                norm_error = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
                error_vis = torch.from_numpy(norm_error).unsqueeze(0).repeat(3,1,1)
                save_image(error_vis, f"results/patch_error_{defect_type.name}.png")

            if is_anomaly:
                mask_path = gt_dir / defect_type.name / (img_path.stem + "_mask.png")
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
                mask = np.array(mask) / 255.0
            else:
                mask = np.zeros((RESOLUTION, RESOLUTION))
            
            anomaly_maps.append(error_map)
            gt_masks.append((mask > 0.5).astype(np.uint8))
            image_scores.append(error_map.max()) 
            image_labels.append(1 if is_anomaly else 0)
    
    pixel_aupro = compute_aupro(anomaly_maps, gt_masks)
    image_auroc = compute_image_auroc(image_scores, image_labels)
    
    print(f"\n📊 {category.upper()} (Patch-Level 128x128)")
    print(f"   Pixel AUPRO: {pixel_aupro:.4f}")
    print(f"   Image AUROC: {image_auroc:.4f}")
    return pixel_aupro, image_auroc

if __name__ == "__main__":
    evaluate_category_features(category="bottle")

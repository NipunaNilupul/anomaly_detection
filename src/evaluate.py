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

# --- Configuration ---
RESOLUTION = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAUSSIAN_SIGMA = 4.0
# Define the same normalization as in the dataloader
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_memory_bank(category="bottle"):
    """Loads the pre-built FAISS index."""
    bank_path = f"models/{category}_memory_bank.index"
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Memory bank not found: {bank_path}. Run build_memory_bank.py first.")
        
    print(f"Loading memory bank from {bank_path}")
    return faiss.read_index(bank_path)

def evaluate_category_features(category="bottle"):
    """
    Evaluates a category using the feature-distance method.
    """
    print(f"\n🚀 Starting feature-based evaluation for '{category}'...")
    
    # 1. Load Model and Memory Bank
    model = get_feature_extractor().to(DEVICE)
    index = load_memory_bank(category)
    feature_dim = index.d # Get dimension (1792) from the index

    # 2. Setup Dirs & Transforms
    test_dir = Path("data/mvtec_ad") / category / "test"
    gt_dir = Path("data/mvtec_ad") / category / "ground_truth"
    os.makedirs("results", exist_ok=True)
    
    # Create the transform for test images
    test_transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),
        T.ToTensor(),
        IMAGENET_NORM
    ])
    
    anomaly_maps, gt_masks, image_scores, image_labels = [], [], [], []

    # 3. Evaluation Loop
    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue
        is_anomaly = defect_type.name != "good"
        
        for i, img_path in enumerate(tqdm(sorted(defect_type.glob("*.png")), desc=f"Evaluating {defect_type.name}")):
            
            # 1. Load and Transform Image
            image = Image.open(img_path).convert("RGB")
            x = test_transform(image).unsqueeze(0).to(DEVICE)
            
            # 2. Extract Features
            with torch.no_grad():
                features = model(x).cpu().numpy().astype(np.float32)
                
            # 3. Score Features
            # Find K-Nearest Neighbors (K=1)
            # D = distances, I = indices
            D, I = index.search(features, 1) # D is shape [1, 1]
            
            # The anomaly score for the *entire image* is its L2 distance
            # to the closest normal image in the memory bank
            image_anomaly_score = D[0][0]
            
            # 4. Load Ground Truth Mask
            if is_anomaly:
                mask_path = gt_dir / defect_type.name / (img_path.stem + "_mask.png")
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
                mask = np.array(mask) / 255.0
            else:
                mask = np.zeros((RESOLUTION, RESOLUTION))
            
            # 5. Store Metrics
            # For this method (global feature vector), the anomaly map
            # is just a constant value (the image score) across all pixels.
            anomaly_map = np.full((RESOLUTION, RESOLUTION), image_anomaly_score)
            
            anomaly_maps.append(anomaly_map)
            gt_masks.append((mask > 0.5).astype(np.uint8))
            image_scores.append(image_anomaly_score)
            image_labels.append(1 if is_anomaly else 0)
    
    # 6. Compute Final Metrics
    pixel_aupro = compute_aupro(anomaly_maps, gt_masks)
    image_auroc = compute_image_auroc(image_scores, image_labels)
    
    print(f"\n📊 {category.upper()} (Feature-Level)")
    print(f"   Pixel AUPRO: {pixel_aupro:.4f}")
    print(f"   Image AUROC: {image_auroc:.4f}")
    return pixel_aupro, image_auroc

if __name__ == "__main__":
    evaluate_category_features(category="bottle")

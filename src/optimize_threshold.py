import torch
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import precision_recall_curve
from src.models import get_feature_extractor

# --- Configuration ---
RESOLUTION = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_components(category):
    # Load Model
    model = get_feature_extractor().to(DEVICE)
    
    # Load Memory Bank
    bank_path = f"models/{category}_patch_memory_bank.index"
    projector_path = f"models/{category}_patch_projector.pth"
    
    if not Path(bank_path).exists():
        raise FileNotFoundError("Memory bank not found. Run build_memory_bank.py first.")
        
    index = faiss.read_index(bank_path)
    projector = torch.load(projector_path)
    
    return model, index, projector

def find_optimal_threshold(category="bottle"):
    print(f"\n🚀 Optimizing Threshold for '{category}'...")
    
    model, index, projector = load_components(category)
    test_dir = Path("data/mvtec_ad") / category / "test"
    
    test_transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),
        T.ToTensor(),
        IMAGENET_NORM
    ])
    
    y_true = []
    y_scores = []
    
    # 1. Collect Scores for ALL Test Images
    print("Collecting anomaly scores...")
    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue
        is_anomaly = defect_type.name != "good"
        
        for img_path in tqdm(sorted(defect_type.glob("*.png")), desc=defect_type.name):
            # Load
            image = Image.open(img_path).convert("RGB")
            x = test_transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # Extract Patch Features
                patch_features = model(x) # (1, 1792, 16, 16)
                
                # Reshape
                H, W = patch_features.shape[2:]
                raw_dim = 1792
                patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, raw_dim).cpu().numpy()
                
                # Project
                projected_vectors = projector.transform(patch_vectors).astype(np.float32)
                
                # Search (1-NN)
                D, _ = index.search(projected_vectors, 1)
                
                # The image score is the MAXIMUM patch distance found in the image
                # (If even one patch is very far, the image is likely defective)
                image_score = np.max(D)
                
            y_true.append(1 if is_anomaly else 0)
            y_scores.append(image_score)

    # 2. Calculate Precision-Recall Curve
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # 3. Compute F1 Scores
    # F1 = 2 * (P * R) / (P + R)
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # 4. Find Optimal Threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n✅ Optimization Complete for '{category}'")
    print(f"   Optimal Threshold (tau): {best_threshold:.6f}")
    print(f"   Max F1-Score:           {best_f1:.4f}")
    
    # Save threshold for the future Control System
    with open(f"models/{category}_threshold.txt", "w") as f:
        f.write(str(best_threshold))
        
    return best_threshold

if __name__ == "__main__":
    find_optimal_threshold("bottle")

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.utils import save_image
from scipy.ndimage import gaussian_filter
from src.models import CAE, VAE
from src.utils import compute_aupro, compute_image_auroc

# Configuration
RESOLUTION = 256 
GAUSSIAN_SIGMA = 4.0 

def load_model(model_type, path, latent_dim=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "cae":
        model = CAE(latent_dim).to(device)
    else:
        model = VAE(latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, device

def evaluate_pixel_model(model_type="cae", model_path="models/cae_bottle_best.pth", category="bottle", latent_dim=512):
    model, device = load_model(model_type, model_path, latent_dim)
    test_dir = Path("data/mvtec_ad") / category / "test"
    gt_dir = Path("data/mvtec_ad") / category / "ground_truth"
    os.makedirs("results", exist_ok=True)
    
    anomaly_maps, gt_masks, image_scores, image_labels = [], [], [], []
    
    print(f"\n🚀 Starting PIXEL-BASED evaluation for '{category}'")

    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue
        is_anomaly = defect_type.name != "good"
        for i, img_path in enumerate(tqdm(sorted(defect_type.glob("*.png")), desc=f"Evaluating {defect_type.name}")):
            
            image = Image.open(img_path).convert("RGB")
            x = T.Compose([T.Resize((RESOLUTION, RESOLUTION)), T.ToTensor()])(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(x)
                recon = output if model_type == "cae" else output[0]
                
                # L1 Pixel Error
                l1_map_numpy = torch.abs(x - recon).mean(dim=1).squeeze().cpu().numpy()
                
                # Gaussian Smoothing
                error_map = gaussian_filter(l1_map_numpy, sigma=GAUSSIAN_SIGMA) 
            
            if (defect_type.name == "good" and i == 0) or (defect_type.name == "broken_large" and i == 0):
                save_image(x[0], f"results/pixel_{model_type}_input_{defect_type.name}.png")
                save_image(recon[0], f"results/pixel_{model_type}_recon_{defect_type.name}.png")
                error_map_tensor = torch.from_numpy(error_map)
                error_vis = error_map_tensor.unsqueeze(0).repeat(3, 1, 1)
                save_image(error_vis, f"results/pixel_{model_type}_error_{defect_type.name}.png")
            
            if is_anomaly:
                mask = Image.open(gt_dir / defect_type.name / (img_path.stem + "_mask.png")).convert("L")
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
    
    print(f"\n📊 {model_type.upper()} on '{category}' (Pixel-Based)")
    print(f"   Pixel AUPRO: {pixel_aupro:.4f}")
    print(f"   Image AUROC: {image_auroc:.4f}")
    return pixel_aupro, image_auroc

if __name__ == "__main__":
    evaluate_pixel_model("cae", "models/cae_bottle_best.pth", "bottle")

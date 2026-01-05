import os  # Standard library for filesystem operations
import torch  # PyTorch deep learning framework
import numpy as np  # Library for numerical operations
from PIL import Image  # Library for image loading and manipulation
from pathlib import Path  # Object-oriented path handling
from tqdm import tqdm  # Library for progress bars
import torchvision.transforms as T  # PyTorch image transformations
from torchvision.utils import save_image  # Utility to save tensors as image files
from scipy.ndimage import gaussian_filter  # Function for smoothing error maps
# --- Import your custom project modules ---
from src.models import CAE, VAE  # Import the architectures defined in your models file
from src.utils import compute_aupro, compute_image_auroc  # Import custom metrics

# Configuration
# Resolution 256x256 is often used for pixel-based methods to retain spatial detail
RESOLUTION = 256 
# Sigma=4.0 smooths out high-frequency noise in the reconstruction error map
GAUSSIAN_SIGMA = 4.0 

def load_model(model_type, path, latent_dim=512):
    """Loads a trained CAE or VAE model from disk."""
    # Automatically select GPU for faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the specific architecture based on input string
    if model_type == "cae":
        model = CAE(latent_dim).to(device)
    else:
        model = VAE(latent_dim).to(device)
        
    # Load the trained weights. 'weights_only=True' is a security best practice to prevent code injection from pickles.
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    
    # Set to evaluation mode (critical: disables Dropout and fixes BatchNorm stats)
    model.eval()
    return model, device

def evaluate_pixel_model(model_type="cae", model_path="models/cae_bottle_best.pth", category="bottle", latent_dim=512):
    # 1. Initialize Model
    model, device = load_model(model_type, model_path, latent_dim)
    
    # 2. Setup Paths
    test_dir = Path("data/mvtec_ad") / category / "test"
    gt_dir = Path("data/mvtec_ad") / category / "ground_truth"
    os.makedirs("results", exist_ok=True)  # Create output directory for debug images
    
    # Lists to aggregate results for final metric calculation
    anomaly_maps, gt_masks, image_scores, image_labels = [], [], [], []
    
    print(f"\n Starting PIXEL-BASED evaluation for '{category}'")

    # Iterate through all test folders (e.g., 'good', 'broken_large', 'contamination')
    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue
        
        # Binary Classification: 'good' folder is class 0, everything else is class 1 (Anomaly)
        is_anomaly = defect_type.name != "good"
        
        # Process every image in the folder
        for i, img_path in enumerate(tqdm(sorted(defect_type.glob("*.png")), desc=f"Evaluating {defect_type.name}")):
            
            # --- Inference Pipeline ---
            image = Image.open(img_path).convert("RGB") # Load image
            # Resize and normalize to Tensor (0-1 range). No ImageNet stats here, usually standard 0-1 for Autoencoders.
            x = T.Compose([T.Resize((RESOLUTION, RESOLUTION)), T.ToTensor()])(image).unsqueeze(0).to(device)
            
            with torch.no_grad(): # Disable gradients for speed
                output = model(x)
                # Handle different return types: CAE returns just reconstruction; VAE returns (recon, mean, logvar)
                recon = output if model_type == "cae" else output[0]
                
                # --- Anomaly Scoring ---
                # Calculate L1 Loss (Absolute Difference) per pixel across RGB channels
                # L1 is preferred over MSE here as it produces sharper error boundaries for defects
                l1_map_numpy = torch.abs(x - recon).mean(dim=1).squeeze().cpu().numpy()
                
                # Apply Gaussian Blur: This aggregates pixel errors into a coherent "region"
                # Helps distinguish a cluster of error pixels (defect) from random noise
                error_map = gaussian_filter(l1_map_numpy, sigma=GAUSSIAN_SIGMA) 
            
            # --- Visualization (Debug Output) ---
            # Save qualitative examples only for the first image of specific categories
            if (defect_type.name == "good" and i == 0) or (defect_type.name == "broken_large" and i == 0):
                save_image(x[0], f"results/pixel_{model_type}_input_{defect_type.name}.png") # Input
                save_image(recon[0], f"results/pixel_{model_type}_recon_{defect_type.name}.png") # Reconstruction
                
                # Normalize and save the error map as a heatmap-like image
                error_map_tensor = torch.from_numpy(error_map)
                error_vis = error_map_tensor.unsqueeze(0).repeat(3, 1, 1) # Expand to 3 channels for saving
                save_image(error_vis, f"results/pixel_{model_type}_error_{defect_type.name}.png")
            
            # --- Ground Truth Handling ---
            if is_anomaly:
                # Load corresponding binary mask for defects
                mask = Image.open(gt_dir / defect_type.name / (img_path.stem + "_mask.png")).convert("L")
                mask = mask.resize((RESOLUTION, RESOLUTION), Image.NEAREST) # Resize preserving binary values
                mask = np.array(mask) / 255.0 # Normalize 0-1
            else:
                # Create empty mask for non-defective images
                mask = np.zeros((RESOLUTION, RESOLUTION))
            
            # Collect data for metrics
            anomaly_maps.append(error_map)
            gt_masks.append((mask > 0.5).astype(np.uint8)) # Threshold mask to ensure binary
            image_scores.append(error_map.max()) # Image-level score is the highest error pixel
            image_labels.append(1 if is_anomaly else 0)
    
    # --- Final Metric Calculation ---
    # Compute localization (AUPRO) and detection (AUROC) metrics
    pixel_aupro = compute_aupro(anomaly_maps, gt_masks)
    image_auroc = compute_image_auroc(image_scores, image_labels)
    
    print(f"\n {model_type.upper()} on '{category}' (Pixel-Based)")
    print(f"   Pixel AUPRO: {pixel_aupro:.4f}")
    print(f"   Image AUROC: {image_auroc:.4f}")
    return pixel_aupro, image_auroc

if __name__ == "__main__":
    # Entry point: Run evaluation for the CAE model on the 'bottle' category
    evaluate_pixel_model("cae", "models/cae_bottle_best.pth", "bottle")

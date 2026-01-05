import os  # Standard library for interacting with the operating system (e.g., checking file existence)
import torch  # PyTorch deep learning framework for tensor operations and model execution
import numpy as np  # Library for numerical operations and array manipulation
import faiss  # Facebook AI Similarity Search library for efficient nearest-neighbor search
from PIL import Image  # Library for loading and manipulating images
from pathlib import Path  # Object-oriented filesystem paths for easier path handling
from tqdm import tqdm  # Library to display progress bars during long loops
import torchvision.transforms as T  # PyTorch transforms for image preprocessing (resize, normalize)
from torchvision.utils import save_image  # Utility to save PyTorch tensors directly as image files
from scipy.ndimage import gaussian_filter  # Function to apply Gaussian smoothing to the anomaly maps
from src.models import get_feature_extractor  # Imports your custom MobileNetV3 model wrapper
from src.utils import compute_aupro, compute_image_auroc  # Imports your custom metric calculation functions

# Configuration 
# Sets input resolution to 128x128. Matches training configuration to ensure feature consistency.
RESOLUTION = 128
# Automatically selects GPU for fast inference if available, otherwise falls back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sigma=4.0 for Gaussian blur. Balances smoothing noise while preserving defect shapes.
GAUSSIAN_SIGMA = 4.0
# Standard ImageNet normalization stats. Critical for pre-trained MobileNetV3 models to work correctly.
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_memory_bank(category="bottle"):
    """Loads the pre-computed FAISS index from disk."""
    # Constructs the path to the FAISS index file for the specific category
    bank_path = f"models/{category}_patch_memory_bank.index"
    
    # Fail-fast check: ensures the memory bank file exists before proceeding
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Memory bank not found: {bank_path}. Run build_memory_bank.py first.")
        
    print(f"Loading patch memory bank from {bank_path}")
    return faiss.read_index(bank_path)  # Loads the index from disk into RAM for searching

def evaluate_category_features(category="bottle"):
    print(f"\n Starting PATCH-LEVEL evaluation for '{category}' (128x128)...")
    
    # 1. Initialize Model & Index
    # Loads the feature extractor (MobileNetV3) and moves it to the active device (GPU)
    model = get_feature_extractor().to(DEVICE)
    # Loads the specific memory bank for the category being evaluated
    index = load_memory_bank(category)
    # Defines the dimension of the feature vector (MobileNetV3 Layers 1+2+3 concatenated = 176)
    raw_feature_dim = 176 

    # 2. Setup Data Paths
    # Path to the test images for this category
    test_dir = Path("data/mvtec_ad") / category / "test"
    # Path to the ground truth masks (pixel-level labels)
    gt_dir = Path("data/mvtec_ad") / category / "ground_truth"
    # Creates the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # 3. Define Preprocessing Pipeline (Must match training!)
    test_transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),  # Resizes image to 128x128
        T.ToTensor(),  # Converts image to Tensor (0-1 float)
        IMAGENET_NORM  # Normalizes using ImageNet mean/std
    ])
    
    # Initialize lists to store results for final metric calculation
    anomaly_maps, gt_masks, image_scores, image_labels = [], [], [], []

    # Iterate through each defect type folder (e.g., 'broken_large', 'contamination', 'good')
    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue  # Skip files, only process directories
        
        # Determine if the current folder contains anomalies (anything not named "good")
        is_anomaly = defect_type.name != "good"
        
        # Process every image in the folder with a progress bar
        for i, img_path in enumerate(tqdm(sorted(defect_type.glob("*.png")), desc=f"Evaluating {defect_type.name}")):
            
            # --- Inference Step ---
            image = Image.open(img_path).convert("RGB")  # Load image and ensure 3 channels (RGB)
            x = test_transform(image).unsqueeze(0).to(DEVICE)  # Preprocess & add batch dimension [1, 3, 128, 128]
            
            with torch.no_grad():  # Disable gradient calculation to save memory and speed up inference
                patch_features = model(x)  # Pass image through MobileNetV3 to get feature maps
                
                # Reshape features: [1, C, H, W] -> [H*W, C] (Flatten spatial dimensions)
                H, W = patch_features.shape[2:]
                # Permute ensures channels are last, reshape flattens height/width
                patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(H * W, raw_feature_dim)
                patch_vectors_np = patch_vectors.cpu().numpy() # Convert to Numpy array for FAISS

            # --- Feature Matching Step ---
            # Search for the nearest neighbor in the memory bank for every patch
            # D = Distances (Anomaly Scores), I = Indices (Ignored)
            D, I = index.search(patch_vectors_np, 1) 
            
            # --- Anomaly Map Generation ---
            anomaly_map_hw = D.reshape(H, W)  # Reshape flat distance scores back to 2D grid
            
            # Upsample the low-res feature map back to input resolution (128x128)
            # Bilinear interpolation creates a smooth transition between patch scores
            anomaly_map_full = T.Resize(
                (RESOLUTION, RESOLUTION), 
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True
            )(torch.tensor(anomaly_map_hw).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            
            # Apply Gaussian Blur to smooth blocky artifacts and create a natural-looking heatmap
            error_map = gaussian_filter(anomaly_map_full, sigma=GAUSSIAN_SIGMA)
            
            # --- Visualization (Debug Output) ---
            # Saves visualization only for the first image of 'good' and 'broken_large' classes for inspection
            if (defect_type.name == "good" and i == 0) or (defect_type.name == "broken_large" and i == 0):
                orig_img = Image.open(img_path).resize((RESOLUTION, RESOLUTION)) # Load original for reference
                orig_img.save(f"results/patch_input_{defect_type.name}.png") # Save input
                
                # Normalize error map to 0-1 range for visualization
                norm_error = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
                error_vis = torch.from_numpy(norm_error).unsqueeze(0).repeat(3,1,1) # Make 3-channel for saving
                save_image(error_vis, f"results/patch_error_{defect_type.name}.png") # Save heatmap

            # --- Ground Truth Handling ---
            if is_anomaly:
                # Load ground truth mask corresponding to the defect image
                mask_path = gt_dir / defect_type.name / (img_path.stem + "_mask.png")
                mask = Image.open(mask_path).convert("L") # Load as grayscale
                mask = mask.resize((RESOLUTION, RESOLUTION), Image.NEAREST) # Resize using Nearest Neighbor (preserves binary values)
                mask = np.array(mask) / 255.0 # Normalize to 0-1
            else:
                # Create a blank (all zero) mask for good images
                mask = np.zeros((RESOLUTION, RESOLUTION))
            
            # Store results for metrics calculation
            anomaly_maps.append(error_map) # Store predicted heatmap
            gt_masks.append((mask > 0.5).astype(np.uint8)) # Store binary ground truth mask
            image_scores.append(error_map.max()) # Image-level score is the maximum pixel score
            image_labels.append(1 if is_anomaly else 0) # Store classification label (1=Defect, 0=Good)
    
    # --- Metric Calculation ---
    # Compute AUPRO (Localization metric) and AUROC (Detection metric)
    pixel_aupro = compute_aupro(anomaly_maps, gt_masks)
    image_auroc = compute_image_auroc(image_scores, image_labels)
    
    # Print final results
    print(f"\n {category.upper()} (Patch-Level 128x128)")
    print(f"   Pixel AUPRO: {pixel_aupro:.4f}")
    print(f"   Image AUROC: {image_auroc:.4f}")
    return pixel_aupro, image_auroc

if __name__ == "__main__":
    # Entry point: runs the evaluation for the 'bottle' category
    evaluate_category_features(category="bottle")

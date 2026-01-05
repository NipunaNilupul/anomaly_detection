import os  # Standard library for filesystem operations (checking paths, creating folders)
import cv2  # OpenCV library used here for image processing tasks like heatmap generation
import torch  # The core deep learning framework used for tensor operations and GPU acceleration
import numpy as np  # Essential library for numerical array manipulation and mathematical operations
import faiss  # Facebook AI Similarity Search: The library driving the fast 'Memory Bank' search
from PIL import Image  # Python Imaging Library for loading images in a format compatible with Torchvision
from pathlib import Path  # Modern object-oriented filesystem path handling (cleaner than os.path)
from tqdm import tqdm  # Library to display progress bars for long-running loops
import matplotlib.pyplot as plt  # Library for creating and saving the visualization plots
import torchvision.transforms as T  # PyTorch tools for preprocessing images (resize, normalize)
from scipy.ndimage import gaussian_filter  # Function for smoothing the anomaly map (removing noise)
from torchvision.utils import save_image  # Utility to save PyTorch tensors directly as image files

# --- Import your project modules ---
from src.models import get_feature_extractor  # Imports your custom MobileNetV3 backbone wrapper
from src.utils import compute_aupro, compute_image_auroc  # Imports your custom metric calculation functions

# --- Configuration ---
RESOLUTION = 128  # Sets input resolution to 128x128. This is the key optimization for <40ms latency.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-selects GPU for speed if available
GAUSSIAN_SIGMA = 4.0  # Smoothing factor for the heatmap. 4.0 is a balanced value for defects.
# Standard ImageNet normalization stats. Required because MobileNetV3 was pre-trained on ImageNet.
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_memory_bank(category="bottle"):
    """Loads the pre-computed FAISS index from disk."""
    # Construct path to the specific category's index file
    bank_path = f"models/{category}_patch_memory_bank.index"
    
    # Fail-fast check: Crash immediately with a helpful error if the file is missing
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Memory bank not found: {bank_path}. Run build_memory_bank.py first.")
        
    print(f"Loading patch memory bank from {bank_path}")
    return faiss.read_index(bank_path)  # Load the index structure into RAM

def generate_heatmap(orig_image_path, error_map, output_path):
    """
    Creates a professional Heatmap Overlay and saves it to disk.
    """
    # 1. Load Original Image using OpenCV
    img = cv2.imread(str(orig_image_path))
    if img is None:  # Safety check for broken image paths
        print(f"Error loading image: {orig_image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB (Matplotlib format)
    img = cv2.resize(img, (RESOLUTION, RESOLUTION))  # Resize to match the 128x128 analysis resolution

    # 2. Normalize Error Map (0 to 255)
    # Scales the raw float scores to 8-bit integers for image display
    norm_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    norm_map = (norm_map * 255).astype(np.uint8)

    # 3. Apply Jet Colormap (Blue=Good, Red=Defect)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert heatmap to RGB

    # 4. Create Overlay
    alpha = 0.4  # Transparency factor (40% heatmap, 60% original image)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)  # Blend images

    # 5. Save Side-by-Side Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Create a layout with 3 subplots
    
    axes[0].imshow(img); axes[0].set_title("Input Image"); axes[0].axis('off')  # Plot Input
    axes[1].imshow(heatmap); axes[1].set_title("Anomaly Heatmap"); axes[1].axis('off')  # Plot Heatmap
    axes[2].imshow(overlay); axes[2].set_title("Figure \n(Localization Success)"); axes[2].axis('off')  # Plot Overlay
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(output_path)  # Save to disk
    plt.close()  # Close plot to free memory
    print(f" Figure saved to: {output_path}")

def evaluate_category_features(category="bottle"):
    print(f"\n Starting PATCH-LEVEL evaluation for '{category}' (128x128)...")
    
    # Initialize model and load to GPU
    model = get_feature_extractor().to(DEVICE)
    # Load the FAISS index (Memory Bank)
    index = load_memory_bank(category)
    # Define vector dimension (MobileNetV3 Layers 1+2+3 concatenated = 176)
    raw_feature_dim = 176 

    # Define paths for test data and ground truth
    test_dir = Path("data/mvtec_ad") / category / "test"
    gt_dir = Path("data/mvtec_ad") / category / "ground_truth"
    os.makedirs("results", exist_ok=True)  # Create results folder if missing
    
    # Define preprocessing pipeline (Must match training!)
    test_transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),
        T.ToTensor(),
        IMAGENET_NORM
    ])
    
    # Lists to store results for final metric calculation
    anomaly_maps, gt_masks, image_scores, image_labels = [], [], [], []

    # Iterate through all subfolders (defect types)
    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue
        is_anomaly = defect_type.name != "good"  # Flag for binary classification
        
        # Loop through images with progress bar
        for i, img_path in enumerate(tqdm(sorted(defect_type.glob("*.png")), desc=f"Evaluating {defect_type.name}")):
            
            # --- 1. Inference Pipeline ---
            image = Image.open(img_path).convert("RGB")  # Load image
            x = test_transform(image).unsqueeze(0).to(DEVICE)  # Preprocess & add batch dim
            
            with torch.no_grad():  # Disable gradients for speed
                patch_features = model(x)  # Extract features
                
                # Reshape features: [1, C, H, W] -> [H*W, C] for FAISS
                H, W = patch_features.shape[2:]
                patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(H * W, raw_feature_dim)
                patch_vectors_np = patch_vectors.cpu().numpy()

            # Nearest Neighbor Search in Memory Bank
            D, I = index.search(patch_vectors_np, 1) 
            
            # --- 2. Map Generation ---
            anomaly_map_hw = D.reshape(H, W)  # Reshape scores back to 2D grid
            
            # Upsample low-res feature map to 128x128 using Bilinear Interpolation
            anomaly_map_full = T.Resize(
                (RESOLUTION, RESOLUTION), 
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True
            )(torch.tensor(anomaly_map_hw).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            
            # Apply Gaussian Blur to smooth blocky artifacts
            error_map = gaussian_filter(anomaly_map_full, sigma=GAUSSIAN_SIGMA)
            
            # --- 3. VISUALIZATION GENERATION ---
            # Special logic to generate Figure 4.2 for the thesis (Broken Bottle)
            if defect_type.name == "broken_large" and i == 0:
                print("\n[INFO] Generating Figure for Broken Bottle...")
                generate_heatmap(
                    orig_image_path=img_path,
                    error_map=error_map,
                    output_path="results/Figure_4_2_MobileNet_Success.png"
                )

            # --- 4. Metrics & Masks ---
            if is_anomaly:
                # Load Ground Truth mask if it exists
                mask_path = gt_dir / defect_type.name / (img_path.stem + "_mask.png")
                if mask_path.exists():
                    mask = Image.open(mask_path).convert("L")
                    mask = mask.resize((RESOLUTION, RESOLUTION), Image.NEAREST)  # Keep binary values
                    mask = np.array(mask) / 255.0  # Normalize 0-1
                else:
                    mask = np.zeros((RESOLUTION, RESOLUTION))  # Fallback
            else:
                mask = np.zeros((RESOLUTION, RESOLUTION))  # Good images have empty masks
            
            # Store data for metrics
            anomaly_maps.append(error_map)
            gt_masks.append((mask > 0.5).astype(np.uint8))
            image_scores.append(error_map.max())  # Image score = max pixel score
            image_labels.append(1 if is_anomaly else 0)
    
    # --- 5. Final Metrics ---
    # Compute localization (AUPRO) and detection (AUROC) metrics
    pixel_aupro = compute_aupro(anomaly_maps, gt_masks)
    image_auroc = compute_image_auroc(image_scores, image_labels)
    
    # Print results
    print(f"\n {category.upper()} (Patch-Level 128x128)")
    print(f"   Pixel AUPRO: {pixel_aupro:.4f}")
    print(f"   Image AUROC: {image_auroc:.4f}")
    return pixel_aupro, image_auroc

if __name__ == "__main__":
    evaluate_category_features(category="bottle")

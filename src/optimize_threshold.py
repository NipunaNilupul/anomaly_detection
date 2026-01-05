import torch  # Core Deep Learning library
import numpy as np  # For numerical operations on scores and arrays
import faiss  # Facebook AI Similarity Search (The Memory Bank)
from pathlib import Path  # Object-oriented filesystem path handling
from tqdm import tqdm  # Progress bar for tracking the loop status
from PIL import Image  # Library for loading images from disk
import torchvision.transforms as T  # PyTorch image preprocessing tools
from sklearn.metrics import precision_recall_curve  # Metric tool to calculate P-R curve points
from src.models import get_feature_extractor  # Imports your custom MobileNetV3 wrapper

# --- Configuration ---
# Resolution 128x128 matches the training setup. Consistency is key for valid inference.
RESOLUTION = 128
# Automatically detect GPU availability to ensure fast processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define standard ImageNet normalization. MobileNetV3 expects this distribution of pixel values.
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_components(category):
    """
    Helper function to load the Model and the Memory Bank into RAM/VRAM.
    """
    # Initialize the MobileNetV3 feature extractor and move it to the GPU
    model = get_feature_extractor().to(DEVICE)
    
    # Construct the path to the trained FAISS index for this category
    bank_path = f"models/{category}_patch_memory_bank.index"
    
    # Load the FAISS index from disk. This contains the 'normal' patch embeddings.
    index = faiss.read_index(bank_path)
    return model, index

def find_optimal_threshold(category="bottle"):
    print(f"\n Optimizing Threshold for '{category}' (128x128)...")
    
    # 1. Load the AI Engine (Model + Memory Bank)
    model, index = load_components(category)
    
    # Define paths to test data (contains both 'good' and 'defect' images)
    test_dir = Path("data/mvtec_ad") / category / "test"
    
    # The output dimension of MobileNetV3 (Layers 1+2+3 concatenated)
    raw_feature_dim = 176
    
    # 2. Define Preprocessing Pipeline
    # Must be identical to the training pipeline to ensure features match the memory bank
    test_transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),
        T.ToTensor(),
        IMAGENET_NORM
    ])
    
    # Lists to store the ground truth (0/1) and the predicted anomaly scores
    y_true, y_scores = [], []
    
    print("Collecting anomaly scores...")
    
    # Iterate through all subfolders (e.g., 'good', 'broken_large', 'contamination')
    for defect_type in sorted(test_dir.iterdir()):
        if not defect_type.is_dir(): continue # Skip files, only process folders
        
        # Ground Truth Logic: 'good' folder = 0 (Normal), anything else = 1 (Anomaly)
        is_anomaly = defect_type.name != "good"
        
        # Loop through every image in the current folder
        for img_path in tqdm(sorted(defect_type.glob("*.png")), desc=defect_type.name):
            
            # --- Inference Pipeline (Identical to Deployment) ---
            image = Image.open(img_path).convert("RGB") # Load image
            x = test_transform(image).unsqueeze(0).to(DEVICE) # Preprocess & add batch dim
            
            with torch.no_grad(): # Disable gradients for speed
                # Extract features using MobileNetV3
                patch_features = model(x)
                
                # Reshape features: [1, C, H, W] -> [N_Patches, C] for FAISS search
                H, W = patch_features.shape[2:]
                patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, raw_feature_dim).cpu().numpy()
                
                # Nearest Neighbor Search: Find distance to closest 'normal' patch
                D, _ = index.search(patch_vectors, 1)
                
                # Image Score = The maximum patch distance found in the image
                # (i.e., the "most anomalous" part of the image determines the score)
                image_score = np.max(D)
                
            # Append results to lists for batch calculation later
            y_true.append(1 if is_anomaly else 0)
            y_scores.append(image_score)

    # Convert lists to NumPy arrays for efficient vector math
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 3. Calculate Precision-Recall Curve
    # This function calculates precision/recall at *every possible threshold* in the dataset
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # 4. Calculate F1-Score for every threshold
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    numerator = 2 * precision * recall
    denominator = precision + recall
    
    # Safe division: handles cases where denominator is 0 (to avoid NaNs)
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # 5. Find the Maximum F1 Score
    # The index of the highest F1 score corresponds to the "Optimal Threshold"
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Output Results
    print(f"\n Optimization Complete for '{category}'")
    print(f"   Optimal Threshold (tau): {best_threshold:.6f}")
    print(f"   Max F1-Score:            {best_f1:.4f}")
    
    # 6. Save the Threshold to Disk
    # This allows the 'real-time demo' script to load this exact value later
    with open(f"models/{category}_threshold.txt", "w") as f:
        f.write(str(best_threshold))
        
    return best_threshold

if __name__ == "__main__":
    # Entry point: Optimize threshold for the 'bottle' category
    find_optimal_threshold("bottle")
